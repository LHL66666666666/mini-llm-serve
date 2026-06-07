"""统一训练入口:取代 train_GPT.py / train_Llama.py / train_unified.py。

所有架构(GPT、LLaMA、消融变体)都通过 unified 模型 + 配置开关定义,
训练循环只设置一份。模型由 experiment.model_build.build_model
按 cfg.model(含 preset)构建;损失统一在外部用 CrossEntropyLoss 计算。

用法:
    python train.py --config configs/unified_gpt.yaml
    python train.py --config configs/unified_llama.yaml run_name=llama_base train.max_steps=20000
    python train.py --config configs/llama_gelu.yaml      # 单变量消融
"""
import os
import time
import math
import random
import numpy as np
import torch
from experiment import parse_args_and_load, RunManager, create_train_val_dataloaders
from experiment.model_build import build_model
from experiment.model_spec import resolve_model_cfg


def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(cfg, generator: torch.Generator):
    batch_size = cfg.train.micro_batch_size
    block_size = cfg.data.block_size

    rm = RunManager(cfg)

    # 解析完整 model 配置(填默认 + 校验),vocab_size 后面多处要用
    model_cfg = resolve_model_cfg(cfg.model)
    vocab_size = model_cfg["vocab_size"]

    # tokenizer
    if cfg.data.tokenizer == "gpt2":
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        eos_token_id = tokenizer.encode_single_token("<|endoftext|>")
    else:
        from tokenizer import Tokenizer
        tokenizer = Tokenizer.load(cfg.data.tokenizer)
        eos_token_id = tokenizer.special_tokens["<|endoftext|>"]

    t0 = time.time()
    # 解析训练文件列表:优先 train_files,否则回退单文件 filepath
    files = list(cfg.data.train_files) if cfg.data.train_files else [cfg.data.filepath]
    print(f"[data] count: {len(files)} , shard(shuffle{', True' if cfg.data.shuffle else ', False'})")
    train_loader, val_loader = create_train_val_dataloaders(
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        generator=generator,
        files=files,
        val_files=list(cfg.data.val_files) if cfg.data.val_files else None,
        val_ratio=cfg.data.val_ratio,
        val_shard_index=cfg.data.val_shard_index,
        shuffle=cfg.data.shuffle,
        cache_dir=cfg.data.cache_dir,
        tokenizer_name=cfg.data.tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(f"train samples: {len(train_loader.dataset):,}, val samples: {len(val_loader.dataset):,}")
    print(f"data ready in {time.time() - t0:.2f}s")

    device = torch.device("cuda" if (cfg.train.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    model = build_model(cfg).to(device)

    # ---- torch.compile ----
    if cfg.train.use_compile and hasattr(torch, "compile"):
        print("[compile] applying torch.compile (mode='reduce-overhead')")
        model = torch.compile(model)
    elif cfg.train.use_compile:
        print("[compile] torch.compile not available (torch<2.0), skipping")
    else:
        print("[compile] disabled by config")

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        device_type=device.type,
    )

    accumulation_steps = cfg.train.global_batch_size // cfg.train.micro_batch_size
    assert cfg.train.global_batch_size % cfg.train.micro_batch_size == 0
    tokens_per_step = cfg.train.global_batch_size * block_size
    max_tokens = cfg.train.max_tokens

    # 停止步数:max_steps 为空则由 token 预算反推
    max_steps = cfg.train.max_steps
    if max_steps is None:
        if max_tokens is None:
            raise ValueError("both max_steps and max_tokens are empty")
        max_steps = math.ceil(max_tokens / tokens_per_step)
        print(f"[train] max_steps not set, max_tokens={max_tokens/1e6:.1f}M -> {max_steps} steps")
    # 余弦调度终点默认对齐到 max_steps
    lr_decay_steps = cfg.optim.lr_decay_steps or max_steps

    optimizer.zero_grad()
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    last_log_time = time.time()
    tokens_since_log = 0

    data_iter = iter(train_loader)
    while global_step < max_steps:
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _, _ = model(input_ids=inputs, attention_mask=None, position_ids=None,
                                 past_key_values=None, output_attentions=False, use_cache=False)
            loss = loss_fn(logits.contiguous().view(-1, vocab_size), labels.contiguous().view(-1))
        loss = loss / accumulation_steps
        loss.backward()
        accum_loss += loss.item() * accumulation_steps
        micro_step += 1
        tokens_since_log += batch_size * block_size

        if micro_step % accumulation_steps == 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            lr = get_lr(global_step, cfg.optim.lr, cfg.optim.min_lr, cfg.optim.warmup_steps, lr_decay_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % cfg.train.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.time()
                delta = now - last_log_time
                avg_loss = accum_loss / micro_step
                tps = tokens_since_log / delta
                tokens_seen = global_step * tokens_per_step
                gpu_mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
                print(f"step {global_step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"norm {norm:.3f} | {tps:.0f} tok/s | {tokens_seen/1e6:.2f}M tok | "
                      f"gpu {gpu_mem:.2f}GB")
                rm.log_metrics(global_step,
                               {"loss": avg_loss, "lr": lr, "grad_norm": float(norm),
                                "tokens_per_sec": tps, "gpu_mem_gb": gpu_mem},
                               tokens=tokens_seen, split="train")
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                accum_loss = 0.0
                micro_step = 0
                tokens_since_log = 0
                last_log_time = now

            if global_step % cfg.train.val_interval == 0:
                vt = time.time()
                vloss, vsteps = 0.0, 0
                model.eval()
                with torch.no_grad():
                    for vi, vl in val_loader:
                        vi, vl = vi.to(device), vl.to(device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            vlogits, _, _ = model(input_ids=vi, attention_mask=None, position_ids=None,
                                                  past_key_values=None, output_attentions=False, use_cache=False)
                            l = loss_fn(vlogits.contiguous().view(-1, vocab_size), vl.contiguous().view(-1))
                        vloss += l.item()
                        vsteps += 1
                val_loss = vloss / vsteps
                ppl = math.exp(val_loss)
                tokens_seen = global_step * tokens_per_step
                rm.log_metrics(global_step, {"loss": val_loss, "ppl": ppl},
                               tokens=tokens_seen, split="val")
                print(f"  val: loss {val_loss:.4f} | ppl {ppl:.2f} | {time.time()-vt:.1f}s")
                rm.save_checkpoint(model, optimizer, global_step, tokens_seen,
                                   val_loss=val_loss,
                                   periodic=(global_step % cfg.train.ckpt_interval == 0))
                if global_step % cfg.train.sample_interval == 0:
                    samples = {}
                    for prompt in cfg.train.sample_prompts:
                        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
                        out = model.generate(ids, cfg.train.sample_max_new_tokens,
                                             cfg.train.sample_temperature, cfg.train.sample_top_k)
                        samples[prompt] = tokenizer.decode(out[0].tolist())
                    rm.log_samples(global_step, samples)
                model.train()
                last_log_time = time.time()

            tokens_seen = global_step * tokens_per_step
            if max_tokens and tokens_seen >= max_tokens:
                print(f"reached max_tokens {max_tokens}, stop.")
                break
            if global_step >= max_steps:
                break

    rm.finish()
    print("Training finished.")


if __name__ == "__main__":
    cfg = parse_args_and_load()
    set_seed(cfg.train.seed)
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    print(f"seed: {cfg.train.seed}")
    train(cfg, g)
    # 训练完成后自动关机
    os.system("/usr/bin/shutdown")
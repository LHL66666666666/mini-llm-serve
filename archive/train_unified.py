import torch
from torch import nn
import numpy as np
import time
import tiktoken
import math
from nano_unified import unified_transformer, unified_transformer_config
import random
from experiment import parse_args_and_load, RunManager, create_train_val_dataloaders

# 带预热的余弦衰减学习率调度
def get_lr(it,learning_rate, min_lr, warmup_iters,  lr_decay_iters ):
    """
    :param it: 当前的迭代次数
    :param learning_rate: 初始（最大）学习率
    :param min_lr: 最终衰减到的最低学习率
    :param warmup_iters: 预热阶段的迭代次数
    :param lr_decay_iters: 学习率衰减的总迭代次数（预热结束后开始衰减，到这步时恰好降到 min_lr）
    """
    # 当迭代次数 it 小于预热步数时，学习率从 0 线性增长到 learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 当前步数已经超过了设定的衰减终止步数，学习率直接取最小值 min_lr，保持不变
    if it > lr_decay_iters:
        return min_lr
    # 学习率按照余弦曲线下降
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train(
        cfg,
        generator: torch.Generator,
):
    # 设置 B,T,使用 cfg 中的参数
    batch_size = cfg.train.micro_batch_size  # micro_batch_size 用于单次前向
    block_size = cfg.data.block_size

    # 初始化 RunManager（必须在最开始，创建目录和日志）
    rm = RunManager(cfg)

    # 加载 tokenizer
    if cfg.data.tokenizer == "gpt2":
        # 使用 tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        eos_token_id = tokenizer.encode_single_token("<|endoftext|>")
    else:
        # 若指定自研 tokenizer 文件
        from tokenizer import Tokenizer
        tokenizer = Tokenizer.load(cfg.data.tokenizer)
        eos_token_id = tokenizer.special_tokens["<|endoftext|>"]

    start_time = time.time()
    # 注意：当 vocab_size > 65536 时,自动修改存储数据类型为 np.int32 (已加入自适应模块)
    # 创建 dataloader
    train_loader, val_loader = create_train_val_dataloaders(
        filepath=cfg.data.filepath,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        vocab_size=cfg.model["vocab_size"],  # <-新增,决定 dtype + 缓存隔离
        tokenizer_name=cfg.data.tokenizer,  # <-新增,缓存指纹的一部分
        generator=generator,
        cache_dir=cfg.data.cache_dir,  # <-新增
        block_size=block_size,
        val_ratio=cfg.data.val_ratio,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
    )
    end_time = time.time()
    print(f"train loader length: {len(train_loader)}, validation loader length: {len(val_loader)})")
    print(f"Loading and tokenizing finished, total time: {end_time - start_time:.2f}s")

    device = torch.device("cuda" if (cfg.train.device == "auto" or cfg.train.device == "cuda") and torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()


    # 优化器配置（使用 cfg 中的参数）
    optimizer = model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        device_type=device.type,
    )

    # 梯度累积设置
    accumulation_steps = cfg.train.global_batch_size // cfg.train.micro_batch_size
    assert cfg.train.global_batch_size % cfg.train.micro_batch_size == 0


    # LR 衰减步数
    lr_decay_steps = cfg.optim.lr_decay_steps if cfg.optim.lr_decay_steps is not None else cfg.train.max_steps

    # 训练循环控制变量
    max_steps = cfg.train.max_steps
    max_tokens = cfg.train.max_tokens

    # 记录每个 optimizer step 消耗的 token 数
    tokens_per_step = cfg.train.global_batch_size * block_size

    # 训练
    optimizer.zero_grad()
    global_step = 0  # 已经执行的 optimizer.step 次数
    micro_step = 0  # 当前累积中的 micro‑batch 计数
    accum_loss = 0.0  # 累积的损失（用于求平均值）
    start_time = time.time()
    last_log_time = start_time
    tokens_since_log = 0

    # DataLoader 可以多次循环，直到达到 max_steps
    data_iter = iter(train_loader)  # 可无限循环使用
    while global_step < max_steps:
        # 获取下一个 batch，若 DataLoader 耗尽则重新初始化
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()
        # 混合精度训练
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _, _ = model(
                input_ids=inputs,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )
            #  model 字段是 dict，所以用 cfg.model['vocab_size']
            loss = loss_fn(logits.contiguous().view(-1, cfg.model["vocab_size"]), labels.contiguous().view(-1))
        # 梯度累积
        loss *= 1.0 / accumulation_steps
        loss.backward()
        accum_loss += loss.item() * accumulation_steps  # 恢复成原始 scale
        micro_step += 1
        tokens_since_log += batch_size * block_size
        # 达到累积步数，更新参数
        if micro_step % accumulation_steps == 0:
            # 梯度裁剪，返回梯度范数 norm
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 设置学习率
            lr = get_lr(
                global_step,
                cfg.optim.lr,
                cfg.optim.min_lr,
                cfg.optim.warmup_steps,
                lr_decay_steps
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            # 打印指标
            if global_step % cfg.train.log_interval == 0:
                torch.cuda.synchronize()
                now = time.time()
                delta = now - last_log_time
                avg_loss = accum_loss / (micro_step)  # 这些 micro‑batch 的平均损失
                tokens_per_sec = tokens_since_log / delta
                tokens_seen = global_step * tokens_per_step
                # print 给终端，train.log 里也会有一份
                print(f"step: {global_step:5d}, loss: {avg_loss:.4f}, lr:{lr:.2e}, "
                      f"norm:{norm:.4f}, delta_time: {delta * 1000:.2f}ms, "
                      f"tokens/sec: {tokens_per_sec:.2f}, "
                      f"tokens seen: {tokens_seen / 1e6 :.2f} M")
                # 用 rm 记录训练指标
                rm.log_metrics(global_step,
                               {"loss": avg_loss, "lr": lr, "grad_norm": float(norm),
                                "tokens_per_sec": tokens_per_sec},
                               tokens=tokens_seen, split="train")

                # 重置统计
                accum_loss = 0.0
                micro_step = 0
                tokens_since_log = 0
                last_log_time = now

            # validation
            if global_step % cfg.train.val_interval == 0:
                valid_start_time = time.time()
                valid_loss = 0.0
                valid_steps = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        # 前向 + 损失
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            logits, _, _ = model(
                                input_ids=inputs,
                                attention_mask=None,
                                position_ids=None,
                                past_key_values=None,
                                output_attentions=False,
                                use_cache=False,
                            )
                            loss = loss_fn(logits.contiguous().view(-1, cfg.model["vocab_size"]), labels.contiguous().view(-1))
                        valid_loss += loss.item()
                        valid_steps += 1
                valid_end_time = time.time()
                valid_perplexity = math.exp(valid_loss / valid_steps)
                tokens_seen = global_step * tokens_per_step
                # 记录到 rm
                rm.log_metrics(global_step, {"loss": valid_loss / valid_steps, "ppl": valid_perplexity},
                               tokens=tokens_seen, split="val")

                print(f"valid steps: {valid_steps}, "
                      f"valid loss: {valid_loss / valid_steps:.4f}, "
                      f"valid perplexity: {valid_perplexity:.4f}"
                      f"delta_time: {valid_end_time - valid_start_time:.2f}s")

                # 保存 checkpoint
                rm.save_checkpoint(model, optimizer, global_step, tokens_seen,
                                   val_loss=valid_loss / valid_steps,
                                   periodic=(global_step % cfg.train.ckpt_interval == 0))
                # 生成样本
                if global_step % cfg.train.sample_interval == 0:
                    # 生成时模型已经在 eval 状态
                    samples = {}
                    for prompt in cfg.train.sample_prompts:
                        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
                        out = model.generate(ids,
                                             cfg.train.sample_max_new_tokens,
                                             cfg.train.sample_temperature,
                                             cfg.train.sample_top_k)
                        text = tokenizer.decode(out[0].tolist())
                        samples[prompt] = text
                    rm.log_samples(global_step, samples)
                model.train()  # 切回训练模式

                last_log_time = valid_end_time

            # 停止条件检查
            tokens_seen = global_step * tokens_per_step
            if max_tokens and tokens_seen >= max_tokens:
                print(f"Reached max_tokens budget ({max_tokens}), stopping.")
                break
            if global_step >= max_steps:
                print(f"Reached max_steps ({max_steps}), stopping.")
                break
    # 训练结束
    rm.finish()
    print("Training finished.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用了多GPU

def build_model(config):
    if config.train.arch == "unified":
        return unified_transformer(unified_transformer_config(**config.model))
    else:
        raise ValueError(f"Unsupported arch: {config.train.arch}")

if __name__ == "__main__":
    # 解析配置
    cfg = parse_args_and_load()
    # 设置随机种子
    set_seed(cfg.train.seed)
    # 创建固定 generator 用于 DataLoader
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    print(f"Random seed: {cfg.train.seed}")
    # 调用训练函数，传入配置和 generator
    train(cfg, g)
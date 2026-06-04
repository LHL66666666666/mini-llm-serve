"""Run 管理器:一次实验 = 一个目录,把"复现/解释/定位"需要的东西全部落盘。

职责
----
- 创建 runs/<timestamp>_<run_name>/ 目录结构
- 存 config.yaml、git commit/diff、环境信息、pip freeze
- 把 stdout tee 进 train.log(无需改动现有 print)
- 结构化指标日志:同时写 metrics.jsonl(程序可读,跨 run 对比) 和 TensorBoard(可视化),可选 W&B
- checkpoint:保存 model/optimizer/step/tokens/config/RNG,支持 last/best/周期性,支持续训
- 固定 prompt 生成样本写入 samples.md;初始化时生成 report.md 模板

依赖:torch(必需)、tensorboard(可选)、wandb(可选)。
"""

import os
import sys
import json
import time
import random
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", "ignore").strip()
    except Exception:
        return ""


class _Tee:
    """把写到 stdout 的内容同时写进文件,这样现有的 print 全部进 train.log。"""
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


class RunManager:
    def __init__(self, cfg):
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.out_root, f"{ts}_{cfg.run_name}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 0) stdout tee -> train.log
        self._logfile = open(os.path.join(self.run_dir, "train.log"), "w", encoding="utf-8")
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._logfile)

        # 1) 配置快照
        from experiment.config import save_config
        save_config(cfg, os.path.join(self.run_dir, "config.yaml"))

        # 2) git 状态(连未提交的 diff 一起存,才真正可复现)
        commit = _run_cmd(["git", "rev-parse", "HEAD"])
        branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = _run_cmd(["git", "status", "--porcelain"])
        with open(os.path.join(self.run_dir, "git_commit.txt"), "w", encoding="utf-8") as f:
            f.write(f"commit: {commit}\nbranch: {branch}\ndirty: {'yes' if dirty else 'no'}\n")
        diff = _run_cmd(["git", "diff"])
        if diff:
            with open(os.path.join(self.run_dir, "git_diff.patch"), "w", encoding="utf-8") as f:
                f.write(diff)

        # 3) 环境
        self._dump_env()

        # 4) metrics.jsonl(每行一条记录,程序可读)
        self._jsonl = open(os.path.join(self.run_dir, "metrics.jsonl"), "w", encoding="utf-8")

        # 5) TensorBoard / W&B
        self.tb = None
        if cfg.use_tensorboard and _HAS_TB:
            self.tb = SummaryWriter(os.path.join(self.run_dir, "tb"))
        elif cfg.use_tensorboard and not _HAS_TB:
            print("[RunManager] 未安装 tensorboard,跳过 TB(pip install tensorboard)")
        self.wandb = None
        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))
                self.wandb = wandb
            except Exception as e:
                print(f"[RunManager] W&B 初始化失败,跳过:{e}")

        # 6) report.md 模板(逼自己事后写三段)
        self._write_report_template()

        self.t0 = time.time()
        self.best_val = float("inf")
        print(f"[RunManager] run dir: {self.run_dir}")

    # ---------- 环境 ----------
    def _dump_env(self):
        lines = [
            f"python: {platform.python_version()}",
            f"platform: {platform.platform()}",
            f"torch: {torch.__version__}",
            f"cuda available: {torch.cuda.is_available()}",
            f"cuda(torch built): {torch.version.cuda}",
        ]
        if torch.cuda.is_available():
            lines.append(f"gpu count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"  gpu[{i}]: {torch.cuda.get_device_name(i)}")
        with open(os.path.join(self.run_dir, "env.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        freeze = _run_cmd([sys.executable, "-m", "pip", "freeze"])
        if freeze:
            with open(os.path.join(self.run_dir, "requirements_lock.txt"), "w", encoding="utf-8") as f:
                f.write(freeze + "\n")

    def _write_report_template(self):
        c = self.cfg
        path = os.path.join(self.run_dir, "../report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f"# {c.run_name}\n\n"
                f"- arch: {c.train.arch}\n"
                f"- notes: {c.notes}\n"
                f"- model: {c.model}\n"
                f"- global_batch(seq): {c.train.global_batch_size}, "
                f"block_size: {c.data.block_size}, "
                f"global_batch(tokens): {c.train.global_batch_size * c.data.block_size}\n"
                f"- max_steps: {c.train.max_steps}, max_tokens: {c.train.max_tokens}\n"
                f"- tokenizer: {c.data.tokenizer}\n\n"
                f"## 1. 做了什么改动\n\n(相对哪个 baseline,只改了哪一个变量)\n\n"
                f"## 2. 观察到什么现象\n\n(final val loss / ppl、曲线特征、生成样本)\n\n"
                f"## 3. 怎么解释\n\n(为什么会这样,下一步)\n"
            )

    # ---------- 指标 ----------
    def log_metrics(self, step, metrics: dict, tokens=None, split="train"):
        rec = {"step": step, "tokens": tokens, "wall": round(time.time() - self.t0, 2), "split": split}
        rec.update(metrics)
        self._jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(f"{split}/{k}", v, step)
                if tokens is not None:
                    # 额外按 token 记一份:不同 vocab/block_size 下唯一可比的横轴
                    self.tb.add_scalar(f"{split}_by_tokens/{k}", v, tokens)
        if self.wandb:
            payload = {f"{split}/{k}": v for k, v in metrics.items()}
            payload["step"] = step
            if tokens is not None:
                payload["tokens"] = tokens
            self.wandb.log(payload)

    def log_samples(self, step, samples: dict):
        """samples: {prompt: generated_text}"""
        path = os.path.join(self.run_dir, "samples.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## step {step}\n")
            for prompt, text in samples.items():
                f.write(f"\n**prompt:** `{prompt}`\n\n```\n{text}\n```\n")
        if self.tb:
            block = "\n\n".join(f"[{p}] -> {t}" for p, t in samples.items())
            self.tb.add_text("samples", block, step)

    # ---------- checkpoint ----------
    def save_checkpoint(self, model, optimizer, step, tokens, val_loss=None, periodic=False):
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens": tokens,
            "val_loss": val_loss,
            "config": asdict(self.cfg),
            "torch_version": torch.__version__,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(ckpt, os.path.join(self.ckpt_dir, "last.pt"))
        if periodic:
            torch.save(ckpt, os.path.join(self.ckpt_dir, f"step_{step:06d}.pt"))
        if val_loss is not None and val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(ckpt, os.path.join(self.ckpt_dir, "best.pt"))
            print(f"[RunManager] new best val_loss={val_loss:.4f} @ step {step}")

    @staticmethod
    def load_checkpoint(path, model, optimizer=None, restore_rng=False, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if restore_rng and ckpt.get("rng"):
            rng = ckpt["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        return ckpt

    def finish(self):
        if self.tb:
            self.tb.flush()
            self.tb.close()
        if self.wandb:
            self.wandb.finish()
        self._jsonl.close()
        sys.stdout = self._orig_stdout
        self._logfile.close()

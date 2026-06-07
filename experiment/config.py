"""实验配置系统:dataclass 定义 + YAML 双向序列化 + 命令行点路径覆盖。

设计目标
--------
1. 配置是唯一事实来源:所有超参都从这里来,训练循环里不出现魔法数字。
2. 一份配置能完整 round-trip 到 YAML,跑前存档到 run 目录,保证可复现。
3. 支持命令行点路径覆盖(如 train.max_steps=5000),方便做 sweep / 消融,
   一次只改一个变量,符合控制变量法。
4. 模型结构字段用 dict 存(model 字段),按 arch 交给已有的 GPTConfig /
   LlamaConfig 去实例化,避免在这里重复定义模型结构。

只依赖 pyyaml,不引入 torch,保持轻量、可单独测试。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import yaml


@dataclass
class DataConfig:
    filepath: str = "fineweb_edu_q3_20M.jsonl"              # 向后兼容:单文件;若 train_files 非空则忽略
    train_files: List[str] = field(default_factory=list)    # 多 shard 显式列表(按此顺序读)
    val_files: List[str] = field(default_factory=list)      # 可选:独立验证文件;为空则从 val_shard 切
    val_shard_index: int = -1           # 从哪个训练 shard 末尾切 val(默认 -1 = 最后一个)
    shuffle: bool = False               # 默认顺序读(读完一个 shard 再下一个);True=全局打乱
    tokenizer: str = "gpt2"            # "gpt2" 或自研 tokenizer.json 的路径
    block_size: int = 256             # 序列长度 T
    val_ratio: float = 0.05
    cache_dir: str = "data/cache"     # 预 tokenize 缓存目录(避免每次重新编码)
    num_workers: int = 0


@dataclass
class OptimConfig:
    lr: float = 3e-4                  # peak lr
    min_lr: float = 3e-5
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_decay_steps: Optional[int] = None   # None 表示 = train.max_steps,训练时填充


@dataclass
class TrainConfig:
    arch: str = "gpt"                 # "gpt" | "llama"
    use_compile: bool = True          # 默认开启 torch.compile
    micro_batch_size: int = 8         # 单次前向的 batch(受显存限制)
    global_batch_size: int = 32       # 梯度累积后的有效 batch(序列数);必须能被 micro 整除
    max_steps: Optional[int] = None             # 以 optimizer.step 计;留空则由 max_tokens 反推
    max_tokens: Optional[int] = None  # 可选:按 token 预算停止(设了就覆盖 max_steps 作为停止条件)
    dtype: str = "bfloat16"
    device: str = "auto"
    seed: int = 42
    # 记录 / 验证 / 存档节奏(都以 optimizer.step 为单位)
    log_interval: int = 50
    val_interval: int = 100
    ckpt_interval: int = 500
    sample_interval: int = 500
    # 定性生成:跨 checkpoint 用固定 prompt 观察"模型在学什么"
    sample_prompts: List[str] = field(default_factory=lambda: ["The", "Once upon a time"])
    sample_max_new_tokens: int = 100
    sample_temperature: float = 1.0
    sample_top_k: Optional[int] = 50


@dataclass
class TokenizerTrainConfig:
    vocab_size: int = 32768
    special_tokens: List[str] = field(default_factory=lambda: [
        "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>",
        "<|fim_suffix|>", "<|endofprompt|>"
    ])
    pattern: str = "GPT4"           # "GPT2", "GPT4", "char"
    corpus_path: str = ""           # 训练分词器所用的纯文本文件
    sample_size: Optional[int] = None
    num_chunks: int = 64
    num_processes: int = 8
    save_name: str = "tokenizer"

@dataclass
class ExperimentConfig:
    run_name: str = "gpt_baseline"
    out_root: str = "experiments/training"
    notes: str = ""                   # 一句话:这次相对 baseline 改了什么(写进 report)
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "nano-llm"
    model: dict = field(default_factory=dict)   # arch 专属字段,如 {n_layer:4, hidden_size:256, ...}
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)






# --------- 序列化 ---------

def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)


def config_from_dict(d: dict) -> ExperimentConfig:
    d = dict(d or {})
    sub = {}
    if "data" in d:
        sub["data"] = DataConfig(**(d.pop("data") or {}))
    if "optim" in d:
        sub["optim"] = OptimConfig(**(d.pop("optim") or {}))
    if "train" in d:
        sub["train"] = TrainConfig(**(d.pop("train") or {}))
    return ExperimentConfig(**d, **sub)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return config_from_dict(d)


# --------- 命令行覆盖:python train.py --config x.yaml train.max_steps=5000 model.n_layer=6 ---------

def _cast(raw: str, ref):
    """把字符串 raw 转成与参考值 ref 同类型;ref 为 None 时自动推断。"""
    if ref is None:
        for conv in (int, float):
            try:
                return conv(raw)
            except ValueError:
                pass
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if raw.lower() in ("none", "null"):
            return None
        return raw
    if isinstance(ref, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(ref, int):
        return int(raw)
    if isinstance(ref, float):
        return float(raw)
    return raw


def apply_overrides(cfg: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """overrides 形如 ["train.max_steps=5000", "model.n_layer=6"]。"""
    for ov in overrides:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"非法 override(缺少 =):{ov}")
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(p, {})
            else:
                obj = getattr(obj, p)
        leaf = parts[-1]
        if isinstance(obj, dict):
            obj[leaf] = _cast(val, obj.get(leaf))
        else:
            setattr(obj, leaf, _cast(val, getattr(obj, leaf)))
    return cfg


def parse_args_and_load(argv: Optional[List[str]] = None) -> ExperimentConfig:
    """统一入口:--config 指定 yaml,其余 a.b=c 形式的参数作为覆盖。"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args(argv)
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg

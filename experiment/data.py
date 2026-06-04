"""共享数据加载模块:GPT 和 LLaMA 训练脚本共用,消除重复代码。

相对原 create_train_val_dataloaders 的改动:
1. 用 data_cache 预 tokenize,命中缓存秒回,不再每次全量重编码。
2. dtype 按 vocab_size 自动选(uint16/uint32),不再静默强转 uint16。
3. Dataset 直接 memmap,不强制 ascontiguousarray 拷贝整份(省内存)。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from experiment.data_cache import load_or_build_token_cache


class PackedTokenDataset(Dataset):
    """把一维 token 流按 block_size 切成 (x, y) 对。接收 memmap 或 ndarray。"""

    def __init__(self, tokens, block_size: int):
        super().__init__()
        self.tokens = tokens  # memmap 或 np.ndarray,不在此处拷贝
        self.block_size = block_size
        self.num_samples = (len(tokens) - 1) // block_size
        print(f"Sub-dataset ready: {len(tokens)/1e6:.2f}M tokens, "
              f"{self.num_samples:,} samples (block_size={block_size}, dtype={tokens.dtype})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        # 取一段并升到 int64(embedding 需要 long);memmap 切片是惰性的
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]


def create_train_val_dataloaders(
    filepath: str,
    tokenizer,
    eos_token_id: int,
    vocab_size: int,
    generator: torch.Generator,
    cache_dir: str = "data/cache",
    tokenizer_name: str = "tokenizer",
    block_size: int = 1024,
    val_ratio: float = 0.05,
    batch_size: int = 8,
    num_workers: int = 0,
):
    # 1) 预 tokenize(带缓存)
    tokens = load_or_build_token_cache(
        filepath=filepath,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        declared_name=tokenizer_name,
        add_eos=True,
    )
    print(f"Total tokens: {len(tokens)/1e6:.2f}M")

    # 2) 顺序切分 train/val,对齐 block_size
    val_size = int(len(tokens) * val_ratio)
    val_size = (val_size // block_size) * block_size
    train_size = len(tokens) - val_size
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    print(f"Train: {train_size/1e6:.2f}M tokens | Val: {val_size/1e6:.2f}M tokens")

    train_ds = PackedTokenDataset(train_tokens, block_size)
    val_ds = PackedTokenDataset(val_tokens, block_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=False,
    )
    return train_loader, val_loader

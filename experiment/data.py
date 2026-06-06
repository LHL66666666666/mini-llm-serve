"""共享数据加载模块:多 shard 顺序读 + token 预算驱动 + 缓存复用 + 低内存。

设计(new)
--------------------
- 多个训练 shard:每个 shard 仍走 data_cache 单独缓存(缓存 key 含文件路径,
  天然每文件一份),编码阶段流式写盘,不堆大 list,避免 OOM。
- 训练时用 map-style 的 ConcatShardDataset 把各 shard 的 memmap 顺序拼接,
  DataLoader(shuffle=False) 即得到"shard0 全部 -> shard1 全部 -> ..."的顺序;
  一轮读完触发 StopIteration,train.py 现有的重建 iter 逻辑实现"读完再循环"。
- 停止条件由 token 预算控制(train.py 里按 max_tokens 停),数据集本身可被无限循环。
- 内存:memmap 惰性分页,8 个 shard 全开也只占虚拟地址;索引只存每 shard 的块数。
- 验证集:从指定 shard(默认最后一个)末尾切固定 val,训练范围跳过这段,保证不重叠;
  也支持用 val_files 显式指定独立验证文件。
"""

import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from experiment.data_cache import load_or_build_token_cache


# ---------------- 纯函数:索引数学(无 torch,便于单测) ----------------

def build_cumulative(block_counts):
    """各 shard 的可用块数 -> 前缀和边界。返回 (cum, total)。

    cum[i] 是前 i 个 shard 的累计块数;total 是总块数。
    """
    cum = [0]
    for c in block_counts:
        cum.append(cum[-1] + c)
    return cum, cum[-1]


def locate(global_block_idx, cum):
    """全局块下标 -> (shard_idx, local_block_idx)。"""
    # cum 形如 [0, b0, b0+b1, ...];找 global 落在哪个区间
    shard = bisect.bisect_right(cum, global_block_idx) - 1
    local = global_block_idx - cum[shard]
    return shard, local


def shard_block_count(n_tokens, block_size, block_start=0, block_end=None):
    """某 shard 在 [block_start, block_end) 范围内可切出的样本数。"""
    total_blocks = (n_tokens - 1) // block_size
    if block_end is None or block_end > total_blocks:
        block_end = total_blocks
    return max(0, block_end - block_start)


# ---------------- map-style 数据集 ----------------

class ConcatShardDataset(Dataset):
    """把多个 shard 的 token memmap 按给定块区间顺序拼接成一个数据集。

    shards: list of dict,每个含:
        - tokens: 一维 memmap/ndarray
        - block_start, block_end: 该 shard 参与的块区间(用于排除 val 尾巴)
    block_size: 序列长度 T
    """

    def __init__(self, shards, block_size):
        super().__init__()
        self.block_size = block_size
        self.shards = shards
        counts = [s["block_end"] - s["block_start"] for s in shards]
        self.cum, self.total = build_cumulative(counts)
        tok = sum(c * block_size for c in counts)
        print(f"ConcatShardDataset: {len(shards)} shards, {self.total:,} samples, "
              f"~{tok/1e6:.2f}M train tokens (block_size={block_size})")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        shard_idx, local = locate(idx, self.cum)
        s = self.shards[shard_idx]
        block = s["block_start"] + local
        start = block * self.block_size
        end = start + self.block_size + 1
        chunk = np.asarray(s["tokens"][start:end], dtype=np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]


class PackedTokenDataset(Dataset):
    """单段 token 流按 block_size 切 (x, y)。用于固定 val 集(小、全量评估)。"""

    def __init__(self, tokens, block_size, block_start=0, block_end=None):
        super().__init__()
        self.tokens = tokens
        self.block_size = block_size
        self.block_start = block_start
        total = (len(tokens) - 1) // block_size
        self.block_end = total if (block_end is None or block_end > total) else block_end
        self.num_samples = max(0, self.block_end - self.block_start)
        print(f"PackedTokenDataset: {len(tokens)/1e6:.2f}M tokens, "
              f"{self.num_samples:,} samples [{self.block_start}:{self.block_end}]")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        block = self.block_start + idx
        start = block * self.block_size
        end = start + self.block_size + 1
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]


# ---------------- 编排:缓存所有 shard + 切 val + 建 loader ----------------

def _cache_all_shards(files, tokenizer, eos_token_id, vocab_size, cache_dir, tokenizer_name):
    """对每个 shard 建/读缓存,返回 [(path, memmap, n_tokens), ...](保持文件顺序)。"""
    shards = []
    for fp in files:
        mm = load_or_build_token_cache(
            filepath=fp, tokenizer=tokenizer, eos_token_id=eos_token_id,
            vocab_size=vocab_size, cache_dir=cache_dir,
            declared_name=tokenizer_name, add_eos=True,
        )
        shards.append((fp, mm, len(mm)))
    return shards


def create_train_val_dataloaders(
    tokenizer,
    eos_token_id: int,
    vocab_size: int,
    generator: torch.Generator,
    files=None,
    filepath: str = None,        # 向后兼容:单文件
    val_files=None,
    val_ratio: float = 0.05,
    val_shard_index: int = -1,   # 从哪个训练 shard 末尾切 val(默认最后一个)
    shuffle: bool = False,       # 默认顺序读;True=全局打乱(牺牲严格 shard 顺序)
    cache_dir: str = "data/cache",
    tokenizer_name: str = "tokenizer",
    block_size: int = 1024,
    batch_size: int = 8,
    num_workers: int = 0,
):
    # 解析文件列表(向后兼容单文件)
    if files is None:
        if filepath is None:
            raise ValueError("必须提供 files(列表)或 filepath(单文件)")
        files = [filepath]
    assert len(files) >= 1

    train_shards_raw = _cache_all_shards(files, tokenizer, eos_token_id, vocab_size,
                                         cache_dir, tokenizer_name)

    # ---- 确定 val 来源 ----
    val_tokens = None
    val_block_range = (0, None)
    # 训练各 shard 默认用满
    train_specs = [{"tokens": mm, "block_start": 0, "block_end": (len(mm) - 1) // block_size}
                   for (_, mm, _) in train_shards_raw]

    if val_files:
        # 独立验证文件:单独缓存,训练 shard 全部用满
        vshards = _cache_all_shards(val_files, tokenizer, eos_token_id, vocab_size,
                                    cache_dir, tokenizer_name)
        # val 若多文件,拼成 ConcatShardDataset;这里简单起见取第一个(或可扩展)
        val_tokens = vshards[0][1]
        print(f"[data] val 来自独立文件: {val_files[0]}")
    else:
        # 从指定训练 shard 末尾切 val
        vi = val_shard_index if val_shard_index >= 0 else len(train_specs) + val_shard_index
        assert 0 <= vi < len(train_specs), f"val_shard_index 越界: {val_shard_index}"
        mm = train_shards_raw[vi][1]
        total_blocks = (len(mm) - 1) // block_size
        val_blocks = int(total_blocks * val_ratio)
        val_blocks = max(1, val_blocks)
        head_blocks = total_blocks - val_blocks
        # 训练:该 shard 只用 head;val:该 shard 的尾部
        train_specs[vi]["block_end"] = head_blocks
        val_tokens = mm
        val_block_range = (head_blocks, total_blocks)
        print(f"[data] val 从 shard#{vi} 末尾切: 训练用块 [0:{head_blocks}], "
              f"val 用块 [{head_blocks}:{total_blocks}]")

    # ---- 建数据集 + loader ----
    train_ds = ConcatShardDataset(train_specs, block_size)
    val_ds = PackedTokenDataset(val_tokens, block_size,
                                block_start=val_block_range[0], block_end=val_block_range[1])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=False,
    )
    return train_loader, val_loader
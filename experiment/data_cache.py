"""预 tokenize 缓存层:把语料编码一次,存成 memmap 的 .bin,后续训练直接读。

为什么需要
----------
现状是每次训练都重新 encode 整个 20M 语料。用 tiktoken 还能忍,但做 vocab
sweep 时要换自研的纯 Python tokenizer,且要跑 4 个 vocab_size × 2 个架构 = 8 次,
每次都全量重编码会非常慢。缓存一次即可全部复用。

正确性关键(否则 vocab 实验直接失效)
-----------------------------------
缓存 key 必须包含 tokenizer 的指纹。4k/8k/16k/32k 等编码出的 token 流完全不同,
若 key 只用 filepath,第二个 tokenizer 会错误命中第一个的缓存。这里用
fingerprint(tokenizer) + 文件路径 + mtime + add_eos 共同决定缓存文件名。

dtype 选择
----------
- vocab_size <= 65536: uint16(省一半内存/磁盘)
- 否则:               uint32
并在写入时校验最大 token id 没有溢出(原代码强转 uint16 是静默回绕,危险)。

缓存文件:
  <cache_dir>/<hash>.bin    # 扁平 token 数组
  <cache_dir>/<hash>.json   # 元信息:dtype, n_tokens, tokenizer指纹, 源文件等
"""

import os
import json
import time
import hashlib
import numpy as np


def _tokenizer_fingerprint(tokenizer, declared_name: str) -> dict:
    """为缓存 key 生成 tokenizer 指纹。

    - tiktoken: 用 encoding name + n_vocab。
    - 自研 Tokenizer: 用 vocab_size + pattern + merges 数量 + 一段 merges 的哈希。
    """
    fp = {"declared": declared_name}
    # tiktoken
    if hasattr(tokenizer, "n_vocab") and not hasattr(tokenizer, "merge"):
        fp["kind"] = "tiktoken"
        fp["name"] = getattr(tokenizer, "name", declared_name)
        fp["n_vocab"] = int(tokenizer.n_vocab)
        return fp
    # 自研 Tokenizer
    fp["kind"] = "custom"
    fp["vocab_size"] = int(getattr(tokenizer, "vocab_size", 0))
    fp["pattern"] = getattr(tokenizer, "pattern", "")
    merge = getattr(tokenizer, "merge", {}) or {}
    fp["n_merges"] = len(merge)
    # 对 merges 内容取稳定哈希(顺序无关:按 new_id 排序)
    h = hashlib.sha1()
    for pair, new_id in sorted(merge.items(), key=lambda kv: kv[1]):
        h.update(f"{pair[0]},{pair[1]}->{new_id};".encode())
    fp["merges_sha1"] = h.hexdigest()
    return fp


def _cache_key(fp: dict, filepath: str, add_eos: bool) -> str:
    st = os.stat(filepath)
    payload = {
        "fp": fp,
        "file": os.path.abspath(filepath),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "add_eos": add_eos,
        "v": 2,  # 缓存格式版本,改了编码逻辑时 +1 即可整体失效
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _pick_dtype(vocab_size: int):
    if vocab_size <= 2 ** 16:
        return np.uint16
    elif vocab_size <= 2 ** 32:
        return np.uint32
    raise ValueError(f"vocab_size {vocab_size} 过大")


def load_or_build_token_cache(
    filepath: str,
    tokenizer,
    eos_token_id: int,
    vocab_size: int,
    cache_dir: str,
    declared_name: str = "tokenizer",
    add_eos: bool = True,
    text_field: str = "text",
    flush_tokens: int = 2_000_000,
) -> np.ndarray:
    """返回一维 token 数组(memmap,只读)。命中缓存则秒回,否则编码并落盘。"""
    os.makedirs(cache_dir, exist_ok=True)
    fp = _tokenizer_fingerprint(tokenizer, declared_name)
    key = _cache_key(fp, filepath, add_eos)
    bin_path = os.path.join(cache_dir, f"{key}.bin")
    meta_path = os.path.join(cache_dir, f"{key}.json")
    dtype = _pick_dtype(vocab_size)

    # 命中
    if os.path.exists(bin_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[cache] HIT {bin_path} ({meta['n_tokens']/1e6:.2f}M tokens, dtype={meta['dtype']})")
        return np.memmap(bin_path, dtype=np.dtype(meta["dtype"]), mode="r")

    # 未命中:流式编码并分块写盘
    # 关键:不再把整文件 token 堆成一个 Python list(2000万 token 的 list ~500MB+,
    # 多 shard 会 OOM)。改为每攒够 flush_tokens 就转 numpy 写一次盘,内存上限恒定。
    print(f"[cache] MISS -> encoding {filepath} with {fp.get('kind')} (vocab_size={vocab_size})")
    t0 = time.time()
    dtype_max = int(np.iinfo(dtype).max)
    n_tokens = 0
    max_id_seen = 0
    n_docs = 0
    buf = []  # 临时 token 缓冲,定期 flush

    tmp = bin_path + ".tmp"
    try:
        with open(filepath, "r", encoding="utf-8") as fin, open(tmp, "wb") as fout:
            def flush():
                nonlocal buf, max_id_seen, n_tokens
                if not buf:
                    return
                chunk_max = max(buf)
                if chunk_max > max_id_seen:
                    max_id_seen = chunk_max
                # 溢出校验:绝不静默回绕(写盘前拦下)
                if chunk_max >= dtype_max:
                    raise ValueError(
                        f"max token id {chunk_max} 超出 dtype {dtype.__name__} 上限 "
                        f"{dtype_max};请提高 vocab_size 对应的 dtype。"
                    )
                arr = np.asarray(buf, dtype=dtype)
                arr.tofile(fout)  # 追加写到当前文件位置
                n_tokens += arr.size
                buf = []

            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get(text_field, "")
                if not text:
                    continue
                ids = tokenizer.encode(text)
                buf.extend(ids)
                if add_eos:
                    buf.append(eos_token_id)
                n_docs += 1
                if len(buf) >= flush_tokens:
                    flush()
            flush()  # 收尾
        os.replace(tmp, bin_path)  # 原子替换,避免中断留下半截缓存
    except BaseException:
        # 失败时清理临时文件,不污染缓存目录
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    meta = {
        "n_tokens": int(n_tokens),
        "n_docs": n_docs,
        "dtype": np.dtype(dtype).name,
        "max_id_seen": int(max_id_seen),
        "vocab_size": int(vocab_size),
        "add_eos": add_eos,
        "fingerprint": fp,
        "source": os.path.abspath(filepath),
        "encode_seconds": round(time.time() - t0, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[cache] built {bin_path} ({n_tokens / 1e6:.2f}M tokens, dtype={meta['dtype']}, "
          f"{meta['encode_seconds']}s)")
    return np.memmap(bin_path, dtype=dtype, mode="r")

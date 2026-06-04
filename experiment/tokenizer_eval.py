"""第三节:tokenizer 评估指标。

指标
----
- 压缩率:bytes/token、chars/token(直接决定同样 token 预算能塞多少信息)
- 平均 token 长度(字节)、token 长度分布(直方图数据)
- 编码/解码速度:tokens/sec、MB/sec
- roundtrip 正确性:decode(encode(x)) == x,必须 100%
- byte-fallback 比例:输出里落在前 256(单字节 token)的比例,偏高说明 vocab 没覆盖好
- vocab 利用率:held-out 文本里实际用到的 token 种类 / vocab_size

公平对比的前提:所有 tokenizer 必须在**同一份固定的 held-out 文本**上评估。
本模块只负责"给定一个 tokenizer + 一段文本,算出指标",sweep 脚本负责对齐文本。

输出:
- metrics.json:所有数值指标
- samples.md:几个固定 prompt 的切分可视化(定性观察常见词/数字/标点怎么切)
"""

import os
import json
import time
import unicodedata
from collections import Counter


# 几个固定的定性观察 prompt:覆盖常见词、数字、标点、代码、罕见词
DEFAULT_QUALITATIVE = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, revenue grew by 17.5% to $1,234,567.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    "antidisestablishmentarianism",
    "Hello, world!!! :)  multiple   spaces\tand\ttabs",
]


def _vocab_size_of(tokenizer) -> int:
    if hasattr(tokenizer, "vocab_size"):
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "n_vocab"):
        return int(tokenizer.n_vocab)
    return -1


def _token_byte_len(tokenizer, tok_id: int) -> int:
    """返回某个 token id 对应的字节长度;拿不到则返回 None。"""
    vocab = getattr(tokenizer, "vocab", None)
    if vocab is not None and tok_id in vocab:
        return len(vocab[tok_id])
    # tiktoken
    if hasattr(tokenizer, "decode_single_token_bytes"):
        try:
            return len(tokenizer.decode_single_token_bytes(tok_id))
        except Exception:
            return None
    return None


def compute_metrics(tokenizer, text: str, declared_name: str = "tokenizer",
                    n_speed_repeat: int = 1) -> dict:
    """在给定 held-out 文本上计算全部数值指标。"""
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))

    # 编码速度
    t0 = time.time()
    for _ in range(n_speed_repeat):
        ids = tokenizer.encode(text)
    enc_time = (time.time() - t0) / max(1, n_speed_repeat)

    n_tokens = len(ids)

    # 解码速度 + roundtrip
    t0 = time.time()
    decoded = tokenizer.decode(ids)
    dec_time = time.time() - t0
    roundtrip_ok = decoded == text

    # 压缩率
    bytes_per_tok = n_bytes / n_tokens if n_tokens else 0.0
    chars_per_tok = n_chars / n_tokens if n_tokens else 0.0

    # token 长度分布(字节)
    len_counter = Counter()
    total_known_len = 0
    n_known = 0
    for tid in ids:
        bl = _token_byte_len(tokenizer, tid)
        if bl is not None:
            len_counter[bl] += 1
            total_known_len += bl
            n_known += 1
    avg_tok_byte_len = total_known_len / n_known if n_known else None
    # 直方图:长度 -> 占比
    length_hist = {str(k): len_counter[k] / n_tokens for k in sorted(len_counter)} if n_tokens else {}

    # byte-fallback 比例:落在前 256 的 token(单字节)
    n_bytefallback = sum(1 for t in ids if t < 256)
    byte_fallback_ratio = n_bytefallback / n_tokens if n_tokens else 0.0

    # vocab 利用率
    used = len(set(ids))
    vsz = _vocab_size_of(tokenizer)
    vocab_util = used / vsz if vsz > 0 else None

    return {
        "tokenizer": declared_name,
        "vocab_size": vsz,
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "n_tokens": n_tokens,
        "bytes_per_token": round(bytes_per_tok, 4),
        "chars_per_token": round(chars_per_tok, 4),
        "avg_token_byte_len": round(avg_tok_byte_len, 4) if avg_tok_byte_len else None,
        "length_hist": length_hist,
        "byte_fallback_ratio": round(byte_fallback_ratio, 6),
        "vocab_used": used,
        "vocab_utilization": round(vocab_util, 6) if vocab_util else None,
        "roundtrip_ok": roundtrip_ok,
        "encode_tokens_per_sec": round(n_tokens / enc_time, 1) if enc_time > 0 else None,
        "encode_MB_per_sec": round(n_bytes / 1e6 / enc_time, 3) if enc_time > 0 else None,
        "decode_tokens_per_sec": round(n_tokens / dec_time, 1) if dec_time > 0 else None,
    }


def qualitative_samples(tokenizer, prompts=None) -> dict:
    """返回 {prompt: [token的可读表示, ...]},用于观察切分粒度。"""
    prompts = prompts or DEFAULT_QUALITATIVE
    out = {}
    for p in prompts:
        ids = tokenizer.encode(p)
        pieces = []
        for tid in ids:
            vocab = getattr(tokenizer, "vocab", None)
            b = None
            if vocab is not None and tid in vocab:
                b = vocab[tid]
            elif hasattr(tokenizer, "decode_single_token_bytes"):
                try:
                    b = tokenizer.decode_single_token_bytes(tid)
                except Exception:
                    b = None
            piece = b.decode("utf-8", errors="replace") if b is not None else f"<{tid}>"
            # 把空白可视化
            piece = piece.replace(" ", "·").replace("\n", "\\n").replace("\t", "\\t")
            pieces.append(piece)
        out[p] = pieces
    return out


def evaluate_tokenizer(tokenizer, heldout_text: str, out_dir: str,
                       declared_name: str = "tokenizer",
                       prompts=None, n_speed_repeat: int = 1) -> dict:
    """完整评估并写盘:metrics.json + samples.md。返回 metrics dict。"""
    os.makedirs(out_dir, exist_ok=True)
    metrics = compute_metrics(tokenizer, heldout_text, declared_name, n_speed_repeat)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    samples = qualitative_samples(tokenizer, prompts)
    with open(os.path.join(out_dir, "samples.md"), "w", encoding="utf-8") as f:
        f.write(f"# {declared_name} 切分样本\n\n")
        f.write(f"vocab_size={metrics['vocab_size']}, "
                f"bytes/token={metrics['bytes_per_token']}, "
                f"chars/token={metrics['chars_per_token']}, "
                f"byte_fallback={metrics['byte_fallback_ratio']}\n\n")
        for p, pieces in samples.items():
            f.write(f"**输入:** `{p}`\n\n")
            f.write(f"**{len(pieces)} tokens:** " + " | ".join(pieces) + "\n\n")
    print(f"[tok-eval] {declared_name}: {metrics['n_tokens']} tokens, "
          f"bytes/tok={metrics['bytes_per_token']}, roundtrip={metrics['roundtrip_ok']}")
    return metrics

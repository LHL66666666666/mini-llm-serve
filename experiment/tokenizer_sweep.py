"""vocab_size sweep:一键训练 4k/8k/16k/32k 等多个 tokenizer,在同一份固定的
held-out 文本上评估,产出对比表 + 长度分布图。

目录:
  experiments/tokenizer/
    bpe_4k/   {tokenizer.json, metrics.json, samples.md}
    bpe_8k/   ...
    bpe_16k/  ...
    bpe_32k/  ...
    comparison.md      # 横向对比表
    length_hist.png    # 各 size 的 token 长度分布

公平性保证:所有 size 用同一 train 语料训练、同一 held-out 文本评估。
held-out 必须与训练语料不重叠(这里约定单独一个文件)。

用法:
    python -m experiment.tokenizer_sweep \
        --train-corpus data/tok_train.txt \
        --heldout data/tok_heldout.txt \
        --sizes 4096 8192 16384 32768 \
        --out experiments/tokenizer
"""

import os
import json
import argparse


# SPECIAL_TOKENS = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>",
#                   "<|fim_suffix|>", "<|endofprompt|>"]
SPECIAL_TOKENS = ["<|endoftext|>"]


def run_sweep(train_corpus, heldout_path, sizes, out_root,
              num_chunks=32, num_processes=8):
    from train_tokenizer import train_tokenizer
    from tokenizer import Tokenizer
    from experiment.tokenizer_eval import evaluate_tokenizer

    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_text = f.read()

    rows = []
    hist_data = {}
    for vs in sizes:
        name = f"bpe_{vs//1000}k" if vs % 1000 == 0 else f"bpe_{vs}"
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        tok_path = os.path.join(out_dir, "tokenizer.json")

        # 训练(已存在则复用,支持中断重跑)
        if os.path.exists(tok_path):
            print(f"[sweep] {name}: tokenizer.json 已存在,复用")
            tokenizer = Tokenizer.load(tok_path)
        else:
            print(f"[sweep] training {name} (vocab_size={vs}) ...")
            vocab, merges, special_map, pattern = train_tokenizer(
                filepath=train_corpus, vocab_size=vs, special_tokens=SPECIAL_TOKENS,
                num_chunks=num_chunks, num_processes=num_processes,
            )
            tokenizer = Tokenizer(vocab=vocab, merge=merges,
                                  special_tokens=special_map, pattern=pattern)
            tokenizer.save(tok_path)

        m = evaluate_tokenizer(tokenizer, heldout_text, out_dir, declared_name=name)
        rows.append(m)
        hist_data[name] = m["length_hist"]

    _write_comparison(rows, out_root)
    _plot_hist(hist_data, out_root)
    return rows


def _write_comparison(rows, out_root):
    path = os.path.join(out_root, "comparison.md")
    cols = ["tokenizer", "vocab_size", "n_tokens", "bytes_per_token", "chars_per_token",
            "avg_token_byte_len", "byte_fallback_ratio", "vocab_utilization",
            "roundtrip_ok", "encode_MB_per_sec"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Tokenizer vocab_size 对比(同一 held-out 文本)\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        f.write("\n说明:bytes_per_token 越高 = 压缩越好;byte_fallback 越低越好;"
                "roundtrip 必须全 True;vocab_utilization 过低说明 vocab 偏大有浪费。\n")
    print(f"[sweep] 对比表写入 {path}")


def _plot_hist(hist_data, out_root):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[sweep] 无 matplotlib,跳过长度分布图")
        return
    plt.figure(figsize=(8, 5))
    for name, hist in hist_data.items():
        if not hist:
            continue
        xs = sorted(int(k) for k in hist)
        ys = [hist[str(k)] for k in xs]
        plt.plot(xs, ys, marker="o", ms=3, label=name)
    plt.xlabel("token 字节长度")
    plt.ylabel("占比")
    plt.title("Token 长度分布 by vocab_size")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_root, "length_hist.png")
    plt.savefig(p, dpi=120)
    print(f"[sweep] 长度分布图写入 {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-corpus", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--out", default="experiments/tokenizer")
    # 可在命令行控制 num_chunks 和 num_processes
    ap.add_argument("--num-chunks", type=int, default=32,
                    help="训练 BPE 时的文件分块数（语料越小需要设越小）")
    ap.add_argument("--num-processes", type=int, default=8,
                    help="并行进程数")
    args = ap.parse_args()
    run_sweep(args.train_corpus, args.heldout, args.sizes, args.out,
              num_chunks=args.num_chunks, num_processes=args.num_processes)


if __name__ == "__main__":
    main()

"""跨 run 对比:读多个 run 的 metrics.jsonl,画 val_loss vs tokens,
并计算"达到目标 val loss 用了多少 tokens / steps"(token efficiency)。

这是整套实验真正的产出:GPT vs LLaMA、不同 vocab_size、各种消融,
都靠这张图和这个表来下结论——而不是靠肉眼看生成质量。

用法:
    python compare_runs.py experiments/training/run_a experiments/training/run_b \
        --target-loss 4.0 --x tokens
"""

import os
import json
import argparse


def load_curve(run_dir, split="val"):
    """返回该 run 在指定 split 下的 (steps, tokens, losses)。"""
    path = os.path.join(run_dir, "metrics.jsonl")
    steps, tokens, losses = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") != split or "loss" not in rec:
                continue
            steps.append(rec["step"])
            tokens.append(rec.get("tokens"))
            losses.append(rec["loss"])
    return steps, tokens, losses


def tokens_to_reach(tokens, losses, target):
    """首次达到 target loss 所需的 token 数(线性插值);没达到则返回 None。"""
    for i in range(1, len(losses)):
        if losses[i] <= target:
            # 在 [i-1, i] 间线性插值
            l0, l1 = losses[i - 1], losses[i]
            t0, t1 = tokens[i - 1], tokens[i]
            if t0 is None or t1 is None or l0 == l1:
                return t1
            frac = (l0 - target) / (l0 - l1)
            return t0 + frac * (t1 - t0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 目录列表")
    ap.add_argument("--split", default="val")
    ap.add_argument("--x", default="tokens", choices=["tokens", "steps"])
    ap.add_argument("--target-loss", type=float, default=None)
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False
        print("[compare] 未安装 matplotlib,只输出文字表格")

    print(f"\n{'run':40s} {'final_'+args.split+'_loss':>18s} {'tokens_to_target':>18s}")
    print("-" * 78)
    if have_plt:
        plt.figure(figsize=(8, 5))

    for run in args.runs:
        steps, tokens, losses = load_curve(run, args.split)
        if not losses:
            print(f"{os.path.basename(run):40s} {'(no data)':>18s}")
            continue
        name = os.path.basename(run.rstrip("/"))
        final = losses[-1]
        ttt = tokens_to_reach(tokens, losses, args.target_loss) if args.target_loss else None
        ttt_str = f"{ttt/1e6:.2f}M" if ttt else "-"
        print(f"{name:40s} {final:18.4f} {ttt_str:>18s}")
        if have_plt:
            xs = tokens if args.x == "tokens" else steps
            plt.plot(xs, losses, marker="o", ms=3, label=name)

    if have_plt:
        if args.target_loss:
            plt.axhline(args.target_loss, ls="--", c="gray", lw=1)
        plt.xlabel("tokens seen" if args.x == "tokens" else "optimizer steps")
        plt.ylabel(f"{args.split} loss")
        plt.legend()
        plt.title("Run comparison")
        plt.tight_layout()
        plt.savefig(args.out, dpi=120)
        print(f"\n图已保存:{args.out}")


if __name__ == "__main__":
    main()

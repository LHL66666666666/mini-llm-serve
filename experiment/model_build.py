"""torch 端:用统一模型构建 + 真实参数审计。

build_model:唯一的模型构建入口。GPT/LLaMA/消融变体都走这里 ->
unified_transformer。彻底取代 nano_GPT / nano_Llama 作为训练路径
(两个文件保留作教学参考,不再用于训练)。

audit_params:实例化真实模型逐参数计数,并与 model_spec 的解析式结果对拍
(assert 一致),然后把 parity 对比表写盘。这一步是"等参数量对比"的守门人:
跑对比实验前先确认要对齐的预设确实对齐。
"""

import os
import json

from experiment.model_spec import resolve_model_cfg, count_params, PRESETS


def build_model(cfg):
    """从 ExperimentConfig 构建统一模型。cfg.model 可含 preset 键 + 覆盖字段。"""
    from nano_unified import unified_transformer, unified_transformer_config
    resolved = resolve_model_cfg(cfg.model)
    return unified_transformer(unified_transformer_config(**resolved))


def _count_real(model) -> dict:
    """对真实模型逐参数计数。注意权重绑定时 named_parameters 不会重复计 lm_head。"""
    total = sum(p.numel() for p in model.parameters())
    tok = model.token_embedding.weight.numel()
    pos = 0
    if hasattr(model, "position_embedding"):
        pos = model.position_embedding.weight.numel()
    return {"total": total, "embedding": tok + pos, "token_emb": tok, "pos_emb": pos}


def audit_one(model_dict: dict, verify_with_torch: bool = True) -> dict:
    """计算单个配置的参数明细;若 verify_with_torch,则实例化真模型对拍。"""
    analytic = count_params(model_dict)
    if verify_with_torch:
        from nano_unified import unified_transformer, unified_transformer_config
        model = unified_transformer(unified_transformer_config(**resolve_model_cfg(model_dict)))
        real = _count_real(model)
        # 对拍:解析式必须和真实模型一致,否则解析式公式过时了
        assert real["total"] == analytic["total"], (
            f"参数计数不一致! 解析式={analytic['total']} 真实={real['total']};"
            f"model_spec.count_params 公式需要更新"
        )
        assert real["embedding"] == analytic["embedding"], "embedding 计数不一致"
        del model
    return analytic


def audit_params(named_configs: dict, out_dir: str = "experiments",
                 verify_with_torch: bool = True) -> list:
    """对一组 {名字: model_dict} 做审计,写 param_report.{md,json},返回明细列表。

    named_configs 例如:
        {"gpt": {"preset":"gpt", "hidden_size":256, ...},
         "llama": {"preset":"llama", "hidden_size":256, ...}, ...}
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, mcfg in named_configs.items():
        a = audit_one(mcfg, verify_with_torch=verify_with_torch)
        rows.append({
            "name": name,
            "total": a["total"],
            "total_M": round(a["total"] / 1e6, 4),
            "embedding": a["embedding"],
            "embedding_M": round(a["embedding"] / 1e6, 4),
            "blocks": a["blocks"],
            "blocks_M": round(a["blocks"] / 1e6, 4),
            "lm_head_extra": a["lm_head_extra"],
            "swiglu_hidden_dim": a["swiglu_hidden_dim"],
            "resolved": a["resolved"],
        })

    # JSON
    with open(os.path.join(out_dir, "param_report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Markdown 对比表 + parity 提示
    md = os.path.join(out_dir, "param_report.md")
    cols = [("name", "config"), ("total_M", "total(M)"), ("embedding_M", "emb(M)"),
            ("blocks_M", "blocks(M)"), ("lm_head_extra", "lm_head_extra"),
            ("swiglu_hidden_dim", "swiglu_hd")]
    with open(md, "w", encoding="utf-8") as f:
        f.write("# 参数量对比表\n\n")
        f.write("| " + " | ".join(h for _, h in cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k, _ in cols) + " |\n")
        # parity 提示
        f.write("\n## parity 检查\n\n")
        if "gpt" in named_configs and "llama" in named_configs:
            g = next(r for r in rows if r["name"] == "gpt")["total"]
            l = next(r for r in rows if r["name"] == "llama")["total"]
            f.write(f"- GPT vs LLaMA total 差异:{abs(g-l):,} ({abs(g-l)/l*100:.2f}%)。"
                    f"权重绑定后嵌入对齐,残差差异主要来自 SwiGLU 宽度与归一化参数。\n")
        if "llama" in named_configs and "llama_gelu" in named_configs:
            lb = next(r for r in rows if r["name"] == "llama")["blocks"]
            gb = next(r for r in rows if r["name"] == "llama_gelu")["blocks"]
            f.write(f"- SwiGLU vs GELU blocks 差异:{abs(lb-gb):,} ({abs(lb-gb)/gb*100:.2f}% of blocks)。"
                    f"**此消融默认非等参**:SwiGLU 中间维按 256 向上取整会超出 8/3 规则。"
                    f"若要隔离'FFN 类型'效应,需调中间维对齐参数量;否则在 report 中显式记录此 delta。\n")
        f.write("\n说明:lm_head_extra 在权重绑定时为 0;非 0 说明未绑定,会让 total 虚高一个 vocab×hidden。\n")
    print(f"[audit] 参数报告写入 {md}")
    return rows


def _default_audit_set(hidden_size=256, n_layer=4, n_head=4, vocab_size=50304, max_seq_len=256):
    sizes = dict(hidden_size=hidden_size, n_layer=n_layer, n_head=n_head,
                 vocab_size=vocab_size, max_seq_len=max_seq_len)
    return {name: {"preset": name, **sizes} for name in
            ["gpt", "llama", "llama_layernorm", "llama_gelu", "llama_learnedpe"]}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="对预设/消融做参数量审计并落盘对比表")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--no-torch-verify", action="store_true",
                    help="只用解析式,不实例化真模型对拍(无 torch 环境时用)")
    args = ap.parse_args()
    configs = _default_audit_set(args.hidden_size, args.n_layer, args.n_head,
                                 args.vocab_size, args.max_seq_len)
    audit_params(configs, out_dir=args.out, verify_with_torch=not args.no_torch_verify)


if __name__ == "__main__":
    main()
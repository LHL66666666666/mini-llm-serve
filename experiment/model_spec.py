"""统一模型规格(torch-free):默认值、预设、配置解析校验、解析式参数计算器。

为什么独立成一个无 torch 的模块
--------------------------------
- 让 GPT/LLaMA/消融变体都由"同一份 unified 模型 + 一组开关"定义,彻底消除
  三套模型实现并存导致的代码漂移。GPT、LLaMA 只是这里的两个预设。
- 参数计算器是纯 Python 解析式实现,不需要实例化模型,可在任何环境跑;
  同时作为真实模型参数计数的交叉校验(见 model_build.audit_params)。

unified_transformer_config 的 12 个字段全部在这里给出默认值,因此 YAML 只需
写与预设不同的部分;未知键(如把 use_RMSNorm 误写成 use_RMSnorm)会被 resolve
直接报错,而不是等到构造模型时 TypeError。
"""

from copy import deepcopy

# 统一模型的全部字段 + 默认值(默认 = 全现代特性开启 + 权重绑定,即 LLaMA 风格基线)
UNIFIED_DEFAULTS = {
    "max_seq_len": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 12,
    "hidden_size": 768,
    "max_position_embeddings": 1024,
    "dropout": 0.0,
    "bias": False,
    "use_rope": True,
    "use_RMSNorm": True,
    "use_SwiGLU": True,
    "use_weight_tying": True,
}

# 预设:只列出与 UNIFIED_DEFAULTS 不同的开关。
# 关键:两个预设都开 weight_tying —— 等参数量前提。
PRESETS = {
    # GPT:learned PE + LayerNorm + GELU-MLP
    "gpt": {
        "use_rope": False,
        "use_RMSNorm": False,
        "use_SwiGLU": False,
        "use_weight_tying": True,
    },
    # LLaMA:RoPE + RMSNorm + SwiGLU(= 默认)
    "llama": {
        "use_rope": True,
        "use_RMSNorm": True,
        "use_SwiGLU": True,
        "use_weight_tying": True,
    },
    # 消融变体:从 llama 基线一次只翻一个开关
    "llama_layernorm": {"use_rope": True, "use_RMSNorm": False, "use_SwiGLU": True, "use_weight_tying": True},
    "llama_gelu":      {"use_rope": True, "use_RMSNorm": True,  "use_SwiGLU": False, "use_weight_tying": True},
    "llama_learnedpe": {"use_rope": False, "use_RMSNorm": True, "use_SwiGLU": True, "use_weight_tying": True},
}

_BOOL_FIELDS = {"bias", "use_rope", "use_RMSNorm", "use_SwiGLU", "use_weight_tying"}
_INT_FIELDS = {"max_seq_len", "vocab_size", "n_layer", "n_head", "hidden_size", "max_position_embeddings"}


def resolve_model_cfg(model_dict: dict) -> dict:
    """把(可能含 preset 键、可能不完整的)model dict 解析成完整的 12 字段 dict。

    顺序:UNIFIED_DEFAULTS -> 应用 preset(若有) -> 应用用户显式字段。
    校验:未知键报错(拼写)、hidden_size 必须被 n_head 整除、类型基本检查。
    """
    m = deepcopy(model_dict or {})
    preset = m.pop("preset", None)

    base = deepcopy(UNIFIED_DEFAULTS)
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"未知 preset: {preset!r};可选:{sorted(PRESETS)}")
        base.update(PRESETS[preset])

    unknown = set(m) - set(UNIFIED_DEFAULTS)
    if unknown:
        raise ValueError(
            f"model 配置出现未知字段 {sorted(unknown)};"
            f"合法字段:{sorted(UNIFIED_DEFAULTS)}(注意大小写,如 use_RMSNorm)"
        )
    base.update(m)

    # 轻量类型/约束校验
    for k in _INT_FIELDS:
        if not isinstance(base[k], int) or isinstance(base[k], bool):
            raise TypeError(f"{k} 必须是 int,得到 {base[k]!r}")
    for k in _BOOL_FIELDS:
        if not isinstance(base[k], bool):
            raise TypeError(f"{k} 必须是 bool,得到 {base[k]!r}")
    if base["hidden_size"] % base["n_head"] != 0:
        raise ValueError(f"hidden_size({base['hidden_size']}) 必须能被 n_head({base['n_head']}) 整除")
    return base


def swiglu_hidden_dim(hidden_size: int, multiple_of: int = 256) -> int:
    """与 nano_Llama.SwiGLU 完全一致的中间维计算。"""
    return (((hidden_size * 8) // 3) + multiple_of - 1) // multiple_of * multiple_of


def count_params(model_dict: dict) -> dict:
    """解析式参数量计算。返回各部分明细 + 总量。

    分桶说明(为对比的可解释性,比 unified.get_num_params 更细):
    - token_emb / pos_emb:嵌入
    - blocks:所有 Transformer 层(注意力 + 归一化 + FFN)
    - final_norm:最后一层归一化
    - lm_head_extra:权重绑定时为 0;不绑定时 = vocab*hidden(+bias)
    """
    c = resolve_model_cfg(model_dict)
    H, L, V = c["hidden_size"], c["n_layer"], c["vocab_size"]
    bias = c["bias"]

    token_emb = V * H
    pos_emb = 0 if c["use_rope"] else c["max_seq_len"] * H

    # 注意力 q/k/v/o
    attn = 4 * H * H + (4 * H if bias else 0)
    # 归一化:RMSNorm 仅 gamma(H);LayerNorm 有 weight+bias(2H)。每层两个
    norm_each = H if c["use_RMSNorm"] else 2 * H
    norm = 2 * norm_each
    # FFN
    if c["use_SwiGLU"]:
        hd = swiglu_hidden_dim(H)
        mlp = 3 * H * hd + ((2 * hd + H) if bias else 0)  # gate/up/down 的 bias
    else:
        mlp = 8 * H * H  # GPTMLP 硬编码 bias=False
    per_layer = attn + norm + mlp
    blocks = L * per_layer

    final_norm = H if c["use_RMSNorm"] else 2 * H
    lm_head_extra = 0 if c["use_weight_tying"] else V * H
    lm_head_bias = V if (bias and not c["use_weight_tying"]) else (V if bias and c["use_weight_tying"] else 0)
    # 说明:绑定时 weight 共享,但若 bias=True 仍有独立 bias;此项目预设 bias=False,通常为 0

    total = token_emb + pos_emb + blocks + final_norm + lm_head_extra + lm_head_bias
    embedding = token_emb + pos_emb
    non_embedding = total - embedding  # 含 lm_head_extra(不绑定时)

    return {
        "total": total,
        "embedding": embedding,
        "token_emb": token_emb,
        "pos_emb": pos_emb,
        "blocks": blocks,
        "final_norm": final_norm,
        "lm_head_extra": lm_head_extra + lm_head_bias,
        "non_embedding": non_embedding,
        "swiglu_hidden_dim": swiglu_hidden_dim(H) if c["use_SwiGLU"] else None,
        "resolved": c,
    }
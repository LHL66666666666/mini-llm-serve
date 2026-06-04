import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import Tuple, Optional
import inspect


@dataclass
class LlamaConfig:
    max_seq_len: int = 256  # 最大序列长度
    vocab_size: int = 50304  # 词表大小
    n_layer: int = 4        # Transformer 层数
    n_head: int = 4         # 注意力头数
    hidden_size: int = 256   # 嵌入维度
    max_position_embeddings: int = 2048 # 模型支持的最大序列长度（位置编码的最大索引）
    dropout: float = 0.0
    bias: bool = False


# 创建因果掩码 causal mask 掩码是加性的
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
):
    B, tgt_len = input_ids_shape

    # 创建下半全为0，其余全为-inf的方阵
    mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device, dtype=dtype), diagonal=1)
    mask.masked_fill_(mask == 1, float("-inf"))
    mask = mask.to(dtype)

    # 过去的 token 已经计算过 K/V 并缓存起来
    # 当前这一步要处理的是新输入 token，当前 token 可以看见所有历史缓存 token
    # 但仍然不能看见未来 token，所以 mask 的列数要从原来的 tgt_len 扩展成：past_key_values_length + tgt_len
    if past_key_values_length > 0:
        # 在mask之前加上拼接past_key_values
        mask = torch.cat((torch.zeros((tgt_len, past_key_values_length), device=device, dtype=dtype), mask), dim=-1)

    return mask[None, None, :, :].expand(B, 1, tgt_len, tgt_len + past_key_values_length)

# 创建填充掩码 padding mask 掩码是加性的
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """ `[bsz, seq_len]` -> `[bsz, 1, tgt_seq_len, src_seq_len]`"""
    # mask的shape是(batch_size, source_sequence_length)
    # 通常它是 0/1 张量：1 表示这个位置有效，0 表示这个位置是 padding，需要遮住
    bsz, src_len = mask.size()

    # 如果有target length则输出掩码的最后一个维度为target length
    # 如果没有target length,则为src_len
    # 这个参数的意义是：扩展后的 mask 要和哪一个 attention score 的 query 长度对齐。
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 把 [bsz, src_len] 扩展成四维
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    # 反转 mask
    inverted_mask = 1 - expanded_mask
    # 把 1 的位置替换成最小值
    # 结果可以和QK^T的结果直接相加,进行位置掩码操作
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# LayerNorm -> RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # 注意 RMSNorm 需要转为 float32 计算
        input_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        x = x * torch.rsqrt( (x ** 2).mean(dim=-1, keepdim=True) + self.variance_epsilon )
        return (x * self.gamma).to(dtype=input_dtype)

# GPT 的 FFN -> SwiGLU
class SwiGLU(nn.Module):
    def __init__(self, input_size, dropout, bias,  multiple_of=256) -> None:
        super().__init__()
        hidden_dim = (((input_size * 8) // 3) +  multiple_of - 1) // multiple_of * multiple_of
        self.gate_proj = nn.Linear(input_size, hidden_dim, bias=bias) # gate_proj: Swish 门控路径
        self.up_proj = nn.Linear(input_size, hidden_dim, bias=bias)  # up_proj: 线性路径
        self.down_proj = nn.Linear(hidden_dim, input_size, bias=bias)  # down_proj: 输出投影
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)

class LlamaMLP(nn.Module):
    def __init__(self,config:LlamaConfig):
        super().__init__()
        self.SwiGLU = SwiGLU(config.hidden_size, config.dropout, bias=config.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.SwiGLU(x)


# 可学习的位置嵌入 -> RoPE 旋转位置编码
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        :param dim: 每个注意力头的维度head_dim
        :param max_position_embeddings: 预计算 cos/sin 缓存时使用的最大序列长度
        :param base: 频率计算的基数，默认 10000
        :param device: device
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 初始频率表
        dim_indices = torch.arange(0, dim, step=2, dtype=torch.float).to(device)
        inv_freq = base ** (-dim_indices / dim)
        # inv_freq 注册为 buffer
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        pos_indices = torch.arange(seq_len,dtype=self.inv_freq.dtype, device=device)
        # 下面两行实现等价
        freqs = torch.einsum("i,j->ij", pos_indices, self.inv_freq)
        # freqs = pos_indices[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs,freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch_size, n_head, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# 交换&取负
def rotate_half(x):
    n_dim = x.shape[-1]
    x1 = x[..., :n_dim // 2]
    x2 = x[..., n_dim // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    :param q, k:          [batch, n_head, seq_len, head_dim]
    :param cos, sin:      [1, 1, max_seq_len, head_dim]
    :param position_ids:  [batch, seq_len]，支持填充、KV cache 偏移等场景
    """
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # position_ids：每个序列位置的实际位置索引，形状 [batch_size, seq_len] 。用于正确处理填充、变长序列，支持自回归生成与 KV cache等
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# KV cache
# 实现加入了KV cache的多头注意力
class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_head = config.n_head
        assert self.hidden_size % self.n_head == 0
        self.head_dim = self.hidden_size // self.n_head
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self._init_rope()

    def _shape(self, x: torch.Tensor):
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.max_position_embeddings
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.shape

        q_state, k_state, v_state = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q_state, k_state, v_state = self._shape(q_state), self._shape(k_state), self._shape(v_state)

        # kv_seq_len = 当前 seq_len + 已缓存的历史长度
        kv_seq_len = T
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]


        cos, sin = self.rotary_emb(q_state, kv_seq_len)
        q_state, k_state = apply_rotary_pos_emb(q_state, k_state, cos, sin, position_ids)

        # 拼接历史 KV（拼接在 seq_len 维度，即 dim=-2）
        if past_key_value is not None:
            k_state = torch.cat((past_key_value[0], k_state), dim=-2)
            v_state = torch.cat((past_key_value[1], v_state), dim=-2)

        past_key_value = (k_state, v_state) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)

        # Flash Attention
        # 特别注意：推理阶段如果 KV cache，不要使用 is_causal=True 构造因果掩码
        # is_causal=True 自动构造的是下三角矩阵，使用 KV cache 当 len(query) != len(key) 的时候，没有未来的token但是还做了屏蔽，出现错误
        if self.flash and (not output_attentions):
            if self.training:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q_state,
                    k_state,
                    v_state,
                    dropout_p=self.config.dropout if self.training else 0.0,
                    is_causal=True,
                    scale=scale,
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q_state,
                    k_state,
                    v_state,
                    attn_mask=attention_mask, # 传入一个加性掩码
                    dropout_p=self.config.dropout if self.training else 0.0,
                    is_causal=False, # 已通过 attn_mask 手动构造因果掩码
                    scale=scale,
                )

        else:
            attn_score = q_state @ k_state.transpose(-2, -1) * scale
            # 加性掩码
            if attention_mask is not None:
                if attention_mask.size() != (B, 1, T, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(B, 1, T, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_score += attention_mask

            # softmax 在 float32 下计算更稳定，再转回原始 dtype
            attn_weights = nn.functional.softmax(attn_score, dim=-1, dtype=torch.float32).to(q_state.dtype)
            attn_output = attn_weights @ v_state


        if attn_output.size() != (B, self.n_head, T, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(B, self.n_head, T, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = self.o_proj(attn_output.transpose(1,2).contiguous().view(B, T, -1))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# Decoder层
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.hidden_size)

        self.attn = LlamaAttention(config)

        self.ln_2 = RMSNorm(config.hidden_size)

        self.mlp = LlamaMLP(config)


    def forward(
            self,
            hidden_states: torch.Tensor,  # [batch_size, seq_len, hidden_size]
            attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            position_ids: Optional[torch.LongTensor] = None,  # [batch_size, seq_len]
            past_key_value: Optional[Tuple[torch.Tensor]] = None,  # [batch_size, past_seq_len]
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 层输入，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): 注意力掩码，形状为`(batch, 1, tgt_len, src_len)` 其中填充元素以非常大的负值表示
            output_attentions (`bool`, *optional*):是否返回所有注意力层的注意力张量。
            use_cache (`bool`, *optional*):若设置为`True`，则返回`past_key_values`键值状态，可用于加速解码
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):  缓存的过去键和值投影状态
        """
        residual = hidden_states
        attn_output, attn_weights, past_key_value =  self.attn(self.ln_1(hidden_states), attention_mask, position_ids, past_key_value, output_attentions, use_cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(self.ln_2(hidden_states))

        attn_weights = attn_weights if output_attentions else None

        past_key_value = past_key_value if use_cache else None

        return hidden_states, attn_weights, past_key_value


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # RMSNorm 被用作前置归一化：应用的三个位置：自注意力层之前，前馈网络层之前，最终输出层之前
        # dropout在预训练阶段完全禁用（dropout = 0.0）
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layer)])
        self.ln_final = RMSNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=config.bias)

        # 权重绑定 (Llama 设计中没有，但这里为了对比试验加上了)
        self.lm_head.weight = self.token_embedding.weight

        # 权重初始化
        self.apply(self._init_weights)
        # 对残差投影进行特殊缩放(GPT模型中使用的残差投影缩放),(Llama 中未使用，但这里为了对比试验加上了)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 参数统计
        total, token_embedding, transformer = self.get_num_params()
        print(f"total parameters: {total / 1e6:.2f} M")
        print(f"embedding: {token_embedding / 1e6:.2f} M")
        print(f"transformer: {transformer / 1e6:.2f} M")

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        token_embedding = self.token_embedding.weight.numel()
        transformer = total - token_embedding
        return total, token_embedding, transformer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_decoder_attention_mask(
            self,
            attention_mask, # [bsz, seq_len] 这里的seq_len是src_len
            input_shape,
            inputs_embeds,
            past_key_values_length
    ):
        # 结合因果掩码和填充掩码生成加性掩码
        B, T = input_shape
        causal_attn_mask = None
        if T > 1:
            causal_attn_mask = _make_causal_mask(
                input_ids_shape=input_shape,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length
            )

        # attention_mask一定不为None
        expand_attn_mask = _expand_mask(
            mask=attention_mask,
            dtype=inputs_embeds.dtype,
            tgt_len=T,
        ).to(inputs_embeds.device)

        combined_attention_mask = (
            expand_attn_mask if causal_attn_mask is None else expand_attn_mask + causal_attn_mask
        )

        return combined_attention_mask


    def forward(
            self,
            input_ids: torch.LongTensor = None,  # [batch_size, seq_len]
            attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            position_ids: Optional[torch.LongTensor] = None,  # [batch_size, seq_len]
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,  # [batch_size, past_seq_len]
            output_attentions: Optional[bool] = None,
            use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, ...], ...], Tuple[Tuple[torch.Tensor, ...], ...]]:
        """简化后的前向传播，返回 (output, output_self_attentions, next_kv_cache)"""
        B, T = input_ids.shape
        input_embd = self.token_embedding(input_ids)

        # 处理 KV‑Cache 相关长度
        past_kv_len = 0
        # past_key_values 是一个Tuple，Tuple中每一个元素是一个Tuple，表示每一个layer的(key_cache, value_cache)
        if past_key_values is not None:
            past_kv_len = past_key_values[0][0].shape[-2]

        # 生成 position_ids
        if position_ids is None:
            position_ids = torch.arange(
                start=past_kv_len, end=past_kv_len + T, dtype=torch.long, device=input_embd.device
            ).view(-1, T)
        else:
            position_ids = position_ids.view(-1, T)

        # 创建 attention mask（保留因果 + padding 组合逻辑）
        if attention_mask is None:
            attention_mask = torch.ones((B, past_kv_len + T), dtype=torch.bool, device=input_embd.device)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (B, T),
            input_embd,
            past_kv_len
        )

        # 逐层解码
        hidden_states = input_embd
        next_kv_cache = ()
        output_self_attentions = ()
        for idx, layer in enumerate(self.layers):
            # 取出对应层的kv cache
            past_key_value = past_key_values[idx] if (use_cache and past_key_values is not None) else None

            hidden_states, attn_weights, kv_cache = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )

            if use_cache:
                next_kv_cache += (kv_cache, )
            if output_attentions:
                output_self_attentions += (attn_weights, )


        # 最终 layer norm 和 lm_head
        hidden_states = self.ln_final(hidden_states)
        output = self.lm_head(hidden_states)

        # 返回最后输出，attention_weights和 KV cache
        return output, output_self_attentions, next_kv_cache


    # 自回归生成
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # prompt
        output, _, kv_cache = self(
            input_ids=idx,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=True,
        )
        output = output[:, -1, :] / temperature

        # top_k
        if top_k is not None:
            top_k_values, _ = torch.topk(output, k=top_k, dim=-1)
            output[ output < top_k_values[:, [-1]] ] = float("-inf")

        probs = F.softmax(output, dim=-1)
        # 多项式分布采样
        idx_next = torch.multinomial(probs, num_samples=1)
        # 拼接
        idx = torch.cat((idx, idx_next), dim=-1)

        idx_input = idx_next
        for _ in range(1, max_new_tokens):
            logits, _, kv_cache = self(
                input_ids=idx_input,
                attention_mask=None,
                position_ids=None,
                past_key_values=kv_cache,
                output_attentions=False,
                use_cache=True,
            )

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, k=top_k, dim=-1)
                logits[ logits < top_k_values[:, [-1] ]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)

            idx_input = idx_next

        return idx


    # 优化器配置函数
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 从模型所有命名参数中构建字典param_dict，键为参数名，值为参数张量
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要梯度的参数（例如冻结层的参数），这些参数不需要优化
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 将参数划分为两组：需衰减 vs 不需衰减
        # 偏置项,LayerNorm 的缩放因子等不需要权重衰减
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        # 构建优化器参数组
        # 将两组参数以列表形式传入优化器，每组可以拥有独立的 weight decay 系数
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 统计参数量并打印
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 判断是否使用融合版 AdamW（Fused AdamW）
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # 创建优化器并返回
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

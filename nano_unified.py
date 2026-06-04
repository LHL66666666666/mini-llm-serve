"""
架构消融实验，构建一个统一 Transformer 模型，通过配置字段控制是否启用 RoPE、SwiGLU、RMSNorm、Weight Tying 等关键组件
GPT 和 LLaMA 成为该模型的两种预设，中间变体可直接由 YAML 配置

提取两种架构的共性，将差异封装为可插拔模块：
位置编码：LearnablePE vs RotaryEmbedding（use_rope）
归一化：LayerNorm vs RMSNorm（use_RMSnorm）
前馈：GELU‑MLP vs SwiGLU（use_SwiGLU）
注意力：CausalSelfAttention，RoPE 在其内部可选应用
权重绑定：lm_head.weight = token_embedding.weight（use_weight_tying）
"""
import torch
from torch import nn
from nano_Llama import _make_causal_mask, _expand_mask, RMSNorm, \
    SwiGLU, RotaryEmbedding, rotate_half, apply_rotary_pos_emb
from nano_GPT import GPTMLP
from dataclasses import dataclass
import math
from typing import Tuple, Optional
import inspect


@dataclass
class unified_transformer_config:
    # 基础配置
    max_seq_len: int  # 最大序列长度
    vocab_size: int  # 词表大小
    n_layer: int  # Transformer 层数
    n_head: int  # 注意力头数
    hidden_size: int  # 嵌入维度
    max_position_embeddings: int  # 模型支持的最大序列长度（位置编码的最大索引）
    dropout: float
    bias: bool
    # 架构消融开关
    use_rope: bool  # 是否使用 RoPE
    use_RMSNorm: bool  # 是否使用 RMSNorm
    use_SwiGLU: bool  # 是否使用 SwiGLU
    use_weight_tying: bool  # 是否开启权重绑定


class unified_transformer_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        assert config.hidden_size % config.n_head == 0
        self.head_dim = config.hidden_size // config.n_head
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.use_rope = config.use_rope
        if self.use_rope:
            self._init_rope()


    def _shape(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
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

        kv_seq_len = T
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if self.use_rope:
            # 在使用 RoPE 时，q 和 k 的实际长度（参与旋转编码的维度）是完全相同的
            cos, sin = self.rotary_emb(q_state, kv_seq_len)
            q_state, k_state = apply_rotary_pos_emb(q_state, k_state, cos, sin, position_ids)

        if past_key_value is not None:
            k_state = torch.cat((past_key_value[0], k_state), dim=-2)
            v_state = torch.cat((past_key_value[1], v_state), dim=-2)

        scale = 1 / math.sqrt(self.head_dim)

        if self.flash and not output_attentions:
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
                    attn_mask=attention_mask,
                    dropout_p=self.config.dropout if self.training else 0.0,
                    is_causal=False,
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

            attn_weights = torch.nn.functional.softmax(attn_score, dim=-1, dtype=torch.float32).to(q_state.dtype)
            if self.training:
                attn_weights = self.dropout(attn_weights)

            attn_output = attn_weights @ v_state

        if attn_output.size() != (B, self.n_head, T, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(B, self.n_head, T, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = self.o_proj(attn_output.transpose(1, 2).contiguous().view(B, T, -1))

        if use_cache:
            past_key_value = (k_state, v_state)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class unified_transformer_layer(nn.Module):
    def __init__(self, config: unified_transformer_config):
        super().__init__()
        self.config = config
        self.ln1 = RMSNorm(config.hidden_size) if config.use_RMSNorm else nn.LayerNorm(config.hidden_size)
        self.attention = unified_transformer_Attention(config)

        self.ln2 = RMSNorm(config.hidden_size) if config.use_RMSNorm else nn.LayerNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.dropout, config.bias) if config.use_SwiGLU \
            else GPTMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            hidden_states: torch.Tensor,  # [batch_size, seq_len, hidden_size]
            attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            position_ids: Optional[torch.LongTensor] = None,  # [batch_size, seq_len]
            past_key_value: Optional[Tuple[torch.Tensor]] = None,  # [batch_size, past_seq_len]
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        attn_output, attn_weights, past_key_value = self.attention(
            self.ln1(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        mlp_output = self.mlp(self.ln2(hidden_states))
        hidden_states = residual + self.dropout(mlp_output)

        attn_weights = attn_weights if output_attentions else None
        past_key_value = past_key_value if use_cache else None
        return hidden_states, attn_weights, past_key_value

class unified_transformer(nn.Module):
    def __init__(self, config: unified_transformer_config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_size)

        self.layers = nn.ModuleList([unified_transformer_layer(config) for _ in range(config.n_layer)])

        self.ln_final = RMSNorm(self.hidden_size) if config.use_RMSNorm else nn.LayerNorm(self.hidden_size)

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # 权重绑定
        if config.use_weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # 权重初始化
        self.apply(self._init_weights)
        # 对残差投影进行特殊缩放(GPT模型中使用的残差投影缩放),(Llama 中未使用，但这里为了对比试验加上了)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 打印输出消融实验的选项
        print(
            f"use_rope: {config.use_rope}\n"
            f"use_RMSNorm: {config.use_RMSNorm}\n"
            f"use_SwiGLU: {config.use_SwiGLU}\n"
            f"use_weight_tying: {config.use_weight_tying}\n"
        )

        # 参数统计
        total, embedding, transformer = self.get_num_params()
        print(f"total parameters: {total / 1e6:.2f} M")
        print(f"embedding: {embedding / 1e6:.2f} M")
        print(f"transformer: {transformer / 1e6:.2f} M")

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        num_embedding = self.token_embedding.weight.numel()
        if not self.config.use_rope:
            num_embedding += self.position_embedding.weight.numel()
        transformer = total - num_embedding
        return total, num_embedding, transformer


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

        # 处理 KV‑Cache 相关长度
        past_kv_len = 0
        # past_key_values 是一个Tuple，Tuple中每一个元素是一个Tuple，表示每一个layer的(key_cache, value_cache)
        if past_key_values is not None:
            past_kv_len = past_key_values[0][0].shape[-2]

        # 生成 position_ids
        if position_ids is None:
            position_ids = torch.arange(
                start=past_kv_len, end=past_kv_len + T, dtype=torch.long, device=input_ids.device
            ).view(-1, T)
        else:
            position_ids = position_ids.view(-1, T)


        input_embd = self.token_embedding(input_ids) if self.config.use_rope \
            else self.dropout( self.position_embedding(position_ids) + self.token_embedding(input_ids) )

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
                next_kv_cache += (kv_cache,)
            if output_attentions:
                output_self_attentions += (attn_weights,)

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

        probs = torch.nn.functional.softmax(output, dim=-1)
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

            probs = torch.nn.functional.softmax(logits, dim=-1)
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


if __name__ == "__main__":
    pass

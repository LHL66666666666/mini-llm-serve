import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import Tuple, Optional
import inspect

# 创建因果掩码 causal mask 掩码是加性的
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
):
    B, tgt_len = input_ids_shape
    mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device, dtype=dtype), diagonal=1)
    mask.masked_fill_(mask == 1, float("-inf"))
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat((torch.zeros((tgt_len, past_key_values_length), device=device, dtype=dtype), mask), dim=-1)

    return mask[None, None, :, :].expand(B, 1, tgt_len, tgt_len + past_key_values_length)

# 创建填充掩码 padding mask 掩码是加性的
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@dataclass
class GPTConfig:
    max_seq_len: int = 1024  # 最大序列长度
    vocab_size: int = 50304  # 词表大小
    n_layer: int = 12        # Transformer 层数
    n_head: int = 12         # 注意力头数
    hidden_size: int = 768   # 嵌入维度
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.n_head == 0
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.n_head

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.register_buffer(name="mask", tensor=torch.tril(
            torch.ones((config.max_seq_len, config.max_seq_len))
        ).view(1, 1, config.max_seq_len, config.max_seq_len))

    def _shape(self, x):
        B, T, hidden_size = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

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

        past_kv_len = T
        if past_key_value is not None:
            past_kv_len += past_key_value[0].shape[-2]
            k_state = torch.cat((past_key_value[0], k_state), dim=-2)
            v_state = torch.cat((past_key_value[1], v_state), dim=-2)

        past_key_value = (k_state, v_state) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)

        # Flash Attention
        # 特别注意：推理阶段如果启用 KV cache，不要使用 is_causal=True 构造因果掩码
        # is_causal=True 自动构造的是下三角矩阵，使用 KV cache 当 len(query) != len(key) 的时候，没有未来的token但是还做了屏蔽，出现错误
        if self.flash and not output_attentions:
            if self.training:
                attn_output = F.scaled_dot_product_attention(
                    q_state,
                    k_state,
                    v_state,
                    dropout_p=self.config.dropout if self.training else 0.0,
                    is_causal=True,
                    scale=scale,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
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
                if attention_mask.size() != (B, 1, T, past_kv_len):
                    raise ValueError(
                        f"Attention mask should be of size {(B, 1, T, past_kv_len)}, but is {attention_mask.size()}"
                    )
                attn_score += attention_mask

            # softmax 在 float32 下计算更稳定，再转回原始 dtype
            attn_weights = nn.functional.softmax(attn_score, dim=-1, dtype=torch.float32).to(q_state.dtype)
            if self.training:
                attn_weights = self.dropout(attn_weights)

            attn_output = attn_weights @ v_state


        if attn_output.size() != (B, self.n_head, T, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(B, self.n_head, T, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = self.o_proj(attn_output.transpose(1, 2).contiguous().view(B, T, -1))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




class GPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4*config.hidden_size, bias=False)
        self.gelu = nn.GELU()
        self.o_proj = nn.Linear(4*config.hidden_size, config.hidden_size, bias=False)
        self.ffn = nn.Sequential(
            self.c_fc,
            self.gelu,
            self.o_proj
        )

    def forward(self, inputs):
        return self.ffn(inputs)

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ffn = GPTMLP(config)
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
        attn_output, attn_weights, past_key_value = self.attn(
            self.ln1(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        ffn_output = self.ffn(self.ln2(hidden_states))
        hidden_states = residual + self.dropout(ffn_output)

        attn_weights = attn_weights if output_attentions else None
        past_key_value = past_key_value if use_cache else None
        return hidden_states, attn_weights, past_key_value

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_seq_len, config.hidden_size)

        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])

        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定 weight-tying
        self.lm_head.weight = self.wte.weight
        # 可以使用下面这一行观察，绑定之后发现 parameters 中已经不存在 lm_head
        # print([name for name, param in self.named_parameters()])


        # 权重初始化
        self.apply(self._init_weights)
        # 残差投影层的特殊缩放: / sqrt(2 * n_layer)
        for param_name, param in self.named_parameters():
            if param_name.endswith("o_proj.weight"):
                nn.init.normal_(param, mean=0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 参数统计
        total, wte, wpe, emb, transformer = self.get_num_params()
        print(f"total parameters: {total / 1e6:.2f} M")
        print(f"total parameters without position embedding: {(total - wpe) / 1e6:.2f} M")
        print(f"embedding: {emb / 1e6:.2f} M")
        print(f"position embedding: {wpe / 1e6:.2f} M")
        print(f"transformer: {transformer / 1e6:.2f} M")

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        wte = self.wte.weight.numel()
        wpe = self.wpe.weight.numel()
        emb = wte + wpe
        transformer = total - emb
        return total, wte, wpe, emb, transformer

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def _prepare_decoder_attention_mask(
            self,
            attention_mask, # [bsz, seq_len] 这里的seq_len是src_len
            input_shape,
            inputs_embeds,
            past_key_values_length
    ):
        B, T = input_shape
        causal_attn_mask = None
        if T > 1:
            causal_attn_mask = _make_causal_mask(
                input_ids_shape=input_shape,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length
            )

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
        """返回 (output, output_self_attentions, next_kv_cache)"""
        B, T = input_ids.shape

        past_kv_len = 0
        if past_key_values is not None:
            past_kv_len += past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(
                start=past_kv_len, end=past_kv_len + T, dtype=torch.long, device=input_ids.device
            ).view(-1, T)
        else:
            position_ids = position_ids.view(-1, T)

        input_embd = self.dropout( self.wte(input_ids) + self.wpe(position_ids) )


        if attention_mask is None:
            attention_mask = torch.ones((B, past_kv_len + T), dtype=torch.bool, device=input_embd.device)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (B, T),
            input_embd,
            past_kv_len
        )

        hidden_states = input_embd
        next_kv_cache = ()
        output_self_attentions = ()
        for idx, block in enumerate(self.blocks):
            past_key_value = past_key_values[idx] if (use_cache and past_key_values is not None) else None
            hidden_states, attn_weights, kv_cache = block(
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

        hidden_states = self.ln(hidden_states)
        output = self.lm_head(hidden_states)

        return output, output_self_attentions, next_kv_cache


    # 自回归生成
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        :param idx: 形状为 (batch_size, seq_len) 的长整型张量，表示当前已生成的 token 序列（条件上下文）
        :param max_new_tokens: 指定要新生成的 token 数量
        :param temperature: 控制生成随机性的标量，>0。值越小（<1.0）会使高概率 token 更可能被选中，模型更确定性；值越大（>1.0）会软化概率分布，增加多样性
        :param top_k: 可选整数。如果设置，则每一步生成只从概率最高的前 k 个 token 中采样，其余 token 的概率被置零。这能避免模型采样到极低概率的 token，平衡生成质量与多样性
        """
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
            output[output < top_k_values[:, [-1]]] = float("-inf")

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
                logits[logits < top_k_values[:, [-1]]] = float("-inf")

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

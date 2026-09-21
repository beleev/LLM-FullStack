"""
DeepSeek-V3 (2024) / V3.2 (2025) — "大容量, 小推理" 的 decoder-only LM

相对 LLaMA 改了两处, V3.2 再改一处:
    FFN → DeepSeekMoE : sigmoid 打分 + top-k + 共享专家; 671B 总参, 每 token 只激活 ~37B
    GQA → MLA         : KV cache 每 token 只存 (r + rope) 个数, 而不是 2·Hkv·Dh
    V3.2: MLA → DSA   : Lightning Indexer 选 top-k 个 key, 注意力 O(L²) → O(L·k)

Aux-loss-free 负载均衡 (本文件的看点): aux loss 会和 LM loss 抢梯度。V3 改用不参与梯度的
per-expert bias, 只影响 "选谁", 不影响 "权重多少"; 每步训练后:
    bias_i += γ · sign(mean_load − load_i)        # 过载的专家降 bias, 欠载的升

读代码时盯住: DeepSeekMoE.forward 里的 select_scores (带 bias, 只用于 top-k)
             与 sigmoid_scores (不带 bias, 用于加权) —— 两者分开就是 aux-loss-free 的全部。
教学省略: Multi-Token Prediction, FP8, 专家并行, node-limited routing。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.attention import (
    MultiHeadLatentAttention,
    MultiHeadLatentSparseAttention,
)
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V3 MoE 层:
        s      = sigmoid(router(x))                      [N, E]  各专家独立打分 (不像 softmax 互相竞争)
        topk   = top-K(s + routing_bias)                 bias 只参与选择
        w_i    = s_i / Σ_{j∈topk} s_j                    权重用不带 bias 的原始分
        y      = Σ_{i∈topk} w_i · routed_i(x) + Σ shared_j(x)

    共享专家始终激活, 兜住通用模式, 让细粒度的 routed 专家可以放心专业化。
    不含 dropout: 残差 dropout 由外层 Block 统一做一次。

    forward 返回 (output, routing_info):
        router_logits [N, E] (未 detach) / selected_experts [N, K] /
        routing_weights [N, K] / routing_probs [N, E] (sigmoid 原始分, 行和 ≠ 1)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 6,
        use_aux_free_bias: bool = True,
    ):
        super().__init__()

        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, num_routed_experts, bias=False)

        # buffer 而非 Parameter: 不走梯度, 由 update_routing_bias 按规则更新;
        # persistent=True → 进 state_dict, checkpoint 才能复现路由
        if use_aux_free_bias:
            self.register_buffer("routing_bias", torch.zeros(num_routed_experts), persistent=True)
        else:
            self.routing_bias = None

        self.routed_experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(num_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(num_shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [N, D], N = B*T; 路由是 token 级的

        # 共享专家: 所有 token 都过
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        router_logits = self.router(x_flat)            # [N, E]
        sigmoid_scores = torch.sigmoid(router_logits)  # [N, E]

        # bias 只改 "选谁"
        select_scores = sigmoid_scores
        if self.routing_bias is not None:
            select_scores = sigmoid_scores + self.routing_bias
        _, selected_experts = torch.topk(select_scores, self.top_k, dim=-1)  # [N, K]

        # 权重取自不带 bias 的分数, 再归一到和为 1
        topk_sigmoid = sigmoid_scores.gather(-1, selected_experts)  # [N, K]
        routing_weights = topk_sigmoid / (topk_sigmoid.sum(dim=-1, keepdim=True) + 1e-9)

        # 教学实现: 按专家循环; 工业实现用 grouped GEMM / 专家并行
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            # token_idx: 哪些 token 选了专家 i; nth: 是该 token 的第几个选择
            token_idx, nth = torch.where(selected_experts == i)
            if token_idx.numel() == 0:
                continue
            w = routing_weights[token_idx, nth].unsqueeze(-1)  # [m, 1]
            routed_out.index_add_(0, token_idx, expert(x_flat[token_idx]) * w)

        out = (shared_out + routed_out).view(B, T, D)
        routing_info = {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
            "routing_weights": routing_weights,
            "routing_probs": sigmoid_scores,
        }
        return out, routing_info

    @torch.no_grad()
    def update_routing_bias(self, selected_experts: torch.Tensor, gamma: float = 1e-3) -> torch.Tensor:
        """
        Aux-loss-free 均衡 (每个训练 step 后调一次): bias_i += γ·sign(mean_load − load_i)。
        只用 sign: 步长恒为 γ, 与 batch 大小无关。返回本 batch 的 load [E] 供监控。
        """
        load = torch.bincount(
            selected_experts.flatten(), minlength=self.num_routed_experts
        ).float()  # [E] 每个专家本 batch 接到的 token 数
        if self.routing_bias is not None:
            self.routing_bias += gamma * torch.sign(load.mean() - load)
        return load


class DeepSeekBlock(nn.Module):
    """
    Pre-RMSNorm Block:  x → norm → MLA → +  → norm → MoE → +
    attn_cls 是子类唯一需要改的地方 (V3.2 换成 DSA), 多余的 **attn_kwargs 原样传给它。
    """

    attn_cls = MultiHeadLatentAttention

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 6,
        dropout: float = 0.1,
        latent_dim: Optional[int] = None,
        qk_rope_head_dim: int = 32,
        use_aux_free_bias: bool = True,
        **attn_kwargs,
    ):
        super().__init__()

        self.attn = self.attn_cls(
            d_model=d_model,
            num_heads=n_heads,
            latent_dim=latent_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            **attn_kwargs,
        )
        self.moe = DeepSeekMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            use_aux_free_bias=use_aux_free_bias,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # 残差 dropout, 每个子层恰好一次

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, D]
        mask: Optional[torch.Tensor] = None,     # [1, T, S]
        rope: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.norm1(x)
        h = self.attn(q=h, mask=mask, rope=rope, position_ids=position_ids, cache=cache)
        x = x + self.dropout(h)

        h, routing_info = self.moe(self.norm2(x))
        x = x + self.dropout(h)

        # DSA 的 indexer 对齐 loss 搭 routing_info 的便车出去 (MoELMLoss 会按权重加进总 loss)
        index_loss = getattr(self.attn, "last_index_loss", None)
        if index_loss is not None:
            routing_info["index_loss"] = index_loss
        return x, routing_info


class DeepSeekV32Block(DeepSeekBlock):
    """V3.2 Block: 只把 attention 换成 MLA + Lightning Indexer (DSA), 其余完全继承。
    额外 kwargs: sparse_top_k, indexer_heads, indexer_head_dim。"""

    attn_cls = MultiHeadLatentSparseAttention


class DeepSeekV3(GenerationMixin, nn.Module):
    """
    idx → Embed·sqrt(D) → N × DeepSeekBlock → RMSNorm → lm_head (与 embedding 共享权重)

    forward 返回 (logits, all_routing_info); generate() 来自 GenerationMixin (MLA latent cache)。

    Args:
        latent_dim:        MLA 的 kv_lora_rank r
        qk_rope_head_dim:  MLA 解耦 RoPE 段维度 (RoPE 模块按它建, 不是 d_model / n_heads)
        d_ff:              每个专家的 SwiGLU 隐藏维度, 默认 int(4·D·2/3)
        **block_kwargs:    原样传给 block_cls (V3.2 用它传 sparse_top_k / indexer_heads / ...)
    """

    block_cls = DeepSeekBlock

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 2048,
        n_heads: int = 16,
        num_layers: int = 24,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 6,
        max_len: int = 4096,
        dropout: float = 0.1,
        latent_dim: Optional[int] = None,
        d_ff: Optional[int] = None,
        qk_rope_head_dim: int = 32,
        use_aux_free_bias: bool = True,
        **block_kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(qk_rope_head_dim, max_len)

        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)  # SwiGLU 有 3 个矩阵, 乘 2/3 保持参数量 ≈ 4D 的普通 FFN

        # 每层只建一次: V3.2 通过 block_cls 换 Block, 而不是先建 V3 的层再整体替换
        self.layers = nn.ModuleList(
            [
                self.block_cls(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    num_routed_experts=num_routed_experts,
                    num_shared_experts=num_shared_experts,
                    top_k=top_k,
                    dropout=dropout,
                    latent_dim=latent_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    use_aux_free_bias=use_aux_free_bias,
                    **block_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        causal = build_causal_mask(max_len, device=torch.device("cpu"))  # [1, L, L]
        self.register_buffer("causal_mask", causal, persistent=False)

        # 默认 N(0,1) embedding + tying + ·sqrt(D) 会让初始 CE ≈ 250; N(0, 0.02²) 后 ≈ ln V
        init_weights(self)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(
        self, idx: torch.Tensor, cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """idx [B, T] → (logits [B, T, V], 每层一份 routing_info)。"""
        B, T = idx.shape
        past = cache.pos if cache is not None else 0  # 已缓存 token 数: 同时平移 RoPE 位置和 mask 行
        position_ids = torch.arange(past, past + T, device=idx.device)
        mask = self._causal_mask(past + T)[:, past:, :]  # [1, T, past+T]

        x = self.token_embedding(idx) * math.sqrt(self.d_model)  # [B, T, D]

        all_routing_info: List[Dict[str, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            x, routing_info = layer(
                x, mask=mask, rope=self.rope, position_ids=position_ids,
                cache=cache.layers[i] if cache is not None else None,
            )
            all_routing_info.append(routing_info)
        if cache is not None:
            cache.pos += T

        return self.lm_head(self.ln_f(x)), all_routing_info

    @torch.no_grad()
    def update_routing_bias(
        self, all_routing_info: List[Dict[str, torch.Tensor]], gamma: float = 1e-3
    ) -> torch.Tensor:
        """optimizer.step() 之后调用; 用 forward 已返回的 routing_info。返回各层 load [L, E]。"""
        return torch.stack([
            layer.moe.update_routing_bias(info["selected_experts"], gamma)
            for layer, info in zip(self.layers, all_routing_info)
        ])

    def get_num_active_params(self) -> Dict[str, int]:
        """MoE 的卖点: 总参数 (容量) ≫ 每 token 激活参数 (算力)。routed 专家按 K/E 计 (假设路由均匀)。"""
        count = lambda m: sum(p.numel() for p in m.parameters())
        moe = self.layers[0].moe
        shared = sum(count(l.moe.shared_experts) for l in self.layers)
        routed = sum(count(l.moe.routed_experts) for l in self.layers)
        total = count(self)  # tied lm_head 与 embedding 是同一个 Parameter, 不会重复计
        routed_active = routed * moe.top_k // moe.num_routed_experts
        return {
            "total_params": total,
            "active_params": total - routed + routed_active,
            "moe_total_params": shared + routed,
            "moe_active_params": shared + routed_active,
        }


class DeepSeekV3_2(DeepSeekV3):
    """
    DeepSeek-V3.2 = V3 + DSA。MLA 解决了 cache 体积, 但注意力算力仍是 O(L²);
    DSA 让每个 query 只看 indexer 选出的 top-k 个 key。

    额外构造参数 (经 **block_kwargs 传到 MultiHeadLatentSparseAttention):
        sparse_top_k=128, indexer_heads=4, indexer_head_dim=None
    训练: routing_info[i]["index_loss"] 是第 i 层 indexer 的对齐 KL, 用
          MoELMLoss(index_loss_weight=...) 加进总 loss; 否则 indexer 永远拿不到梯度。
    """

    block_cls = DeepSeekV32Block

    def set_dense_warmup(self, flag: bool) -> None:
        """True: 注意力走稠密, indexer 只旁听学习 (论文 warm-up 阶段); False: 真正 top-k 稀疏。"""
        for layer in self.layers:
            layer.attn.dense_warmup = flag

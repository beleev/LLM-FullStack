"""
DeepSeek-V3 / V3.2 模型模块

实现 DeepSeek-V3 (DeepSeek-AI, 2024) 与 DeepSeek-V3.2 (2025) 的核心架构。

设计哲学:
    DeepSeek-V3 把 GPT 范式推向"大容量, 小推理":
        - 总参数 671B, 但每个 token 只激活 ~37B (MoE 稀疏化)
        - 长上下文下 KV cache 是显存瓶颈, 用 MLA 把每头 KV 压成低秩潜变量
    DeepSeek-V3.2 进一步在 MLA 之上加 DSA (DeepSeek Sparse Attention):
        Lightning Indexer 选出 top-k 关键 token, 让注意力从 O(T^2) 接近线性。

核心技术点:
- MLA (Multi-Head Latent Attention):
    KV 低秩压缩 + 解耦 RoPE。
    动机: 长上下文推理时, KV cache 占显存远超模型权重; 把 KV 压到一个
    小维度 latent, cache 体积线性下降, 解耦的 RoPE 段单独保留位置信息。
- SwiGLU FFN: 比 GELU 更优的门控激活, 已成 LLaMA/PaLM 等的标配
- RMSNorm: 比 LayerNorm 少一次均值计算, 在 LLM 上效果相当但更快
- Fine-grained Routed + Shared Experts:
    routed 专家通过 router 稀疏激活 (大容量); shared 专家始终激活
    (托底通用能力, 防止稀有 token 学不到东西)
- 路由: sigmoid gating → top-k → renormalize (V3 官方做法, 非 softmax)
- Aux-loss-free 负载均衡:
    传统做法用 auxiliary loss 推动专家被均匀使用, 但会与主 loss 拉锯。
    V3 改为可学习/手动调节的 per-expert bias, 仅影响"被选中与否",
    不污染加权权重, 训练目标更纯净。
- Multi-Token Prediction (训练目标, 教学版未实现):
    一次预测未来多个 token, 提高数据利用率、对 speculative decoding 友好。
- V3.2 DSA: 在 MLA 前接 Lightning Indexer, 做长上下文稀疏 top-k 注意力。
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
from llm_models.utils.masks import build_causal_mask


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V3 风格 MoE 层

    路由公式 (与官方 V3 一致):
        s_i    = sigmoid( router_logits_i )              # 原始 gating score
        s_i'   = s_i + bias_i                            # 仅用于 top-k 选择
        topk   = argmax_k over {s_i' : i = 1..E}         # 选 k 个专家
        w_i    = s_i / Σ_{j∈topk} s_j   if i ∈ topk     # top-k 再归一化
                 0                       otherwise
        y      = Σ w_i * expert_i(x) + Σ shared_expert(x)

    为什么要 MoE?
        把单个大 FFN 切成 E 个小 FFN, 每 token 只激活其中 k 个,
        以"E/k 倍参数量、k 倍激活量"换取容量与算力的解耦。
        DeepSeek-V3 的官方比例是 E=256, k=8 (含 1 共享), 实现 ~37B / 671B 激活。

    为什么需要"共享专家"?
        Fine-grained MoE 切得越细, 专家越容易过度专业化、对通用 token
        欠拟合。Shared Experts 始终参与, 承担 backbone-like 的通用能力,
        让 routed experts 专注差异化模式。

    为什么用 sigmoid + renormalize 而不是 softmax?
        softmax 让所有专家分数互相竞争, 选 top-k 时分布尖锐;
        DeepSeek-V3 改用 per-expert sigmoid (各自独立判断),
        再对选中的 k 个分数归一化, 路由更平滑、负载更均衡。

    为什么需要负载均衡 (aux-loss-free bias)?
        若任由 router 自由学习, 容易出现"路由坍塌": 少数热门专家被大量
        token 选中, 其余专家几乎不被使用, 退化为小模型。
        传统做法加 aux loss 强行拉平, 但与主 loss 拉锯影响收敛。
        V3 提出: 在选择阶段加一个可调 bias, 哪个专家过载就降低它的 bias,
        欠载就提高 —— bias 不参与加权计算, 不污染主梯度。
        本教学实现仅暴露 buffer, 真正的更新规则交给训练循环。

    教学注意:
        router_logits 不 detach, 保证外部计算 aux loss 时可回传到 router 权重。

    Args:
        d_model: 模型维度
        d_ff: 每个专家的 SwiGLU 隐藏维度
        num_routed_experts: 路由专家数量
        num_shared_experts: 共享专家数量 (始终激活)
        top_k: 每个 token 激活的路由专家数
        dropout: Dropout 概率
        use_aux_free_bias: 是否启用 aux-loss-free 负载均衡 bias
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 6,
        dropout: float = 0.1,
        use_aux_free_bias: bool = True,
    ):
        super().__init__()

        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.use_aux_free_bias = use_aux_free_bias

        # router 是个轻量线性层: hidden -> 每个专家一个 logits
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)

        # Aux-loss-free bias: 不经梯度更新 (用 buffer), 外部训练脚本按规则调节
        # persistent=True 让它进入 state_dict, 保证 checkpoint 能复现路由行为
        if use_aux_free_bias:
            self.register_buffer(
                "routing_bias", torch.zeros(num_routed_experts), persistent=True
            )
        else:
            self.routing_bias = None

        # routed: 通过 router 稀疏激活; shared: 始终激活
        self.routed_experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(num_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(num_shared_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            output: [B, T, d_model]
            routing_info:
                router_logits:    [N, E]  (未 detach — 保证 aux loss 可回传)
                selected_experts: [N, K]
                routing_weights:  [N, K]  (renormalized sigmoid, 匹配加权路径)
                routing_probs:    [N, E]  (sigmoid 原始分布, 监控用)
        """
        B, T, D = x.shape
        # 把 batch 与 seq 拍平: 路由是 token 级别的, 不区分 batch
        x_flat = x.view(-1, D)  # [N, D], N = B*T
        N = x_flat.size(0)

        # --- 共享专家路径: 所有 token 都过 shared expert ---
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        # --- 路由 ---
        router_logits = self.router(x_flat)                 # [N, E]
        # sigmoid (而非 softmax): 各专家独立打分, 路由更平滑
        sigmoid_scores = torch.sigmoid(router_logits)       # [N, E]

        # 加 bias 仅用于"选择", 不影响加权的原始分数
        # 这是 aux-loss-free 负载均衡的核心: bias 改 routing 不改梯度
        if self.routing_bias is not None:
            select_scores = sigmoid_scores + self.routing_bias
        else:
            select_scores = sigmoid_scores

        # top-k 选择: 每 token 选中 K 个专家
        _, selected_experts = torch.topk(select_scores, self.top_k, dim=-1)  # [N, K]

        # 从"原始" sigmoid_scores (不带 bias) 中取 top-k 对应分数, 再归一化
        # 用原始分而非 select_scores: bias 是路由扰动, 不应进入加权
        topk_sigmoid = sigmoid_scores.gather(-1, selected_experts)            # [N, K]
        # +1e-9 防数值下溢 (top-k 全 0 概率极低但需兜底)
        routing_weights = topk_sigmoid / (topk_sigmoid.sum(dim=-1, keepdim=True) + 1e-9)

        # --- 路由专家计算 ---
        # 教学实现: 按专家循环, 收集"选了我"的 token 子集再算
        # 工业实现会用 grouped GEMM 或专家并行 (EP) 在 GPU 间分布专家
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            # token_idx: 哪些 token 选中了专家 i; nth: 它是该 token 的第几号选择
            token_idx, nth = torch.where(selected_experts == i)
            if token_idx.numel() == 0:
                continue  # 没有 token 选这个专家, 跳过 (稀疏激活的体现)
            w = routing_weights[token_idx, nth].unsqueeze(-1)  # [m, 1]
            # index_add_ 处理"多个专家贡献到同一 token"的累加, 安全且原子
            routed_out.index_add_(0, token_idx, expert(x_flat[token_idx]) * w)

        out = self.dropout(shared_out + routed_out).view(B, T, D)

        routing_info = {
            "router_logits": router_logits,            # 不 detach: 让外部 aux loss 可回传
            "selected_experts": selected_experts,
            "routing_weights": routing_weights,
            "routing_probs": sigmoid_scores,
        }
        return out, routing_info


class DeepSeekBlock(nn.Module):
    """
    DeepSeek-V3 Transformer Block (Pre-RMSNorm)

    数据流:
        x -> RMSNorm -> MLA      -> Add
          -> RMSNorm -> MoE      -> Add

    与 GPTBlock 的差异:
        - LayerNorm  -> RMSNorm   (省掉均值, 计算更快, 大模型上效果相当)
        - MHA        -> MLA       (KV 低秩压缩, 节省长上下文 KV cache)
        - GELU FFN   -> MoE       (稀疏激活, 容量与算力解耦)

    MoE 返回 (out, routing_info), 因此本 Block 无法直接套用通用 PreLNBlock,
    但逻辑是完全一致的, 只是把 FFN 换成 MoE、LayerNorm 换成 RMSNorm。
    """

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
    ):
        super().__init__()

        self.attn = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=n_heads,
            latent_dim=latent_dim,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self.moe = DeepSeekMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            dropout=dropout,
            use_aux_free_bias=use_aux_free_bias,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x:            [B, T, d_model]
            mask:         因果 + padding mask
            rope:         RoPE 编码器 (作用在 MLA 的解耦 rope 段上)
            position_ids: 自定义位置, 多模态/拼接序列下需要
        Returns:
            x:            [B, T, d_model]
            routing_info: MoE 的路由信息 (供外部计算 aux loss / 监控负载)
        """
        # Self-Attention 子层 (Pre-RMSNorm + 残差)
        h = self.norm1(x)
        h = self.attn(q=h, k=h, v=h, mask=mask, rope=rope, position_ids=position_ids)
        x = x + self.dropout(h)

        # MoE 子层 (Pre-RMSNorm + 残差); 同时回吐路由信息
        h = self.norm2(x)
        h, routing_info = self.moe(h)
        x = x + self.dropout(h)
        return x, routing_info


class DeepSeekV32Block(DeepSeekBlock):
    """
    DeepSeek-V3.2 Block: 把 MLA 换成 MLA + Lightning Indexer (DSA)。

    DSA (DeepSeek Sparse Attention) 思路:
        - Lightning Indexer 用极小代价为每个 query 算出 top-k 个"最相关" key
        - 主注意力只在这 k 个 key 上计算, 把复杂度从 O(T^2) 拉向 ~O(T·k)
        - 对长上下文 (如 128K) 显著降算力, 同时保留 MLA 的 KV 压缩收益

    其余结构与 DeepSeekBlock 完全一致, 这里直接继承并只替换 attn 模块。
    """

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
        sparse_top_k: int = 128,
        indexer_heads: int = 4,
        indexer_head_dim: Optional[int] = None,
        use_aux_free_bias: bool = True,
    ):
        super().__init__(
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
        )

        # 覆盖 attn 为 DSA: 在 MLA 之上叠加 Lightning Indexer 做稀疏 top-k
        self.attn = MultiHeadLatentSparseAttention(
            d_model=d_model,
            num_heads=n_heads,
            latent_dim=latent_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            indexer_heads=indexer_heads,
            indexer_head_dim=indexer_head_dim,
            sparse_top_k=sparse_top_k,
        )


class DeepSeekV3(nn.Module):
    """
    DeepSeek-V3 模型 (教学版骨架)

    架构流程:
        Input idx -> TokenEmbed * sqrt(d_model)
                  -> N x DeepSeekBlock (MLA + RMSNorm + MoE(SwiGLU))
                  -> RMSNorm
                  -> lm_head  (与 embedding 权重共享)

    与 GPT3 的对比 (一行总结架构演进):
        GPT3:  TokenEmb -> N x [MHA + GELU-FFN + LN] -> LN -> lm_head
        V3:    TokenEmb -> N x [MLA + MoE(SwiGLU) + RMSNorm] -> RMSNorm -> lm_head

    教学省略:
        - Multi-Token Prediction 训练目标 (论文创新, 此处仍为单 token 预测)
        - FP8 混合精度训练管线
        - 专家并行 / 流水线并行 等分布式策略

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        n_heads: 注意力头数
        num_layers: 层数
        num_routed_experts: 路由专家数量
        num_shared_experts: 共享专家数量
        top_k: 每个 token 激活的路由专家数
        max_len: 最大序列长度 (缓存因果掩码大小)
        dropout: Dropout 概率
        latent_dim: MLA KV 低秩维度
        d_ff: 专家 FFN 隐藏维度 (默认 int(4 * d_model * 2/3))
    """

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
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.qk_rope_head_dim = qk_rope_head_dim

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # MLA 把 Q/K 每头维度拆成 nope + rope，RoPE 只作用在 rope 段
        # 因此 RoPE 的 head_dim 必须是 qk_rope_head_dim，而不是 d_model / n_heads
        self.rope = RotaryPositionalEncoding(qk_rope_head_dim, max_len)

        # SwiGLU 参数量约为普通 FFN 的 1.5×，按 2/3 缩放保持总量近似
        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)

        self.layers = nn.ModuleList(
            [
                DeepSeekBlock(
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
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # 预构建因果掩码 (随设备迁移；max_len 为上限，前向会切片)
        causal = build_causal_mask(max_len, device=torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        """按需切片或动态构建因果 mask, 与 GPT3 同样的两条路径。"""
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        # 容错: 运行时序列超过 max_len 时动态构建
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(
        self, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Args:
            idx: [B, T] token ids
        Returns:
            logits:           [B, T, vocab_size]
            all_routing_info: 每一层 MoE 的路由信息列表, 长度 = num_layers
                              训练侧据此计算负载均衡监控/aux loss
        """
        B, T = idx.shape

        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        mask = self._causal_mask(T)

        # 收集每层路由信息: 训练循环外部用它做负载均衡 / aux-free bias 更新
        all_routing_info: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            x, routing_info = layer(x, mask=mask, rope=self.rope)
            all_routing_info.append(routing_info)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_routing_info

    def get_num_active_params(self) -> Dict[str, int]:
        """
        计算每个 token 实际激活的参数量。

        MoE 模型的关键卖点: 总参数 (容量) >> 激活参数 (推理算力)。
        以 DeepSeek-V3 为例, 671B 总参 / 37B 激活, 推理成本接近 37B 稠密模型。

        把参数分为:
            - 非 MoE 始终激活: embedding / ln_f / attention / norms / router
            - 共享专家始终激活: 所有 shared_experts
            - 路由专家按 top_k/num_experts 比例激活
        """
        non_moe_params = sum(p.numel() for p in self.token_embedding.parameters())
        non_moe_params += sum(p.numel() for p in self.ln_f.parameters())
        # lm_head 与 token_embedding 共享权重，不重复计算

        always_active_per_layer = 0
        shared_expert_params = 0
        routed_expert_params = 0

        for layer in self.layers:
            always_active_per_layer += sum(p.numel() for p in layer.attn.parameters())
            always_active_per_layer += sum(p.numel() for p in layer.norm1.parameters())
            always_active_per_layer += sum(p.numel() for p in layer.norm2.parameters())
            always_active_per_layer += sum(p.numel() for p in layer.moe.router.parameters())
            shared_expert_params += sum(
                p.numel() for p in layer.moe.shared_experts.parameters()
            )
            routed_expert_params += sum(
                p.numel() for p in layer.moe.routed_experts.parameters()
            )

        total_params = (
            non_moe_params
            + always_active_per_layer
            + shared_expert_params
            + routed_expert_params
        )

        # 路由专家平均激活量 = 总专家参数 × (k / E)
        # 假设路由近似均匀, 是稳态训练后的合理估计
        num_routed = self.layers[0].moe.num_routed_experts
        top_k = self.layers[0].moe.top_k
        routed_active = routed_expert_params * top_k // num_routed

        active_params = (
            non_moe_params
            + always_active_per_layer
            + shared_expert_params
            + routed_active
        )

        return {
            "total_params": total_params,
            "active_params": active_params,
            "moe_total_params": shared_expert_params + routed_expert_params,
            "moe_active_params": shared_expert_params + routed_active,
        }


class DeepSeekV3_2(DeepSeekV3):
    """
    DeepSeek-V3.2 模型 (DSA + MLA + MoE)

    相对 V3 的唯一差别: 把每层的 attention 换成 MLA + Lightning Indexer 稀疏化。
    通过继承 + 替换 layers 来表达这一点, 避免重复 forward / get_num_active_params 逻辑。

    为什么需要 V3.2?
        MLA 解决了"KV cache 占显存"的问题, 但注意力本身的算力仍是 O(T^2)。
        当上下文延伸到 100K+ 时, 单次 forward 的 attention 开销主导端到端成本。
        DSA 让每个 query 只看 top-k 关键 token, 是稀疏注意力的工程级落地。
    """

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
        sparse_top_k: int = 128,
        indexer_heads: int = 4,
        indexer_head_dim: Optional[int] = None,
        use_aux_free_bias: bool = True,
    ):
        # 先走父类的基础构建 (会建一个 V3 layer 列表)
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            max_len=max_len,
            dropout=dropout,
            latent_dim=latent_dim,
            d_ff=d_ff,
            qk_rope_head_dim=qk_rope_head_dim,
            use_aux_free_bias=use_aux_free_bias,
        )

        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)

        # 替换为 V3.2 Block
        self.layers = nn.ModuleList(
            [
                DeepSeekV32Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    num_routed_experts=num_routed_experts,
                    num_shared_experts=num_shared_experts,
                    top_k=top_k,
                    dropout=dropout,
                    latent_dim=latent_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    sparse_top_k=sparse_top_k,
                    indexer_heads=indexer_heads,
                    indexer_head_dim=indexer_head_dim,
                    use_aux_free_bias=use_aux_free_bias,
                )
                for _ in range(num_layers)
            ]
        )

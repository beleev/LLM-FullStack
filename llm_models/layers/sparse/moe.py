"""
Mixtral 风格稀疏 MoE 前馈层 — 把一个大 FFN 换成 E 个小 FFN, 每 token 只走其中 K 个

解决什么: dense FFN 的参数量和算力绑死; MoE 让容量 (E 个专家) 与每 token 算力 (K 个) 解耦。
    probs   = softmax(router(x))              [N, E]
    topk    = top-K(probs)                    [N, K]
    weights = topk_probs / Σ topk_probs       K 个权重和为 1
    y       = Σ_{i∈topk} weights_i · expert_i(x)
关键数字: Mixtral 8x7B: E=8, K=2 → 总参 47B, 每 token 激活 ~13B。
与 DeepSeekMoE (models/moe/deepseekV3.py) 对照: 那边是 sigmoid 打分 + 共享专家 + aux-loss-free bias;
这边没有 bias, 负载均衡全靠外部 aux loss (training/loss.py::MoELMLoss)。
读代码时盯住: selected_experts [N, K] —— 它决定了谁干活, 且不可导 (梯度只走 routing_weights)。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.feedforward import SwiGLUFeedForward


class MixtralMoE(nn.Module):
    """
    Args:
        d_model / d_ff: 模型维度 / 每个专家的 SwiGLU 隐藏维度
        num_experts:    E (Mixtral 8x7B 是 8)
        top_k:          K (Mixtral 8x7B 是 2)

    不含 dropout: 残差 dropout 统一由外层 Block 做一次 (以前这里和 Block 各做一次 = 两次)。
    forward 返回 (output, routing_info):
        router_logits [N, E] (未 detach, aux loss 要回传) / selected_experts [N, K] /
        routing_weights [N, K] / routing_probs [N, E]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()

        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) 不能大于 num_experts ({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k

        # router: 线性层打 logits; 无 bias 与现代 LLM 保持一致
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(num_experts)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [N, D], N = B*T

        # 1) 路由打分 (Mixtral 用 softmax, 让所有专家分数总和为 1)
        router_logits = self.router(x_flat)                    # [N, E]
        routing_probs = F.softmax(router_logits, dim=-1)       # [N, E]

        # 2) top-k 选择 + 再归一化
        topk_probs, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # [N, K], [N, K]
        # 归一化让 K 个权重和=1; 与 DeepSeek 不同的是这里用的已经是 softmax 值
        # +1e-9 防万一 (top-k 全 0 概率极低但需兜底)
        routing_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 3) 按专家聚合: 每个专家挑出"选了我"的 token 算一次, 按权重加和
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # token_idx: 哪些 token 选中了专家 i; nth: 它是该 token 的第几号选择
            token_idx, nth = torch.where(selected_experts == i)
            if token_idx.numel() == 0:
                continue
            w = routing_weights[token_idx, nth].unsqueeze(-1)  # [m, 1]
            output.index_add_(0, token_idx, expert(x_flat[token_idx]) * w)

        output = output.view(B, T, D)

        routing_info = {
            "router_logits": router_logits,          # 未 detach, 可回传 aux loss
            "selected_experts": selected_experts,
            "routing_weights": routing_weights,
            "routing_probs": routing_probs,
        }
        return output, routing_info

"""
Mixture-of-Experts (MoE) 稀疏前馈模块 — 经典 Mixtral 风格

历史脉络:
    - Switch Transformer (Fedus et al., 2021): 首次在 transformer 上大规模上 MoE,
      单专家路由 (top-1) + auxiliary loss 做负载均衡
    - Mixtral 8x7B (Jiang et al., 2024): softmax + top-2 专家, 每层 8 个专家,
      成为"经典稀疏 MoE"的事实模板
    - DeepSeek-V3 (2024): fine-grained + shared experts + aux-loss-free bias
      (见 models/deepseekV3.py::DeepSeekMoE)

本文件提供 Mixtral 风格的 **最简版** MoE:
    - softmax(router_logits) → top-k → 选中分数归一化
    - 没有共享专家 (所有专家都是 routed)
    - 没有 aux-loss-free bias (路由均衡通过外部 Switch-style aux loss 实现)

与 DeepSeekMoE 的差异一目了然 (教学对照):
              Mixtral (本文件)           DeepSeek-V3 (deepseekV3.py)
    路由打分   softmax                    sigmoid (各专家独立)
    归一化     softmax 本身即归一         top-k 后再 renormalize
    共享专家   无                         有 (始终激活)
    负载均衡   依赖外部 aux loss          aux-loss-free bias + 可选 aux loss

返回约定与 DeepSeekMoE 一致:
    (output, routing_info)
    routing_info 字段:
        router_logits:    [N, E] 未 detach, 便于外部 aux loss 回传
        selected_experts: [N, K]
        routing_weights:  [N, K] (softmax 归一化后)
        routing_probs:    [N, E] (softmax 原始分布)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.feedforward import SwiGLUFeedForward


class MixtralMoE(nn.Module):
    """
    Mixtral 风格稀疏 MoE 层

    公式:
        logits = router(x)                               # [N, E]
        probs  = softmax(logits, dim=-1)                 # [N, E]
        topk_probs, topk_idx = topk(probs, k)            # [N, K]
        weights = topk_probs / topk_probs.sum(dim=-1)    # 再归一化, 保证权重和=1
        y = Σ_{i in topk} weights_i · expert_i(x)

    工程要点:
        - 专家用 SwiGLU FFN (与 Mixtral 8x7B 官方实现一致)
        - 教学实现按专家 for-loop 聚合, 简单直观;
          生产实现会用 grouped GEMM / 专家并行

    Args:
        d_model: 模型维度
        d_ff:    每个专家的 SwiGLU 隐藏维度
        num_experts: 专家数量 (Mixtral 8x7B 是 8)
        top_k:       每 token 激活的专家数 (Mixtral 8x7B 是 2)
        dropout:     残差 dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
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
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, T, D]
        Returns:
            output: [B, T, D]
            routing_info: 见模块 docstring
        """
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

        output = self.dropout(output).view(B, T, D)

        routing_info = {
            "router_logits": router_logits,          # 未 detach, 可回传 aux loss
            "selected_experts": selected_experts,
            "routing_weights": routing_weights,
            "routing_probs": routing_probs,
        }
        return output, routing_info

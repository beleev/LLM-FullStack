"""
归一化模块

包含:
- RMSNorm: 均方根归一化 (Root Mean Square Layer Normalization)

演进历史与动机:
    1. LayerNorm (Ba et al., 2016) — 原始 Transformer 使用
       公式: (x - mean) / sqrt(var + eps) * gain + bias
    2. RMSNorm (Zhang & Sennrich, 2019) — LLaMA / Qwen / DeepSeek / Gemma 标配
       公式: x / RMS(x) * gain   (省去 mean-centering 和 bias)

为什么去掉均值中心化:
    论文实验显示 LayerNorm 中起主要作用的是 "缩放" 而非 "去均值"。
    去掉 mean & bias 后:
        - 计算量减少 ~25% (省一次均值统计 + 一次减法 + bias 加法)
        - 数值更稳：不再因均值漂移引入额外噪声
        - 大模型 / 长序列下与 LayerNorm 效果差异极小 (<0.1 PPL)
    这是 "更少 = 更好" 的典型例子，故现代 LLM 几乎全面切换。
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMS Normalization (均方根归一化)

    公式:
        RMS(x) = sqrt( mean(x^2) + eps )
        RMSNorm(x) = (x / RMS(x)) * weight

    与 LayerNorm 的关键区别:
        1. 不减去均值 (no mean centering) → 少一次统计与一次减法
        2. 不使用 bias → 少一组参数，且推理更快
        3. 只做缩放 (scale)，不做偏移 (shift)
        4. 计算量减少约 25%，大模型上效果几乎等同 LayerNorm

    Args:
        d_model: 归一化维度 (在最后一维上做归一化)
        eps: 防止除零的小常数 (LLaMA/Qwen 常用 1e-6 ~ 1e-5)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()

        self.eps = eps
        # 可学习的缩放参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [..., d_model]

        Returns:
            归一化后的张量，形状不变
        """
        # 1) RMS = sqrt(E[x^2] + eps)，eps 加在 sqrt 内部 (LLaMA 风格)
        #    keepdim=True 保留维度便于后续广播除法
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 2) 归一化后乘以可学习缩放 weight (相当于让模型自己决定每个通道的方差)
        return (x / rms) * self.weight

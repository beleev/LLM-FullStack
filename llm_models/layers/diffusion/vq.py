"""
Vector Quantization (VQ) 模块 — 把连续向量映射到离散码本

论文出处:
    "Neural Discrete Representation Learning" (van den Oord et al., 2017)
    VQ-VAE 的核心构件。

为什么要向量量化?
    - VQ-VAE 把图像压成离散 token 后, 可直接送 GPT 风格自回归 LM 建模
      (DALL-E, Parti, LlamaGen, VAR)
    - 离散表示天然适配 codec (音频/图像), 便于与语言模型对齐
    - 相比 KL-VAE, VQ 让潜空间严格离散, "像词汇表一样有限可枚举"

核心流程:
    z_e = encoder(x)                       # 连续 [B, D, ...] 或 [B, T, D]
    // 最近邻查码本
    distances = ||z_e - codebook||^2
    indices   = argmin(distances)
    z_q = codebook[indices]                # 量化后的离散向量
    // 反传技巧: straight-through estimator
    z_q = z_e + stop_gradient(z_q - z_e)   # 前向用 z_q, 反向梯度走 z_e

三项 loss (训练时):
    1. 重建 loss:   ||x - decoder(z_q)||^2   (由外部 decoder 算)
    2. codebook loss: ||sg(z_e) - codebook||^2  (拉码本靠近 encoder 输出)
    3. commitment loss: β · ||z_e - sg(codebook)||^2  (防 encoder 飘走)

其中 sg = stop_gradient (即 .detach())。β 常取 0.25。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    VQ 码本 + straight-through 量化 (教学版)

    输入与输出形状保持一致: [B, T, D] (序列) 或 [B, D, H, W] (图像, 会内部 flatten)。
    本实现同时支持两种形态。

    Args:
        num_embeddings: 码本大小 K (VAR / LlamaGen 典型 4096~16384)
        embedding_dim:  每个码字维度 D
        beta:           commitment loss 系数, VQ-VAE 论文默认 0.25
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # 码本实现为 Embedding: weight.shape = [K, D]
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        # 均匀初始化 (VQ-VAE 官方做法)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def _quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        最近邻码字查找。

        Args:
            z: [N, D] flatten 后的连续向量
        Returns:
            z_q: [N, D] 量化后的向量
            indices: [N] 对应码字 ID
        """
        # 展开公式: ||z - e||^2 = ||z||^2 + ||e||^2 - 2·z·e^T
        # 避免显式 subtract+square, 利用 matmul 加速
        z_sq = (z**2).sum(dim=-1, keepdim=True)                     # [N, 1]
        e_sq = (self.codebook.weight**2).sum(dim=-1)                # [K]
        ze = z @ self.codebook.weight.t()                           # [N, K]
        distances = z_sq + e_sq.unsqueeze(0) - 2 * ze               # [N, K]

        indices = distances.argmin(dim=-1)                          # [N]
        z_q = self.codebook(indices)                                # [N, D]
        return z_q, indices

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z: 连续特征, 形状 [B, T, D] 或 [B, D, H, W]
        Returns:
            z_q: 与 z 同形 (straight-through 处理后, 前向 = 离散, 反向 = 直通)
            info: 含 indices, vq_loss, perplexity (码本使用度) 的 dict
        """
        orig_shape = z.shape
        is_image = z.dim() == 4

        if is_image:
            # [B, D, H, W] → [B, H, W, D] → [N, D]
            z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.reshape(-1, self.embedding_dim)  # [N, D]

        z_q_flat, indices = self._quantize(z_flat)

        # codebook loss: 拉码字靠近 encoder 输出 (码字端梯度)
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        # commitment loss: 防 encoder 飘离码字 (encoder 端梯度)
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through: 前向用 z_q, 反传梯度等价于 z (跳过 argmin 不可导)
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # 还原形状
        z_q = z_q_flat.view(*z.shape)
        if is_image:
            z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        # perplexity: 码本实际使用多样性 (训练中监控码本坍塌)
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_embeddings).float()   # [N, K]
            probs = one_hot.mean(dim=0)                                  # [K]
            perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        indices_reshaped = indices.view(*orig_shape[:-1]) if not is_image else \
                           indices.view(orig_shape[0], orig_shape[2], orig_shape[3])

        return z_q, {
            "indices": indices_reshaped,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """反向查表: 把 [..., ] 的整数索引映射回 [..., D] 码字向量。"""
        return self.codebook(indices)

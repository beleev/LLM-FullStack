"""
Vector Quantization — 把连续向量吸附到离散码本 (VQ-VAE, van den Oord et al., 2017)

解决的问题: KL-VAE 的 latent 是连续的, 没法交给 "预测下一个 token" 的自回归模型;
VQ 让潜空间像词表一样有限可枚举 (DALL-E / LlamaGen / VAR 的前提)。

    idx = argmin_k ||z_e - e_k||²          z_q = e[idx]
    z_q = z_e + sg(z_q - z_e)              # straight-through: 前向离散, 反向直通 encoder
    vq_loss = ||sg(z_e) - z_q||² + β·||z_e - sg(z_q)||²   # 拉码本 + commitment, β=0.25

本文件两个类:
    VectorQuantizer — 单尺度码本
    MultiScaleVQ    — VAR 的多尺度残差量化 (1×1 → 2×2 → 4×4, 每级量化上一级留下的残差)
读代码时盯住: MultiScaleVQ.forward 里的 residual 与 f_hat 如何此消彼长。
"""

from typing import Dict, List, Sequence, Tuple

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


class MultiScaleVQ(nn.Module):
    """
    VAR 的多尺度残差量化器 (Tian et al., 2024, Algorithm 1/2), 所有尺度共享一个码本。

    编码 (粗 → 细), f 为 encoder 输出 [B, D, H, W], f_hat 为 "到目前为止的重建":
        for s in scales:                      # e.g. (1, 2, 4), 最后一级 = H
            r_s   = down(f - f_hat, s)        # 把还没解释掉的残差缩到 s×s
            idx_s = nearest_code(r_s)         # 这一级的 token [B, s, s]
            f_hat = f_hat + up(e[idx_s], H)   # 粗尺度负责低频, 细尺度只补细节

    于是一张图 = token 金字塔 (idx_1, ..., idx_K), 共 Σ s² 个 token;
    自回归的 "一步" 是一整张 s×s 的 token map, 而不是光栅序里的一个 token。

    简化: 原版每级 up 之后还有一个小卷积 φ_k, 这里省略。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scales: Sequence[int] = (1, 2, 4),
        beta: float = 0.25,
    ):
        super().__init__()
        self.scales = tuple(scales)
        self.beta = beta
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)   # 只借它的码本与最近邻

    @staticmethod
    def _down(x: torch.Tensor, s: int) -> torch.Tensor:
        return F.interpolate(x, size=(s, s), mode="area")                # [B, D, s, s]

    @staticmethod
    def _up(x: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    def lookup(self, idx: torch.Tensor, size: int) -> torch.Tensor:
        """一级 token [B, s, s] → 码字 → 上采样到 [B, D, size, size] (它对 f_hat 的贡献)。"""
        e = self.vq.decode_indices(idx).permute(0, 3, 1, 2)              # [B, D, s, s]
        return self._up(e, size)

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            f: [B, D, H, W], H == W == scales[-1]
        Returns:
            f_hat: [B, D, H, W] 多尺度重建 (straight-through)
            info:  indices (List[[B, s, s]]), vq_loss, perplexity
        """
        B, D, H, _ = f.shape
        assert H == self.scales[-1], f"latent 边长 {H} 必须等于最细尺度 {self.scales[-1]}"

        f_hat = torch.zeros_like(f)
        indices: List[torch.Tensor] = []
        vq_loss = f.new_zeros(())
        for s in self.scales:
            r = self._down(f.detach() - f_hat.detach(), s)               # [B, D, s, s]
            _, idx = self.vq._quantize(r.permute(0, 2, 3, 1).reshape(-1, D))
            idx = idx.view(B, s, s)
            indices.append(idx)
            f_hat = f_hat + self.lookup(idx, H)                          # 梯度只流向码本
            # 每一级的 "累计重建" 都要贴近 f: 码本端 + β·encoder 端 (commitment)
            vq_loss = vq_loss + F.mse_loss(f_hat, f.detach()) + self.beta * F.mse_loss(f, f_hat.detach())
        vq_loss = vq_loss / len(self.scales)

        with torch.no_grad():                                            # 码本使用度, 监控坍塌
            probs = torch.bincount(
                torch.cat([i.flatten() for i in indices]), minlength=self.vq.num_embeddings
            ).float()
            probs = probs / probs.sum()
            perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        f_hat = f + (f_hat - f).detach()                                 # straight-through
        return f_hat, {"indices": indices, "vq_loss": vq_loss, "perplexity": perplexity}

    def next_scale_inputs(self, indices: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Teacher forcing 用: 第 k 级 (k ≥ 2) 的 Transformer 输入 = down(前 k-1 级的累计重建, s_k)。
        只用到 indices[:-1] —— 最细一级的 token 只当 label, 不当输入。
        Returns: K-1 个 [B, D, s_k, s_k]
        """
        H = self.scales[-1]
        f_hat, outs = 0.0, []
        for idx, s_next in zip(indices[:-1], self.scales[1:]):
            f_hat = f_hat + self.lookup(idx, H)                          # [B, D, H, H]
            outs.append(self._down(f_hat, s_next))
        return outs

    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """token 金字塔 → f_hat [B, D, H, H] (各级贡献求和)。"""
        return sum(self.lookup(idx, self.scales[-1]) for idx in indices)

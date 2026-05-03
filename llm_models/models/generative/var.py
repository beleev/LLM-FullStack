"""
VAR — Visual AutoRegressive 图像生成

论文出处:
    "Visual Autoregressive Modeling: Scalable Image Generation via
     Next-Scale Prediction" (Tian et al., 2024, NeurIPS Best Paper)
    同路线: Parti / LlamaGen 用 next-token 自回归生成 VQ token 序列。

在本库中的位置:
    与扩散 (DiT / MM-DiT) 完全不同的生成范式:
        - 扩散: 连续 latent + 迭代去噪
        - 自回归: 离散 VQ token + next-token 预测, **完全复用 GPT 框架**

为什么 VAR 能做 SOTA 图像生成?
    把图像先压成 16×16 (或多尺度) 的离散 token grid, 再用一个标准 GPT 按
    raster-scan (或 next-scale) 顺序预测下一个 token;
    只要 codebook 足够丰富 + LM 足够大, 质量可达扩散水平。
    对比优势: 共享 LLM 基础设施 (KV cache, batching, scaling law)。

本库提供的最小教学版:
    - VectorQuantizer (在 layers/vq.py) 提供码本
    - 外部 VQ-VAE 编码器 (本库用简化版, 基于 ImageVAEEncoder 的结构) 把图像压到
      [B, H', W'] 的码本索引网格
    - 把网格按 raster scan 展平成 [B, H'·W'] token 序列, 前置 [BOS]
    - 用一个标准 Decoder-only LM (这里直接复用 GPT3) 做 next-token 预测
    - 推理时自回归生成 token, 查码本得向量, 喂 decoder 重建图像

为了不重复造轮子, 本文件只实现:
    - ImageTokenizer: 薄薄一层 VQ 编码器 (conv + VQ + decoder)
    - VARModel:       tokenizer + GPT3 生成器 + 转换函数

教学时重点讲: **图像生成就是 next-token 预测**, 所有已学的 LLM 训练技巧
(causal mask, teacher forcing, KV cache, temperature sampling) 全部适用。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.diffusion.vq import VectorQuantizer
from llm_models.models.language_models.gpt3 import GPT3
from llm_models.models.generative.vae import ImageVAEDecoder, ImageVAEEncoder


class ImageTokenizer(nn.Module):
    """
    薄 VQ-VAE: 只做 encode → quantize → decode 的最小闭环,
    用作 VAR 的图像 <-> token 桥梁。

    设计:
        - 直接复用 ImageVAE{Encoder,Decoder} 的卷积主干
        - 瓶颈用 VectorQuantizer 而非高斯采样 → 得到离散索引
        - 下采样倍数决定 "每张图产生多少 token"
          (e.g. 256×256 图, 下采 4 次 → 16×16 = 256 个 token)

    Args:
        image_size:     输入图像边长 (H=W)
        image_channels: RGB 3 通道
        codebook_size:  码本大小 K
        latent_dim:     码字维度 D
        base_channels:  conv 主干通道
        levels:         下采样次数, 决定 grid_size
    """

    def __init__(
        self,
        image_size: int = 64,
        image_channels: int = 3,
        codebook_size: int = 1024,
        latent_dim: int = 64,
        base_channels: int = 64,
        levels: int = 2,
    ):
        super().__init__()

        self.image_size = image_size
        self.levels = levels
        if image_size % (2 ** levels) != 0:
            raise ValueError(f"image_size {image_size} 必须能被 2^{levels} 整除")
        self.grid_size = image_size // (2 ** levels)
        self.num_tokens = self.grid_size ** 2

        self.encoder = ImageVAEEncoder(
            in_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, downsample_levels=levels,
        )
        self.decoder = ImageVAEDecoder(
            out_channels=image_channels, base_channels=base_channels,
            latent_dim=latent_dim, upsample_levels=levels,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size, embedding_dim=latent_dim,
        )

    def encode_to_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        图像 → 离散 token 网格索引

        Args:
            x: [B, 3, H, W]
        Returns:
            indices: [B, grid_size, grid_size]
            info:    VectorQuantizer 返回的 loss / perplexity 等
        """
        # encoder 返回 (mean, logvar); VQ 路径直接用 mean 作为连续特征 (去掉随机采样)
        z, _ = self.encoder(x)
        _, info = self.quantizer(z)   # info 含 indices & vq_loss
        return info["indices"], info

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        离散 token 网格 → 图像

        Args:
            indices: [B, H', W'] 码字 id
        Returns:
            [B, 3, H, W]
        """
        # codebook lookup: [B, H', W'] → [B, H', W', D] → [B, D, H', W']
        z_q = self.quantizer.decode_indices(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        标准 VQ-VAE 前向: 包含 straight-through 量化, 供 reconstruction 训练。
        """
        z, _ = self.encoder(x)
        z_q, info = self.quantizer(z)
        recon = self.decoder(z_q)
        return {
            "recon": recon,
            "indices": info["indices"],
            "vq_loss": info["vq_loss"],
            "perplexity": info["perplexity"],
        }


class VARModel(nn.Module):
    """
    VAR (简化版): tokenizer + 一个 GPT 自回归 LM

    流程:
        训练:
            x → tokenizer.encode → indices [B, H', W']
            flatten + 前置 [BOS] → token 序列 [B, 1 + N]
            tokens 做 teacher forcing, GPT 学 next-token 预测
        推理:
            从 [BOS] 开始自回归采样 N 个 token → reshape 成 grid
            → tokenizer.decode → 图像

    本实现与真正的 VAR 简化点:
        - VAR 原论文做 **next-scale prediction** (多分辨率递进),
          本教学版退化为 **next-token raster scan**, 概念上与 LlamaGen 等价,
          更便于教学
        - Tokenizer 独立训练; VAR 主体假设 tokenizer 已经冻结

    Args:
        tokenizer:    ImageTokenizer 实例 (一般先单独训练到 reconstruction 质量 OK)
        gpt_*:        GPT3 构造参数, vocab_size = codebook_size + 1 (为 [BOS])
    """

    # 约定: 最后一个 token id 作为 [BOS]
    def __init__(
        self,
        tokenizer: ImageTokenizer,
        gpt_d_model: int = 384,
        gpt_n_heads: int = 6,
        gpt_num_layers: int = 12,
        gpt_dropout: float = 0.0,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        # 冻结 tokenizer, 让 GPT 专心学序列建模
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.codebook_size = tokenizer.quantizer.num_embeddings
        self.num_tokens = tokenizer.num_tokens
        self.grid_size = tokenizer.grid_size

        # [BOS] id = codebook_size; GPT 词表需要 +1
        self.bos_id = self.codebook_size
        gpt_vocab = self.codebook_size + 1
        # 序列长度: 1 (BOS) + num_tokens
        gpt_max_len = 1 + self.num_tokens

        self.gpt = GPT3(
            vocab_size=gpt_vocab,
            d_model=gpt_d_model,
            n_heads=gpt_n_heads,
            num_layers=gpt_num_layers,
            max_len=gpt_max_len,
            dropout=gpt_dropout,
            use_rope=True,
        )

    def images_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        图像 → raster scan token 序列 [B, N], 不含 BOS。
        """
        with torch.no_grad():
            indices, _ = self.tokenizer.encode_to_indices(x)
        return indices.view(indices.size(0), -1)   # [B, H'·W']

    def tokens_to_images(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        raster token 序列 (不含 BOS) → 图像
        """
        B = tokens.size(0)
        grid = tokens.view(B, self.grid_size, self.grid_size)
        return self.tokenizer.decode_from_indices(grid)

    def forward(
        self, images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向:
            input = [BOS, t_1, ..., t_{N-1}]
            labels = [t_1, ..., t_N]          (shift-by-one)

        Returns:
            logits: [B, N, vocab]
            labels: [B, N]
        """
        B = images.size(0)
        tokens = self.images_to_tokens(images)     # [B, N]

        bos = torch.full(
            (B, 1), self.bos_id, device=tokens.device, dtype=torch.long
        )
        input_ids = torch.cat([bos, tokens[:, :-1]], dim=1)   # [B, N]
        labels = tokens                                        # [B, N]

        logits = self.gpt(input_ids)                           # [B, N, vocab]
        return {"logits": logits, "labels": labels}

    @torch.inference_mode()
    def sample(
        self, batch_size: int, temperature: float = 1.0, top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        从 [BOS] 起自回归采样完整 token 序列, 再解码回图像。

        Returns:
            [B, 3, H, W] 生成的图像
        """
        if device is None:
            device = next(self.parameters()).device

        bos = torch.full(
            (batch_size, 1), self.bos_id, device=device, dtype=torch.long
        )
        # 借 GPT3.generate 直接生成 num_tokens 步
        seq = self.gpt.generate(bos, max_new_tokens=self.num_tokens,
                                temperature=temperature, top_k=top_k)
        # 去掉 BOS
        tokens = seq[:, 1:]
        # VQ codebook 的合法 id 区间是 [0, codebook_size-1], 若采到 BOS 则强制改为 0
        tokens = tokens.clamp(max=self.codebook_size - 1)
        return self.tokens_to_images(tokens)

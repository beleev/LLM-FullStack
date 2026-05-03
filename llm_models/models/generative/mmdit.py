"""
MM-DiT (Multimodal Diffusion Transformer) — SD3 / FLUX 核心

论文出处:
    "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    (Esser et al., Stability AI, 2024, Stable Diffusion 3 技术报告)
    "FLUX.1" (Black Forest Labs, 2024) 同门路线

与原 DiT 的核心差异:
    - 原 DiT: 文本/类别通过 **adaLN 的全局调制** 注入 (见 layers/adaln.py),
             文本没有独立 token, 细粒度对齐能力弱
    - MM-DiT: 文本 tokens 与图像 patch tokens **拼到同一个序列**,
             经 **同一层 attention** 做交互 (类似 LLM 里拼 prompt + image);
             但 Q/K/V 投影与 FFN 参数是 **每模态独立** (两套权重,
             只在 attention 阶段聚在一起)
             => "dual-stream" 架构

为什么要 dual-stream?
    两模态分布差异大 (文本离散 token vs 图像连续 latent), 共享 QKV/FFN 参数会
    让模型顾此失彼; 但完全分开又失去跨模态对齐能力。
    "参数分, 注意力合" 在 SD3/FLUX 实证下收益最佳。

本文件实现教学版 MMDiT Block:
    两个独立的 QKV / FFN 投影 → 拼接 → 共享缩放点积 attention → 切回各自流 → 各自 FFN。
    条件 (timestep + text pooler) 仍走 adaLN-Zero, 调制 image 流 (文本流可选).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.diffusion.adaln import FinalLayer, TimestepEmbedding, modulate
from llm_models.layers.core.attention import ScaledDotProductAttention
from llm_models.layers.core.feedforward import GeLUFeedForward


class MMDiTBlock(nn.Module):
    """
    MM-DiT dual-stream block

    数据流:
        img_tokens, txt_tokens 经 adaLN 调制 →
        各自 qkv_img / qkv_txt 投影 → 拼接 → 共享 SDPA →
        切回各自流 → Linear out → gate 残差 →
        各自 FFN + adaLN 调制 → gate 残差

    参数:
        d_model: 两模态共用的隐藏维度 (SD3 原版也设相同)
        c_dim:   条件向量维度 (timestep + text pooler)
        n_heads: attention 头数 (共享)
        d_ff:    FFN 隐藏维度
    """

    def __init__(
        self,
        d_model: int,
        c_dim: int,
        n_heads: int,
        d_ff: int,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须被 n_heads ({n_heads}) 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # === image stream ===
        self.img_norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.img_proj = nn.Linear(d_model, d_model, bias=True)
        self.img_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.img_ffn = GeLUFeedForward(d_model, d_ff)
        self.img_mod = nn.Linear(c_dim, 6 * d_model, bias=True)

        # === text stream ===
        self.txt_norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.txt_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.txt_proj = nn.Linear(d_model, d_model, bias=True)
        self.txt_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.txt_ffn = GeLUFeedForward(d_model, d_ff)
        self.txt_mod = nn.Linear(c_dim, 6 * d_model, bias=True)

        # adaLN-Zero: 调制层初始化为 0, 使 block 起点为恒等映射
        for m in (self.img_mod, self.txt_mod):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        self.attn = ScaledDotProductAttention()

    def _qkv_heads(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [B, T, 3D] → 3 × [B, H, T, Dh]
        """
        B, T, _ = qkv.shape
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Dh]
        return qkv[0], qkv[1], qkv[2]

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img: [B, N_img, D]
            txt: [B, N_txt, D]
            c:   [B, c_dim]  条件
        Returns:
            (img_out, txt_out) 同形
        """
        # --- 解包 6 段调制 ---
        img_shift_a, img_scale_a, img_gate_a, img_shift_f, img_scale_f, img_gate_f = \
            self.img_mod(c).chunk(6, dim=-1)
        txt_shift_a, txt_scale_a, txt_gate_a, txt_shift_f, txt_scale_f, txt_gate_f = \
            self.txt_mod(c).chunk(6, dim=-1)

        # --- 1) 各自 adaLN 后投 QKV ---
        img_mod = modulate(self.img_norm1(img), img_shift_a, img_scale_a)
        txt_mod = modulate(self.txt_norm1(txt), txt_shift_a, txt_scale_a)

        q_img, k_img, v_img = self._qkv_heads(self.img_qkv(img_mod))   # [B, H, N_img, Dh]
        q_txt, k_txt, v_txt = self._qkv_heads(self.txt_qkv(txt_mod))   # [B, H, N_txt, Dh]

        # --- 2) 两流在序列维度拼接, 做共享 attention ---
        Q = torch.cat([q_img, q_txt], dim=2)      # [B, H, N_img+N_txt, Dh]
        K = torch.cat([k_img, k_txt], dim=2)
        V = torch.cat([v_img, v_txt], dim=2)
        attn_out, _ = self.attn(Q, K, V)          # [B, H, N, Dh]

        # 切回两流
        B, H, N, Dh = attn_out.shape
        N_img = img.size(1)
        attn_img = attn_out[:, :, :N_img].transpose(1, 2).reshape(B, N_img, H * Dh)
        attn_txt = attn_out[:, :, N_img:].transpose(1, 2).reshape(B, N - N_img, H * Dh)

        # --- 3) 输出投影 + gate 残差 ---
        img = img + img_gate_a.unsqueeze(1) * self.img_proj(attn_img)
        txt = txt + txt_gate_a.unsqueeze(1) * self.txt_proj(attn_txt)

        # --- 4) FFN 子层: 各流独立 adaLN + FFN + gate 残差 ---
        img_ffn = self.img_ffn(modulate(self.img_norm2(img), img_shift_f, img_scale_f))
        txt_ffn = self.txt_ffn(modulate(self.txt_norm2(txt), txt_shift_f, txt_scale_f))
        img = img + img_gate_f.unsqueeze(1) * img_ffn
        txt = txt + txt_gate_f.unsqueeze(1) * txt_ffn

        return img, txt


class MMDiT(nn.Module):
    """
    MM-DiT (SD3 / FLUX 风格) — 图像 + 文本联合扩散 Transformer

    输入:
        x (含噪 latent): [B, C, H, W]
        t (timestep):    [B]
        text_embeds:     [B, T_txt, d_model]  已被文本 encoder 处理过的 token 序列
                         (教学场景可外部喂随机张量或真实 text encoder 输出)
        text_pooled:     [B, c_dim] 句子级文本向量, 与 timestep 相加做全局调制

    架构:
        image stream: PatchifyConv + learnable 2D pos
        text stream:  直接用 text_embeds 作为序列
        dual-stream blocks: 见 MMDiTBlock
        FinalLayer: 在 image 流上做 adaLN + Linear, unpatchify 回像素
        (文本流不需要输出 prediction, 因为我们只生成图像)

    Args:
        latent_channels, image_size, patch_size, d_model, n_heads, num_layers: 与 DiT 同
        text_seq_len: 预期文本 token 数 (决定 text 流位置嵌入)
        text_dim:     外部文本 encoder 的输出维度 (会被线性投到 d_model)
        c_dim:        条件向量维度
    """

    def __init__(
        self,
        latent_channels: int = 4,
        image_size: int = 32,
        patch_size: int = 2,
        d_model: int = 384,
        n_heads: int = 6,
        num_layers: int = 12,
        text_seq_len: int = 77,
        text_dim: int = 768,
        c_dim: Optional[int] = None,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} 必须能被 patch_size {patch_size} 整除")

        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        if c_dim is None:
            c_dim = d_model

        # image stream
        self.patchify = nn.Conv2d(latent_channels, d_model, patch_size, patch_size)
        self.img_pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.img_pos, std=0.02)

        # text stream: 先投影到 d_model, 加独立位置嵌入
        self.text_proj = nn.Linear(text_dim, d_model, bias=False)
        self.text_pos = nn.Parameter(torch.zeros(1, text_seq_len, d_model))
        nn.init.trunc_normal_(self.text_pos, std=0.02)
        self.text_seq_len = text_seq_len

        # 条件: timestep + (可选) 文本 pooler
        self.t_embed = TimestepEmbedding(c_dim)
        self.text_pool_proj = nn.Linear(text_dim, c_dim, bias=False)

        d_ff = 4 * d_model
        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(d_model=d_model, c_dim=c_dim, n_heads=n_heads, d_ff=d_ff)
                for _ in range(num_layers)
            ]
        )

        # FinalLayer 只作用在 image 流
        self.final = FinalLayer(
            d_model=d_model, c_dim=c_dim,
            patch_out_dim=patch_size * patch_size * latent_channels,
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        C = self.latent_channels
        p = self.patch_size
        H_grid = self.grid_size
        x = x.view(B, H_grid, H_grid, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, C, H_grid * p, H_grid * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        text_pooled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:           [B, C, H, W] 含噪 latent
            t:           [B] timestep
            text_embeds: [B, T_txt, text_dim] 文本 token 序列 (外部 encoder 产出)
            text_pooled: [B, text_dim] 句子级文本向量; 与 t 相加做全局调制
        Returns:
            [B, C, H, W] 预测的噪声或 velocity
        """
        B = x.size(0)

        # image stream
        img = self.patchify(x).flatten(2).transpose(1, 2)                # [B, N_img, D]
        img = img + self.img_pos

        # text stream: 投影 + 截断/补齐到预设长度
        T_txt = text_embeds.size(1)
        if T_txt > self.text_seq_len:
            text_embeds = text_embeds[:, : self.text_seq_len]
            T_txt = self.text_seq_len
        txt = self.text_proj(text_embeds) + self.text_pos[:, :T_txt]     # [B, T_txt, D]

        # 条件
        c = self.t_embed(t)
        if text_pooled is not None:
            c = c + self.text_pool_proj(text_pooled)

        # dual-stream block 栈
        for block in self.blocks:
            img, txt = block(img, txt, c)

        out = self.final(img, c)                                          # [B, N_img, p²·C]
        return self.unpatchify(out)

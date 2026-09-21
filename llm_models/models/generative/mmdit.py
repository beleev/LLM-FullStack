"""
MM-DiT — SD3 / FLUX 的双流扩散 Transformer (Esser et al., 2024)

解决的问题: DiT 只能经 adaLN 注入一个 **全局** 条件向量, 文本被压成一个向量, 细粒度图文对齐弱。
MM-DiT 让文本 token 与图像 patch token 拼成一个序列做 **联合注意力**, 但两种模态分布差异大,
所以 QKV / FFN / adaLN 参数每模态各一套 —— "参数分, 注意力合":

    q,k,v_img = W_img(adaLN(img));  q,k,v_txt = W_txt(adaLN(txt))
    Attn(cat[q_img, q_txt], cat[k_img, k_txt], cat[v_img, v_txt]) → 再切回两条流各走各的 FFN

全局条件 c = TimestepEmbedding(t) + Linear(text_pooled) 仍走 adaLN-Zero。
t 的量纲是 [0, 1000) (Flow Matching 的 t∈[0,1] 由 scheduler ×1000, 见 training/diffusion.py)。
读代码时盯住: MMDiTBlock.forward 里 torch.cat(dim=2) 与之后的切分。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.diffusion.adaln import FinalLayer, TimestepEmbedding, modulate
from llm_models.layers.core.attention import ScaledDotProductAttention
from llm_models.layers.core.feedforward import GeLUFeedForward


class MMDiTBlock(nn.Module):
    """
    双流 block: 各自 adaLN + QKV 投影 → 序列维拼接做一次注意力 → 切回 → 各自 out_proj / FFN, 均带 gate 残差。

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
    图像流: patchify + 可学习位置; 文本流: 外部 encoder 的 token 序列经 Linear 投到 d_model。
    只有图像流接 FinalLayer (只生成图像); 最后一层的文本流输出被丢弃。

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
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()      # [B, C, H/p, p, W/p, p]
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
            t:           [B] timestep, [0, 1000) 量纲 (AddNoiseResult.t_norm)
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

        # 全局条件 [B, c_dim]
        c = self.t_embed(t)
        if text_pooled is not None:
            c = c + self.text_pool_proj(text_pooled)

        # dual-stream block 栈
        for block in self.blocks:
            img, txt = block(img, txt, c)

        out = self.final(img, c)                                          # [B, N_img, p²·C]
        return self.unpatchify(out)

#!/usr/bin/env python
"""
VAR 两阶段训练 (都在一个固定的合成 batch 上, "loss 下降" = "能背下这个 batch", 不代表泛化)

阶段 1: 训练多尺度 VQ tokenizer (重建 MSE + vq_loss) —— 码本是学出来的, 不是随机冻结的
阶段 2: 冻结 tokenizer, 训练 block-causal Transformer 做 next-scale prediction (CE)
"""

import math

import torch
import torch.nn.functional as F

from llm_models.models.generative.var import ImageTokenizer, VARModel
from llm_models.training.config import TrainingConfig
from llm_models.training.data import VARImageDataGenerator
from llm_models.training.loss import VARLoss
from llm_models.training.trainer import Trainer

K = 64  # 码本大小 = Transformer 词表大小


def train_tokenizer(tokenizer: ImageTokenizer, images: torch.Tensor, steps: int = 150):
    opt = torch.optim.Adam(tokenizer.parameters(), lr=3e-3)
    history = []
    for step in range(steps):
        out = tokenizer(images)
        recon_loss = F.mse_loss(out["recon"], images)
        loss = recon_loss + out["vq_loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(recon_loss.item())
        if step % 30 == 0 or step == steps - 1:
            print(f"  [tokenizer] step {step:3d} | recon {recon_loss.item():.4f} "
                  f"| vq {out['vq_loss'].item():.4f} | 码本 perplexity {out['perplexity'].item():.1f}")
    return history


def main():
    cfg = TrainingConfig(
        learning_rate=3e-3, batch_size=8, num_steps=100,
        warmup_steps=5, log_interval=20, seed=42,
    )
    torch.manual_seed(cfg.seed)

    data_gen = VARImageDataGenerator(batch_size=cfg.batch_size, image_size=16)
    images = data_gen.generate_batch()["images"]                 # 固定 batch [8, 3, 16, 16]

    # ---- 阶段 1: tokenizer ----
    tokenizer = ImageTokenizer(image_size=16, codebook_size=K, latent_dim=16,
                               base_channels=16, levels=2, scales=(1, 2, 4))
    codebook_init = tokenizer.quantizer.vq.codebook.weight.detach().clone()
    recon = train_tokenizer(tokenizer, images)
    moved = (tokenizer.quantizer.vq.codebook.weight - codebook_init).abs().max().item()
    print(f"tokenizer recon: {recon[0]:.4f} → {recon[-1]:.4f} | 码本最大位移 {moved:.4f}")
    assert recon[-1] < 0.5 * recon[0], "tokenizer 重建 loss 应明显下降"
    assert moved > 1e-3, "码本没有被训练 (straight-through / vq_loss 断了?)"

    # ---- 阶段 2: next-scale Transformer ----
    model = VARModel(tokenizer.eval(), d_model=96, n_heads=4, num_layers=2)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAR Transformer 可训练参数: {n_train:,} | 序列长度 L = {model.num_tokens}")

    metrics = Trainer(model, cfg, data_gen, VARLoss()).train()
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"AR loss: {first:.4f} → {last:.4f}  (ln K = {math.log(K):.4f})")
    assert abs(first - math.log(K)) < 0.5, "初始 CE 应 ≈ ln K (均匀猜测)"
    assert last < 0.5 * first, "AR loss 应明显下降"

    # 采样: 3 次前向出 21 个 token, 全部是合法码字 (词表里根本没有 BOS)
    tokens = model.eval().sample_tokens(batch_size=4, top_k=10)
    assert all(0 <= t.min() and t.max() < K for t in tokens)
    print("采样 token 形状:", [tuple(t.shape) for t in tokens])


if __name__ == "__main__":
    main()

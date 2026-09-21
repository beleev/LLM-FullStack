#!/usr/bin/env python
"""
Qwen2-VL 训练: loss 只算文本位置; 并构造 "不看图就做不对" 的固定 batch

2 条样本的文本输入**完全相同**, 图不同, labels 不同。只看文本的模型 loss 下界是 ln 2 ≈ 0.693;
降到 ln 2 以下 ⇒ 文本位置确实通过因果注意力读到了视觉前缀。(随机数据, 这是记忆而非看图说话。)
"""

import math

import torch

from llm_models.models.multimodal.qwen2_vl import Qwen2VLModel
from llm_models.training import Trainer, TrainingConfig, StandardLMLoss, VisionLanguageDataGenerator


class SameTextDifferentImage(VisionLanguageDataGenerator):
    def _sample(self):
        batch = super()._sample()
        batch["input_ids"] = batch["input_ids"][:1].repeat(self.batch_size, 1)
        return batch


def main():
    config = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=16,
        num_steps=100, warmup_steps=5, log_interval=20, seed=42,
    )
    torch.manual_seed(config.seed)

    V, IMG, N_LAT = 500, 56, 8                                         # 56/14 = 4 ⇒ 16 patch → Resampler → 8 token
    model = Qwen2VLModel(
        vocab_size=V, text_d_model=128, text_n_heads=4, text_num_kv_heads=2, text_num_layers=2, max_len=64,
        vision_image_size=IMG, vision_patch_size=14, vision_d_model=128, vision_n_heads=4, vision_num_layers=2,
        vision_num_latents=N_LAT, vision_num_latent_layers=1, dropout=0.0,
    )
    print(f"Qwen2-VL Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = SameTextDifferentImage(
        vocab_size=V, batch_size=config.batch_size, seq_len=config.seq_len,
        image_size=IMG, num_vision_tokens=N_LAT,
    )
    labels = data_gen.generate_batch()["labels"]                       # [B, N_LAT + T]
    assert (labels[:, :N_LAT] == -100).all() and (labels[:, N_LAT:] != -100).all()   # 视觉位置不计 loss

    metrics = Trainer(model, config, data_gen, StandardLMLoss()).train()
    first, last = metrics[0]["total_loss"], metrics[-1]["total_loss"]
    print(f"初始 loss {first:.3f} (ln V = {math.log(V):.3f}) -> 终态 {last:.3f} (不看图的下界 ln 2 = {math.log(2):.3f})")
    assert abs(first - math.log(V)) < 0.5, "初始 loss 应 ≈ ln V"
    assert last < 0.5 * math.log(2), "loss 低于 ln 2 才说明文本用上了视觉前缀"


if __name__ == "__main__":
    main()

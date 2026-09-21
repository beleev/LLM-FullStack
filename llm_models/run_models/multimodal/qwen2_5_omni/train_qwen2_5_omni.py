#!/usr/bin/env python
"""
Qwen2.5-Omni 训练: total = text_loss + 0.5 · audio_loss, 两个 loss 的初值分别 ≈ ln V_text / ln V_audio

固定 batch 的 2 条样本 codec 输入**完全相同**、codec 标签不同 ⇒ Talker 只看自己的输入时 audio_loss 下界为 ln 2;
降到 ln 2 以下 ⇒ Talker 确实通过 cross-attention 读了 Thinker 隐状态。(随机数据, 这是记忆。)
"""

import math

import torch

from llm_models.models.multimodal.qwen2_5_omni import Qwen2_5_OmniModel
from llm_models.run_models.multimodal.qwen2_5_omni._config import IMAGE, N_LAT, SPEC, TINY, V_AUDIO, V_TEXT, VIDEO
from llm_models.training import Trainer, TrainingConfig, OmniLoss, OmniDataGenerator


class SameCodecInput(OmniDataGenerator):
    def _sample(self):
        batch = super()._sample()
        batch["audio_input_ids"] = batch["audio_input_ids"][:1].repeat(self.batch_size, 1)
        return batch


def main():
    config = TrainingConfig(
        learning_rate=3e-3, batch_size=2, seq_len=8,
        num_steps=250, warmup_steps=5, audio_loss_weight=0.5, log_interval=50, seed=42,
    )
    torch.manual_seed(config.seed)

    model = Qwen2_5_OmniModel(**TINY)
    print(f"Omni Tiny | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = SameCodecInput(
        vocab_size=V_TEXT, audio_vocab_size=V_AUDIO, batch_size=config.batch_size,
        seq_len=config.seq_len, audio_seq_len=8,
        image_size=IMAGE, audio_spec_size=SPEC, video_size=VIDEO,
        num_vision_tokens=N_LAT, num_audio_tokens=N_LAT, num_video_tokens=N_LAT,
    )
    metrics = Trainer(model, config, data_gen, OmniLoss(audio_loss_weight=config.audio_loss_weight)).train()

    m0, m1 = metrics[0], metrics[-1]
    print(f"text_loss {m0['text_loss']:.3f} (ln {V_TEXT} = {math.log(V_TEXT):.3f}) -> {m1['text_loss']:.3f} | "
          f"audio_loss {m0['audio_loss']:.3f} (ln {V_AUDIO} = {math.log(V_AUDIO):.3f}) -> {m1['audio_loss']:.3f} "
          f"(不读 Thinker 的下界 ln 2 = {math.log(2):.3f})")
    assert abs(m0["text_loss"] - math.log(V_TEXT)) < 0.5 and abs(m0["audio_loss"] - math.log(V_AUDIO)) < 0.5
    assert abs(m0["total_loss"] - (m0["text_loss"] + 0.5 * m0["audio_loss"])) < 1e-4
    assert m1["text_loss"] < 0.5 * m0["text_loss"]
    assert m1["audio_loss"] < 0.5 * math.log(2), "audio_loss 低于 ln 2 才说明 Talker 用上了 Thinker 隐状态"


if __name__ == "__main__":
    main()

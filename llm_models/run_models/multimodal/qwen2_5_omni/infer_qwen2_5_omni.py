#!/usr/bin/env python
"""
Qwen2.5-Omni 推理: 验证 Thinker-Talker 之间的信息流向

盯住: 信息只能 模态输入 → Thinker → (hidden) → Talker 单向流动。
"""

import torch

from llm_models.models.multimodal.qwen2_5_omni import Qwen2_5_OmniModel
from llm_models.run_models.multimodal.qwen2_5_omni._config import IMAGE, N_LAT, SPEC, TINY, V_AUDIO, V_TEXT, VIDEO


def main():
    torch.manual_seed(42)
    model = Qwen2_5_OmniModel(**TINY).eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"Omni Tiny | 总参数 {n(model):,} | Thinker {n(model.thinker):,} | Talker {n(model.talker):,}")
    assert model.thinker_to_talker is not None                         # Talker(32 维) 比 Thinker(64 维) 窄 ⇒ 需要桥接投影

    B, T, T_A = 2, 10, 12
    ids = torch.randint(1, V_TEXT, (B, T))
    codec = torch.randint(1, V_AUDIO, (B, T_A))
    img, spec, vid = torch.randn(B, 3, IMAGE, IMAGE), torch.randn(B, 1, *SPEC), torch.randn(B, 3, *VIDEO)

    with torch.inference_mode():
        out = model(ids, images=img, audio_spectrograms=spec, videos=vid, audio_input_ids=codec)
        text_logits, audio_logits = out["text_logits"], out["audio_logits"]
        assert text_logits.shape == (B, 3 * N_LAT + T, V_TEXT)         # [vision 4; video 4; audio 4; text 10]
        assert audio_logits.shape == (B, T_A, V_AUDIO)

        # 1) 少给一个模态 ⇒ 前缀变短; 不给 codec token ⇒ 不跑 Talker
        o2 = model(ids, images=img)
        assert o2["text_logits"].shape == (B, N_LAT + T, V_TEXT) and o2["audio_logits"] is None

        # 2) 换图 → Talker 输出改变 (图 → Thinker hidden → cross-attn → Talker)
        o3 = model(ids, images=torch.randn_like(img), audio_spectrograms=spec, videos=vid, audio_input_ids=codec)
        d_img = (o3["audio_logits"] - audio_logits).abs().max().item()
        # 3) 换 codec token → Thinker 的文本输出不变 (Talker 不回流)
        o4 = model(ids, images=img, audio_spectrograms=spec, videos=vid,
                   audio_input_ids=torch.randint(1, V_AUDIO, (B, T_A)))
        d_back = (o4["text_logits"] - text_logits).abs().max().item()
        # 4) Talker 自身因果: 改最后一个 codec token, 前面位置不变
        codec2 = codec.clone(); codec2[:, -1] = codec2[:, -1] % (V_AUDIO - 1) + 1
        o5 = model(ids, images=img, audio_spectrograms=spec, videos=vid, audio_input_ids=codec2)
        d_causal = (o5["audio_logits"][:, :-1] - audio_logits[:, :-1]).abs().max().item()
    print(f"换图→语音 logits 变化 {d_img:.1e} | 换 codec→文本 logits 变化 {d_back:.1e} | 改未来 codec→过去变化 {d_causal:.1e}")
    assert d_img > 1e-6 and d_back == 0 and d_causal < 1e-6


if __name__ == "__main__":
    main()

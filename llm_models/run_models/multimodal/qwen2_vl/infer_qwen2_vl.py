#!/usr/bin/env python
"""
Qwen2-VL 推理: 验证 prefix 拼接的因果结构, 以及 M-RoPE 的两个性质
"""

import torch

from llm_models.models.multimodal.qwen2_vl import Qwen2VLDecoder, Qwen2VLModel, build_mrope_position_ids


def main():
    torch.manual_seed(42)
    V, B, T, N_LAT = 1000, 2, 12, 8
    cfg = dict(
        vocab_size=V, text_d_model=96, text_n_heads=4, text_num_kv_heads=2, text_num_layers=2, max_len=128,
        vision_image_size=56, vision_patch_size=14, vision_d_model=96, vision_n_heads=4, vision_num_layers=2,
        vision_num_latent_layers=1, dropout=0.0,
    )
    ids = torch.randint(1, V, (B, T))
    img = torch.randn(B, 3, 56, 56)                                    # 56/14 = 4 ⇒ 4×4 = 16 个 patch

    # ---------- A. prefix 拼接 (Resampler: 16 patch → 8 latent) ----------
    model = Qwen2VLModel(**cfg, vision_num_latents=N_LAT).eval()
    print(f"Qwen2-VL Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    with torch.inference_mode():
        logits = model(ids, images=img)
        assert logits.shape == (B, N_LAT + T, V)                       # 视觉 token 在前
        assert model(ids).shape == (B, T, V)                           # 不给图 = 纯文本 LLM

        # 1) 换图 → 文本位置 logits 变 (文本看得到视觉前缀)
        d_img = (model(ids, images=torch.randn_like(img))[:, N_LAT:] - logits[:, N_LAT:]).abs().max().item()
        # 2) 换文本 → 视觉位置输出不变 (因果: 前缀看不到后面的文本)
        d_txt = (model(torch.randint(1, V, (B, T)), images=img)[:, :N_LAT] - logits[:, :N_LAT]).abs().max().item()
        # 3) 文本 padding mask 自动补上全 1 的视觉段: 改 pad 处 token 不影响其余位置
        am = torch.ones(B, T, dtype=torch.long); am[:, -3:] = 0
        ids2 = ids.clone(); ids2[:, -3:] = 7
        d_pad = (model(ids, images=img, text_attention_mask=am)[:, :-3]
                 - model(ids2, images=img, text_attention_mask=am)[:, :-3]).abs().max().item()
    print(f"A. 换图→文本 logits 变化 {d_img:.4f} | 换文本→视觉位置变化 {d_txt:.1e} | 改 pad token→其余位置变化 {d_pad:.1e}")
    assert d_img > 1e-4 and d_txt < 1e-5 and d_pad < 1e-5

    # ---------- B. M-RoPE (关闭 Resampler, 视觉 token = 4×4 网格) ----------
    pos = build_mrope_position_ids(B, (4, 4), T)                       # [3, B, 16 + T]
    assert pos.shape == (3, B, 16 + T)
    assert pos[:, 0, 5].tolist() == [0, 1, 1]                          # 第 5 个 patch = 第 1 行第 1 列, 时间 0
    assert pos[:, 0, 16].tolist() == [4, 4, 4]                         # 第一个文本 token: 三轴相同, 接在 max(H,W) 之后

    mrope = Qwen2VLModel(**cfg, vision_num_latents=0, use_mrope=True).eval()
    with torch.inference_mode():
        out = mrope(ids, images=img, position_ids=pos)
        assert out.shape == (B, 16 + T, V)
        # 4) 2-D 感知: 把视觉 patch 的 (行, 列) 坐标对调 (= 告诉模型图被转置了) → 输出改变
        pos_t = pos.clone(); pos_t[1, :, :16], pos_t[2, :, :16] = pos[2, :, :16], pos[1, :, :16]
        d_hw = (mrope(ids, images=img, position_ids=pos_t) - out).abs().max().item()

        # 5) 纯文本时三轴相同 ⇒ M-RoPE 是 1-D 相对位置编码: 整体平移位置, logits 不变
        dec: Qwen2VLDecoder = mrope.text_decoder
        p = torch.arange(T).expand(3, B, -1)
        base = dec(input_ids=ids, position_ids=p)
        d_shift = (dec(input_ids=ids, position_ids=p + 9) - base).abs().max().item()

        # 与普通 1-D RoPE 逐数值比较 (同一套权重)
        dec1d = Qwen2VLDecoder(V, 96, 4, 2, 128, num_kv_heads=2, dropout=0.0).eval()
        dec1d.load_state_dict(dec.state_dict())
        d_1d = (dec1d(input_ids=ids) - base).abs().max().item()
    print(f"B. 对调行列坐标→输出变化 {d_hw:.1e} | 文本位置整体 +9 → logits 变化 {d_shift:.1e} | 与 1-D RoPE 的数值差 {d_1d:.1e}")
    # H/W 段分到的是较低频率 (同一条频率轴的后两段), 4×4 网格上角度差小, 未训练时影响 ~1e-4 量级但必须非零
    assert d_hw > 1e-5 and d_shift < 1e-5
    assert d_1d < 1e-6, "三轴位置相同时 M-RoPE 必须与 1-D RoPE 逐位相等 (三轴共用同一条频率轴)"


if __name__ == "__main__":
    main()

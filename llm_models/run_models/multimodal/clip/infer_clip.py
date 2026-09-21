#!/usr/bin/env python
"""
CLIP 推理: 验证特征归一化、EOS 池化, 并演示零样本分类的计算形式
"""

import torch

from llm_models.models.multimodal.clip import CLIPModel


def main():
    torch.manual_seed(42)
    V, B, T, EOS = 1000, 4, 16, 999
    model = CLIPModel(
        embed_dim=256, vocab_size=V,
        text_d_model=128, text_n_heads=4, text_num_layers=2, text_max_len=32,
        image_size=64, patch_size=16,
        vision_d_model=128, vision_n_heads=4, vision_num_layers=2,
    ).eval()
    print(f"CLIP Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    images = torch.randn(B, 3, 64, 64)
    text = torch.randint(1, EOS, (B, T)); text[:, 9] = EOS             # 第 9 位是 EOS, 之后视为 padding

    with torch.inference_mode():
        out = model(images, text, eos_token_id=EOS)
        img, txt = out["image_features"], out["text_features"]
        assert img.shape == txt.shape == (B, 256)
        # 1) L2 归一化 ⇒ 点积就是余弦相似度
        assert torch.allclose(img.norm(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(txt.norm(dim=-1), torch.ones(B), atol=1e-5)
        # 2) EOS 池化 + 因果 mask: EOS 之后放什么都不影响句向量; EOS 之前改一个 token 则会变
        t2 = text.clone(); t2[:, 10:] = torch.randint(1, EOS, (B, T - 10))
        t3 = text.clone(); t3[:, 0] = t3[:, 0] % (EOS - 1) + 1
        d_after = (model.encode_text(t2, EOS) - txt).abs().max().item()
        d_before = (model.encode_text(t3, EOS) - txt).abs().max().item()
        print(f"改 EOS 之后的 token → 句向量变化 {d_after:.1e}; 改 EOS 之前的 → {d_before:.4f}")
        assert d_after < 1e-6 and d_before > 1e-4

        # 3) 零样本分类的形式: 每张图对 B 个 "类别提示语" 做 softmax
        probs = (out["logit_scale"] * img @ txt.t()).softmax(dim=-1)   # [B_img, B_txt]
        assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    print(f"logit_scale = {out['logit_scale'].item():.2f} (= 1/0.07); 未训练时图 0 对各文本的概率: "
          f"{[round(p, 3) for p in probs[0].tolist()]} (均匀 = 0.25)")


if __name__ == "__main__":
    main()

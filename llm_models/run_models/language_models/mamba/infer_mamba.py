#!/usr/bin/env python
"""
Mamba 推理: 验证 "递推解码 == 整段前向", 以及长生成数值稳定

盯住: cache 里的状态形状不随已读长度变化 (Transformer 的 KV cache 会变长)。
"""

import math
import time

import torch

from llm_models.layers.sparse.ssm import SelectiveSSM
from llm_models.models.language_models.mamba import Mamba
from llm_models.utils.generation import KVCache


def main():
    torch.manual_seed(42)
    V, L = 1000, 4
    model = Mamba(vocab_size=V, d_model=128, num_layers=L, d_state=16, d_conv=4).eval()
    print(f"Mamba Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    idx = torch.randint(0, V, (2, 64))

    with torch.inference_mode():
        # 1) 整段前向 vs prefill 8 个 token + 逐 token 递推: logits 必须一致
        full = model(idx)                                              # [2, 64, V]
        cache = KVCache(L)
        outs = [model(idx[:, :8], cache=cache)]
        outs += [model(idx[:, t:t + 1], cache=cache) for t in range(8, 64)]
        diff = (full - torch.cat(outs, dim=1)).abs().max().item()
        state = {k: tuple(v.shape) for k, v in cache.layers[0].items()}
    print(f"递推 vs 整段 logits 最大差: {diff:.2e} | 读完 {cache.pos} 个 token 后每层状态: {state}")
    assert diff < 1e-4
    assert state == {"conv": (2, 256, 3), "h": (2, 256, 16)}           # 与 T=64 无关

    # 2) 因果性: 改最后一个 token, 前面位置的 logits 不变 (没有 mask, 靠递推方向天然因果)
    idx2 = idx.clone(); idx2[:, -1] = (idx2[:, -1] + 1) % V
    with torch.inference_mode():
        assert torch.allclose(model(idx2)[:, :-1], full[:, :-1], atol=1e-6)

    # 3) 长生成: 每步 logits 有限 (曾经的 nan_to_num 兜底已删, 稳定性由 A<0, Δ>0 保证)
    seen = []
    hook = model.lm_head.register_forward_hook(lambda m, i, o: seen.append(torch.isfinite(o).all().item()))
    t0 = time.time(); g_cache = model.generate(idx[:, :5], 300, temperature=0); t_cache = time.time() - t0
    t0 = time.time(); g_naive = model.generate(idx[:, :5], 300, temperature=0, use_cache=False); t_naive = time.time() - t0
    hook.remove()
    assert all(seen) and len(seen) == 600, "生成过程中出现非有限 logits"
    assert torch.equal(g_cache, g_naive)
    model.generate(idx[:, :5], 50, temperature=0.8, top_k=20)          # 采样路径也不应报错
    print(f"生成 300 token: 递推 {t_cache:.2f}s vs 每步重算 {t_naive:.2f}s ({t_naive / t_cache:.0f}×), 600 步 logits 全部有限")

    # 4) 反例: 同一个递推 h = exp(Δ·a)·h + Δ·x, 只把 a 的符号翻过来
    def first_overflow(a: float, delta: float = 0.1, T: int = 400) -> int:
        h = torch.zeros(())
        for t in range(T):
            h = math.exp(delta * a) * h + delta * 1.0
            if not torch.isfinite(h):
                return t
        return -1

    t_neg, t_pos = first_overflow(-16.0), first_overflow(+16.0)
    print(f"稳定性: a=-16 ⇒ 400 步不溢出; a=+16 ⇒ 第 {t_pos} 步溢出为 inf (A=-exp(A_log) 就是为了排除这种情况)")
    assert t_neg == -1 and 0 < t_pos < 100
    ssm = SelectiveSSM(64, 16)
    with torch.no_grad():
        ssm.A_log.normal_(0, 5)                                        # 任意折腾 A_log, A 依然恒负
        assert torch.isfinite(ssm(torch.randn(1, 400, 64))).all()


if __name__ == "__main__":
    main()

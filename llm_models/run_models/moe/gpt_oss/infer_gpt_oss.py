#!/usr/bin/env python
"""
GPT-OSS 推理示例 — 交替 SWA/全注意力 + attention sink + MoE, 全部用 assert 写死:

    1. 激活参数 < 总参数; "top-k 后 softmax" (官方) 与 MixtralMoE 的 "softmax→top-k→重归一" 逐项相等
    2. sink: 每行分给真实 token 的注意力质量 < 1; sink logit → −inf 时退化回普通 softmax
    3. 感受野: 改一个 W 之外的 token, SWA 层 (第 0 层) 在当前位置的输出不变, 但整个模型的输出变了
       (奇数层是全注意力, 直接看得到它)
    4. KV cache: SWA 层长度 ≤ W, full 层长度 = 总长度; 有/无 cache 贪心生成逐 token 相同
模型随机初始化, 生成内容无意义。
"""

import copy

import torch
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.models.moe.gpt_oss import GPTOSSMini
from llm_models.utils.generation import KVCache, benchmark_kv_cache


def attention_mass(attn: GroupedQueryAttention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    不改 GQA 代码读出每行注意力质量: 让 V 恒为 1、w_o 为恒等, 则输出 = Σ_j attn_ij。返回 [T, H]。
    """
    probe = copy.deepcopy(attn)
    D, Dh = probe.d_model, probe.head_dim
    probe.w_v.weight.fill_(1.0 / D)                       # v 输入全 1 → V 的每个分量 = 1
    probe.w_o.weight.copy_(torch.eye(D))
    out = probe(q=x, k=x, v=torch.ones_like(x), mask=mask)   # [1, T, H*Dh], 每个 head 的 Dh 维都等于该行质量
    return out[0, :, ::Dh]                                # [T, H]


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    V, W, E, K, L = 1000, 8, 4, 2, 4
    model = GPTOSSMini(vocab_size=V, d_model=128, n_heads=4, num_kv_heads=2, num_layers=L,
                       num_experts=E, top_k=K, max_len=256, window_size=W).eval()

    # ---- 1) 参数量 + 路由等价 ----
    count = lambda m: sum(p.numel() for p in m.parameters())
    total = count(model)
    experts = sum(count(layer.moe.experts) for layer in model.layers)
    active = total - experts + experts * K // E
    assert active < total
    print(f"[1] 总参数 {total:,} | 每 token 激活 {active:,} ({active / total:.1%})")

    T = 40
    idx = torch.randint(1, V, (1, T))
    logits, routing = model(idx)
    assert logits.shape == (1, T, V) and len(routing) == L
    for info in routing:
        top_logits = info["router_logits"].gather(-1, info["selected_experts"])     # [N, K]
        assert torch.allclose(F.softmax(top_logits, dim=-1), info["routing_weights"], atol=1e-6), \
            "top-k 后 softmax 应与 softmax 后 top-k 重归一相等"
    print("    路由: softmax(top-k logits) == MixtralMoE 的 routing_weights (逐项相等)")

    # ---- 2) attention sink ----
    attn = model.layers[1].attn                           # 全注意力层
    x = torch.randn(1, T, model.d_model)
    causal = model.causal_mask[:, :T, :T]
    mass = attention_mass(attn, x, causal)                # [T, H]
    assert (mass < 1).all() and (mass > 0).all(), "有 sink 时真实 token 分到的质量必须 < 1"
    # 第 0 行只有 1 个 key: 质量 = e^s0 / (e^s0 + e^sink), 是全表最小的
    print(f"[2] 每行注意力质量 (对 head 取平均): 第 0 行 {mass[0].mean():.3f}, "
          f"第 {T - 1} 行 {mass[-1].mean():.3f}  (< 1, 差额进了 sink)")

    no_sink = copy.deepcopy(attn)
    no_sink.sink.fill_(float("-inf"))                     # sink 列概率 = 0
    plain = GroupedQueryAttention(attn.d_model, attn.num_heads, attn.num_kv_heads)
    plain.load_state_dict(attn.state_dict(), strict=False)   # 同一套 w_q/k/v/o, 只是没有 sink
    assert torch.allclose(attention_mass(no_sink, x, causal), torch.ones(T, attn.num_heads), atol=1e-6)
    assert torch.allclose(no_sink(x, mask=causal), plain(x, mask=causal), atol=1e-6)
    assert not torch.allclose(attn(x, mask=causal), plain(x, mask=causal), atol=1e-4)
    print("    sink = −inf: 行和回到 1, 输出与无 sink 的 GQA 相同; sink = 0 时输出不同")

    # ---- 3) 感受野: SWA 层看不到, 整个模型看得到 ----
    layer0_out = []
    hook = model.layers[0].register_forward_hook(lambda m, i, o: layer0_out.append(o[0]))
    changed = idx.clone()
    changed[:, 0] = idx[:, 0] % (V - 1) + 1               # 只改位置 0, 距最后位置 T-1 远超 W
    logits_changed, _ = model(changed)
    model(idx)                                            # 原输入再跑一遍, 让 hook 记下第 0 层输出
    hook.remove()
    d_layer = (layer0_out[0] - layer0_out[1]).abs().amax(dim=(0, 2))       # [T] SWA 层每个位置的变化
    d_model_out = (logits - logits_changed).abs().amax(dim=(0, 2))         # [T]
    assert d_layer[:W].min() > 1e-4, "窗口内 (位置 0..W-1) 应该受影响"
    assert d_layer[W:].max() < 1e-6, "SWA 层: 位置 ≥ W 的输出不应依赖位置 0"
    assert d_model_out[-1] > 1e-4, "full 层能直接看到位置 0, 最终 logits 必须变"
    print(f"[3] 改位置 0 → 第 0 层 (SWA) 位置 ≥ {W} 最大变化 {d_layer[W:].max():.1e}; "
          f"最终 logits 在位置 {T - 1} 变化 {d_model_out[-1]:.2e}")

    # ---- 4) 每层不同长度的 KV cache ----
    cache = KVCache(L)
    model(idx[:, :30], cache=cache)                       # prefill 30
    for t in range(30, T):
        step_logits, _ = model(idx[:, t:t + 1], cache=cache)   # 逐 token decode
    assert torch.allclose(step_logits[:, -1], logits[:, -1], atol=1e-4), "cache 路径 logits 与整段 forward 不一致"
    lens = [c["k"].size(2) for c in cache.layers]
    for i, n in enumerate(lens):
        assert n == (min(T, W) if model.is_swa(i) else T), f"layer {i} cache 长度 {n} 不对"
    saving = 1 - sum(lens) / (L * T)
    print(f"[4] 已读 {cache.pos} 个 token, 每层 cache 长度 {lens} (偶数层 SWA ≤ W={W}, 奇数层 full = {T}); "
          f"比全 full 的 {L * T} 省 {saving:.0%}")

    speedup = benchmark_kv_cache(model, idx[:, :8], max_new_tokens=100)    # 内部 assert 逐 token 相同
    print(f"    贪心生成 100 token: 有/无 cache 输出完全一致, 加速 {speedup:.1f}x")


if __name__ == "__main__":
    main()

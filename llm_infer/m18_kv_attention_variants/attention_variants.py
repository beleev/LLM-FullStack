"""
attention_variants.py — MHA / GQA / MQA / MLA: 同一个 attention, 四种 KV cache 大小

是什么: 只改 "cache 里存什么", 不改 attention 数学。MHA/GQA/MQA 是同一条代码路径
        (n_kv_heads 一个参数); MLA 只缓存低秩 latent c_kv (T, d_c) + 共享 RoPE key (T, d_rope)。
瓶颈:   显存。decode 的 batch×context 上限 = 显存 / 每 token KV 字节数。
关键数字: 每 token 字节 = 2·n_kv·d_head·n_layer·bytes; MLA = (d_c+d_rope)·n_layer·bytes。
        LLaMA-2-7B (MHA) 512 KiB → LLaMA-3-8B (GQA-8) 128 KiB → DeepSeek-V3 (MLA) 68.6 KiB。
盯住:   GQALayer.forward 里的 cache 形状 (T, n_kv, d_head); MLALayer.forward 里 cache 只有
        (T, d_c) 和 (T, d_rope), per-head K/V 在 attention 时才由 C @ W_UK / W_UV 现场还原。
真实系统: vLLM / SGLang 的 GQA kernel 靠 stride 广播不复制 KV; FlashMLA 走 absorb 形式
        (MLA decode ≡ head_dim=d_c+d_rope 的 MQA), 对应本文件 absorb=True 分支。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import dense_attention, rms_norm
from llm_infer.core.tiny_model import apply_rope, precompute_rope

MAX_T = 256


# ---- (a) 显存公式 ----------------------------------------------------- #

def kv_bytes_per_token(n_kv: int, d_head: int, n_layer: int, nbytes: int = 2) -> int:
    return 2 * n_kv * d_head * n_layer * nbytes          # 2 = K 和 V


def mla_bytes_per_token(d_c: int, d_rope: int, n_layer: int, nbytes: int = 2) -> int:
    return (d_c + d_rope) * n_layer * nbytes             # 没有 "2·": K/V 共用同一个 latent


# ---- (b) MHA / GQA / MQA: 一条代码路径 ---------------------------------- #

def _rand(rs, *shape):
    return (rs.randn(*shape) / np.sqrt(shape[0])).astype(np.float32)


class GQALayer:
    """n_kv == n_head → MHA; 1 < n_kv < n_head → GQA; n_kv == 1 → MQA。"""

    def __init__(self, rs, D: int, n_head: int, n_kv: int, d_head: int):
        assert n_head % n_kv == 0
        self.n_head, self.n_kv, self.d_head = n_head, n_kv, d_head
        self.wq = _rand(rs, D, n_head * d_head)
        self.wk = _rand(rs, D, n_kv * d_head)            # KV 投影只有 n_kv 个头 → cache 小 n_head/n_kv 倍
        self.wv = _rand(rs, D, n_kv * d_head)
        self.wo = _rand(rs, n_head * d_head, D)
        self.cos, self.sin = precompute_rope(d_head, MAX_T)

    def new_cache(self):
        z = np.zeros((0, self.n_kv, self.d_head), np.float32)
        return (z, z)

    def forward(self, x, cache):
        """x (T, D) 新 token; cache = (K, V) 各 (ctx, n_kv, d_head), K 存 post-RoPE。"""
        T, ctx = x.shape[0], cache[0].shape[0]
        H, G, dh = self.n_head, self.n_kv, self.d_head
        q = apply_rope((x @ self.wq).reshape(T, H, dh), self.cos, self.sin, ctx)   # (T, H, dh)
        k = apply_rope((x @ self.wk).reshape(T, G, dh), self.cos, self.sin, ctx)   # (T, G, dh)
        v = (x @ self.wv).reshape(T, G, dh)
        K = np.concatenate([cache[0], k])                                          # (Tk, G, dh)
        V = np.concatenate([cache[1], v])
        # 每组 H/G 个 query 头共享 1 个 KV 头: 靠广播, 不 repeat KV (真实 kernel 也不复制)
        qg = q.transpose(1, 0, 2).reshape(G, H // G, T, dh)                        # (G, H/G, T, dh)
        Kg = K.transpose(1, 0, 2)[:, None]                                         # (G, 1, Tk, dh)
        Vg = V.transpose(1, 0, 2)[:, None]
        o = dense_attention(qg, Kg, Vg)                                            # (G, H/G, T, dh)
        o = o.reshape(H, T, dh).transpose(1, 0, 2).reshape(T, H * dh)              # (T, H·dh)
        # astype 是保险: 任何一处混入 fp64 标量 (NumPy2 会升精度) 都会让下一层 cache 变 8 字节, nbytes 断言对不上
        return (o @ self.wo).astype(np.float32), (K, V)


def mha_reference(x, layer: GQALayer):
    """独立写法的 MHA 基线: 把每个 KV 头的权重复制给组内所有 query 头, 再逐头循环 (无 cache)。"""
    T, H, G, dh = x.shape[0], layer.n_head, layer.n_kv, layer.d_head
    rep = lambda w: np.repeat(w.reshape(-1, G, dh), H // G, axis=1)                # (D, G, dh) → (D, H, dh)
    wq, wk, wv = layer.wq.reshape(-1, H, dh), rep(layer.wk), rep(layer.wv)
    heads = []
    for h in range(H):
        q = apply_rope(x @ wq[:, h], layer.cos, layer.sin)                         # (T, dh)
        k = apply_rope(x @ wk[:, h], layer.cos, layer.sin)
        heads.append(dense_attention(q, k, x @ wv[:, h]))                          # (T, dh)
    return np.concatenate(heads, axis=-1) @ layer.wo                               # (T, D)


# ---- (b') MLA: 只缓存 latent + 解耦 RoPE key ---------------------------- #

class MLALayer:
    """DeepSeek-V2/V3 Multi-head Latent Attention (简化: 省掉 q 的低秩压缩和 c_kv 上的 RMSNorm)。"""

    def __init__(self, rs, D: int, n_head: int, d_nope: int, d_rope: int, d_v: int, d_c: int):
        self.n_head, self.d_nope, self.d_rope, self.d_v, self.d_c = n_head, d_nope, d_rope, d_v, d_c
        self.wq = _rand(rs, D, n_head * (d_nope + d_rope))
        self.w_dkv = _rand(rs, D, d_c)                   # 下投影: 这是唯一进 cache 的 KV 信息
        self.w_uk = _rand(rs, d_c, n_head * d_nope)      # 上投影: latent → 每头 K 的 "无位置" 部分
        self.w_uv = _rand(rs, d_c, n_head * d_v)
        self.w_kr = _rand(rs, D, d_rope)                 # 解耦 RoPE key: 所有头共享 1 份 (像 MQA)
        self.wo = _rand(rs, n_head * d_v, D)
        self.cos, self.sin = precompute_rope(d_rope, MAX_T)

    def new_cache(self):
        return (np.zeros((0, self.d_c), np.float32), np.zeros((0, self.d_rope), np.float32))

    def forward(self, x, cache, absorb: bool = False):
        """cache = (C (ctx, d_c), k_rope (ctx, d_rope))。absorb=True 走 FlashMLA 式的吸收形式。"""
        T, ctx = x.shape[0], cache[0].shape[0]
        H, dn, dr, dv, dc = self.n_head, self.d_nope, self.d_rope, self.d_v, self.d_c
        # latent 不加 RoPE: 若加了, 位置相关的旋转夹在 W_UK 前面, W_UK 就无法被吸收进 W_Q
        C = np.concatenate([cache[0], x @ self.w_dkv])                               # (Tk, dc)
        KR = np.concatenate([cache[1], apply_rope(x @ self.w_kr, self.cos, self.sin, ctx)])  # (Tk, dr)
        q = (x @ self.wq).reshape(T, H, dn + dr)
        q_nope = q[..., :dn].transpose(1, 0, 2)                                      # (H, T, dn)
        q_rope = apply_rope(q[..., dn:], self.cos, self.sin, ctx).transpose(1, 0, 2)  # (H, T, dr)
        Tk = C.shape[0]
        if not absorb:
            # 朴素形式: 现场把 latent 还原成每头 K/V (临时张量, 不进 cache)
            k_nope = (C @ self.w_uk).reshape(Tk, H, dn).transpose(1, 0, 2)           # (H, Tk, dn)
            Vh = (C @ self.w_uv).reshape(Tk, H, dv).transpose(1, 0, 2)               # (H, Tk, dv)
            Kh = np.concatenate([k_nope, np.broadcast_to(KR, (H, Tk, dr))], -1)      # (H, Tk, dn+dr)
            o = dense_attention(np.concatenate([q_nope, q_rope], -1), Kh, Vh)        # (H, T, dv)
        else:
            # 吸收形式: q·(C W_UK)ᵀ = (q W_UKᵀ)·Cᵀ → 把 W_UK 乘到 q 上, K 直接就是 cache 本身。
            # 于是 MLA decode ≡ 一个 head_dim = dc+dr 的 MQA, 全程不物化 per-head K/V。
            w_uk = self.w_uk.reshape(dc, H, dn).transpose(1, 2, 0)                   # (H, dn, dc)
            q_abs = np.concatenate([q_nope @ w_uk, q_rope], -1)                      # (H, T, dc+dr)
            q_abs = q_abs * np.sqrt((dc + dr) / (dn + dr))   # dense_attention 会除 √(dc+dr), 补回 MLA 的 √(dn+dr)
            K_shared = np.concatenate([C, KR], -1)[None]                             # (1, Tk, dc+dr)
            o_lat = dense_attention(q_abs, K_shared, C[None])                        # (H, T, dc)
            o = o_lat @ self.w_uv.reshape(dc, H, dv).transpose(1, 0, 2)              # (H, T, dv): W_UV 同理后乘
        return (o.transpose(1, 0, 2).reshape(T, H * dv) @ self.wo).astype(np.float32), (C, KR)


# ---- 多层堆叠 (只有 attention 子层, 够验证 cache 了) -------------------- #

def run_stack(layers, x, caches=None, **kw):
    """x (T, D) → (out (T, D), 新 caches)。caches=None 即 prefill。"""
    caches = caches or [l.new_cache() for l in layers]
    new = []
    for layer, cache in zip(layers, caches):
        h, cache = layer.forward(rms_norm(x, 1.0), cache, **kw)
        x = x + h
        new.append(cache)
    return x, new


def cache_nbytes(caches) -> int:
    return sum(a.nbytes for c in caches for a in c)

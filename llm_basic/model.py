"""
model.py — 极简 Transformer 语言模型，前向 + 反向全部手写。

设计原则
========
- 不依赖 torch / jax；只用 numpy 当"会广播的容器"
- 每个组件成对出现：xxx_forward(...) → (out, cache);  xxx_backward(dout, cache) → (dx, ...dparams)
- forward 把"反向需要的中间量"塞进 cache，backward 直接取用
- 参数全部存在一个 dict W 里；update 时返回新 dict（不就地修改）

形状约定
========
B = batch_size
T = sequence_length
D = model_dim     (单头 attention 的 head_dim 也等于 D)
H = mlp_hidden_dim
V = vocab_size

模型结构（1 层 1 头，最简）
=========================
    ids (B,T)
        │
        ▼
    tok_emb(V,D) + pos_emb(T_max,D)        加法
        │
        ▼
    ┌─ Block ─────────────────────────┐
    │   x → RMSNorm → Attn(causal) ─┐ │
    │                                + │
    │                                ↓ │
    │   h → RMSNorm → MLP(ReLU) ────┐ │
    │                                + │
    └────────────────────────────────┘
        │
        ▼
    RMSNorm
        │
        ▼
    lm_head(D,V) → logits (B,T,V)
"""
from __future__ import annotations

from typing import Any

import numpy as np

# 整个模块统一 float64：
#   - 训练慢一点不要紧，模型很小
#   - gradcheck 需要 float64 的精度，否则数值梯度跟解析梯度对不上
DTYPE = np.float64


# ============================================================
# 工具函数
# ============================================================
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定 softmax：减去 max 再 exp。"""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T: int) -> np.ndarray:
    """返回 (T, T) bool 矩阵，True 表示要被屏蔽（softmax 前置 -inf）的位置。

    位置 (i, j) 表示"query 行 i 看 key 列 j"。
    j > i 是未来位置，要屏蔽 → 上三角（不含对角线）为 True。
    """
    return np.triu(np.ones((T, T), dtype=bool), k=1)


# ============================================================
# 1. Embedding
# ============================================================
def embedding_forward(ids: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    ids: (B, T) int
    W:   (V, D)
    out: (B, T, D)
    """
    out = W[ids]                 # numpy fancy indexing
    cache = (ids, W.shape)
    return out, cache


def embedding_backward(dout: np.ndarray, cache: tuple) -> np.ndarray:
    """
    dout: (B, T, D)
    返回 dW: (V, D)

    每个 ids[b,t] 处 W[ids[b,t]] 被使用了一次；
    多次使用同一行 → 梯度需要相加 → np.add.at 处理重复索引。
    """
    ids, (V, D) = cache
    dW = np.zeros((V, D), dtype=DTYPE)
    np.add.at(dW, ids, dout)
    return dW


# ============================================================
# 2. Linear: y = x @ W + b
# ============================================================
def linear_forward(
    x: np.ndarray, W: np.ndarray, b: np.ndarray | None
) -> tuple[np.ndarray, tuple]:
    out = x @ W
    if b is not None:
        out = out + b
    cache = (x, W, b is not None)
    return out, cache


def linear_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    chain rule:
      y = x @ W + b
      dL/dx = dL/dy @ W.T
      dL/dW = x.T @ dL/dy           ← 把所有 batch 维 flatten 再算
      dL/db = sum_over_batch(dL/dy)
    """
    x, W, has_b = cache
    in_dim = x.shape[-1]
    out_dim = dout.shape[-1]

    x_flat = x.reshape(-1, in_dim)         # (N, in)
    dout_flat = dout.reshape(-1, out_dim)  # (N, out)
    dW = x_flat.T @ dout_flat              # (in, out)
    db = dout_flat.sum(axis=0) if has_b else None
    dx = dout @ W.T                        # (..., in)
    return dx, dW, db


# ============================================================
# 3. RMSNorm
# ============================================================
# 公式：
#   rms(x) = sqrt(mean(x²) + eps)        ← scalar per row
#   x_hat  = x / rms                     ← shape (..., D)
#   y      = g * x_hat                   ← g shape (D,)
#
# 反向传播推导（设 D 为最后一维大小）：
#   dL/dg_i = sum_over_batch( dL/dy_i * x_hat_i )
#
#   dL/dx_i = (g_i / rms) * dL/dy_i
#             - (x_i / (D * rms³)) * Σ_j ( dL/dy_j * g_j * x_j )
#
# 直觉：第二项是"标准化让每行总能量恒为常数"带来的耦合项。
def rmsnorm_forward(
    x: np.ndarray, g: np.ndarray, eps: float = 1e-5
) -> tuple[np.ndarray, tuple]:
    ms = (x * x).mean(axis=-1, keepdims=True)   # (..., 1)
    rms = np.sqrt(ms + eps)                     # (..., 1)
    x_hat = x / rms                             # (..., D)
    out = x_hat * g                             # (..., D)
    cache = (x, g, rms)
    return out, cache


def rmsnorm_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray]:
    x, g, rms = cache
    D = x.shape[-1]

    # dg：在最后一维上保持，其他维全部求和
    x_hat = x / rms
    dg = (dout * x_hat).reshape(-1, D).sum(axis=0)

    # dx：按上面推导
    c = dout * g                                # (..., D)
    s = (c * x).sum(axis=-1, keepdims=True)     # (..., 1)
    dx = c / rms - x * s / (D * rms ** 3)
    return dx, dg


# ============================================================
# 4. ReLU
# ============================================================
def relu_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.maximum(0.0, x)
    return out, x  # cache 直接是 x


def relu_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    x = cache
    return dout * (x > 0)


# ============================================================
# 5. 单头 causal self-attention
# ============================================================
def attention_forward(
    x: np.ndarray,
    Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    """
    x : (B, T, D)
    Wq, Wk, Wv, Wo : (D, D)   — 单头，head_dim = D
    out: (B, T, D)
    """
    B, T, D = x.shape

    Q, q_cache = linear_forward(x, Wq, None)
    K, k_cache = linear_forward(x, Wk, None)
    V, v_cache = linear_forward(x, Wv, None)

    # scores = Q · Kᵀ / √D    形状 (B, T, T)
    scale = 1.0 / np.sqrt(D)
    scores = Q @ K.transpose(0, 2, 1) * scale

    # causal mask：上三角置 -inf，softmax 后变 0
    mask = causal_mask(T)
    scores = np.where(mask, -np.inf, scores)

    attn = softmax(scores, axis=-1)          # (B, T, T)
    ctx = attn @ V                           # (B, T, D)

    out, o_cache = linear_forward(ctx, Wo, None)

    cache = (Q, K, V, attn, scale, mask, q_cache, k_cache, v_cache, o_cache)
    return out, cache


def attention_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    反向传播链路（与 forward 完全镜像）：
      Wo:   dctx, dWo
      ctx = attn @ V:
            dattn = dctx @ Vᵀ
            dV    = attnᵀ @ dctx
      attn = softmax(scores):
            softmax 反向公式（沿最后一维做，逐行独立）：
              ds_i = a_i * (da_i - Σ_j a_j da_j)
            mask 处 a=0 自然为 0；保险起见显式置 0。
      scores = Q · Kᵀ * scale:
            dscores ×= scale
            dQ = dscores @ K
            dK = dscoresᵀ @ Q
      Q,K,V 各自的 linear backward → 三路 dx 相加
    """
    Q, K, V, attn, scale, mask, q_cache, k_cache, v_cache, o_cache = cache

    # 1) 输出投影
    dctx, dWo, _ = linear_backward(dout, o_cache)            # dctx: (B, T, D)

    # 2) ctx = attn @ V
    dattn = dctx @ V.transpose(0, 2, 1)                      # (B, T, T)
    dV = attn.transpose(0, 2, 1) @ dctx                      # (B, T, D)

    # 3) softmax 反向（沿最后一维）
    sum_term = (dattn * attn).sum(axis=-1, keepdims=True)    # (B, T, 1)
    dscores = attn * (dattn - sum_term)                      # (B, T, T)
    dscores = np.where(mask, 0.0, dscores)                   # mask 处显式归零

    # 4) scores = Q @ Kᵀ * scale
    dscores = dscores * scale
    dQ = dscores @ K                                         # (B, T, D)
    dK = dscores.transpose(0, 2, 1) @ Q                      # (B, T, D)

    # 5) Q, K, V 投影反向，三路 dx 合并
    dx_q, dWq, _ = linear_backward(dQ, q_cache)
    dx_k, dWk, _ = linear_backward(dK, k_cache)
    dx_v, dWv, _ = linear_backward(dV, v_cache)

    dx = dx_q + dx_k + dx_v
    return dx, dWq, dWk, dWv, dWo


# ============================================================
# 6. MLP（两层，ReLU）
# ============================================================
def mlp_forward(
    x: np.ndarray,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    h, l1_cache = linear_forward(x, W1, b1)
    a, r_cache = relu_forward(h)
    out, l2_cache = linear_forward(a, W2, b2)
    return out, (l1_cache, r_cache, l2_cache)


def mlp_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l1_cache, r_cache, l2_cache = cache
    da, dW2, db2 = linear_backward(dout, l2_cache)
    dh = relu_backward(da, r_cache)
    dx, dW1, db1 = linear_backward(dh, l1_cache)
    return dx, dW1, db1, dW2, db2


# ============================================================
# 7. Transformer Block（pre-norm + 残差）
#    h   = x + Attn(RMSNorm(x))
#    out = h + MLP(RMSNorm(h))
# ============================================================
def block_forward(
    x: np.ndarray, W: dict[str, np.ndarray], prefix: str = "block_0_"
) -> tuple[np.ndarray, tuple]:
    n1, n1_cache = rmsnorm_forward(x, W[prefix + "norm1_g"])
    a, a_cache = attention_forward(
        n1,
        W[prefix + "attn_Wq"], W[prefix + "attn_Wk"],
        W[prefix + "attn_Wv"], W[prefix + "attn_Wo"],
    )
    h = x + a   # 残差 1

    n2, n2_cache = rmsnorm_forward(h, W[prefix + "norm2_g"])
    m, m_cache = mlp_forward(
        n2,
        W[prefix + "mlp_W1"], W[prefix + "mlp_b1"],
        W[prefix + "mlp_W2"], W[prefix + "mlp_b2"],
    )
    out = h + m  # 残差 2

    cache = (n1_cache, a_cache, n2_cache, m_cache)
    return out, cache


def block_backward(
    dout: np.ndarray, cache: tuple, prefix: str = "block_0_"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    残差的反向：out = h + m  ⇒  dh = dout, dm = dout
    一路梯度走 MLP 回到 h，再加上"绕过 MLP 直接来的 dout"。
    """
    n1_cache, a_cache, n2_cache, m_cache = cache

    # ── 残差 2: out = h + m
    dh = dout.copy()
    dm = dout

    dn2, dW1, db1, dW2, db2 = mlp_backward(dm, m_cache)
    dh_from_n2, dgn2 = rmsnorm_backward(dn2, n2_cache)
    dh = dh + dh_from_n2

    # ── 残差 1: h = x + a
    dx = dh.copy()
    da = dh

    dn1, dWq, dWk, dWv, dWo = attention_backward(da, a_cache)
    dx_from_n1, dgn1 = rmsnorm_backward(dn1, n1_cache)
    dx = dx + dx_from_n1

    grads = {
        prefix + "norm1_g": dgn1,
        prefix + "attn_Wq": dWq,
        prefix + "attn_Wk": dWk,
        prefix + "attn_Wv": dWv,
        prefix + "attn_Wo": dWo,
        prefix + "norm2_g": dgn2,
        prefix + "mlp_W1": dW1,
        prefix + "mlp_b1": db1,
        prefix + "mlp_W2": dW2,
        prefix + "mlp_b2": db2,
    }
    return dx, grads


# ============================================================
# 8. 完整 Transformer
# ============================================================
def transformer_forward(
    W: dict[str, np.ndarray], ids: np.ndarray
) -> tuple[np.ndarray, tuple]:
    """
    ids: (B, T) int
    返回 logits: (B, T, V) 和反向所需的 cache
    """
    B, T = ids.shape

    tok, tok_cache = embedding_forward(ids, W["tok_emb"])

    # 位置 embedding：直接 lookup 0..T-1，再广播相加
    pos_ids = np.broadcast_to(np.arange(T), (B, T))
    pos, pos_cache = embedding_forward(pos_ids, W["pos_emb"])

    h = tok + pos                                    # (B, T, D)

    h, blk_cache = block_forward(h, W, prefix="block_0_")
    h, n_cache = rmsnorm_forward(h, W["norm_f_g"])

    logits, head_cache = linear_forward(h, W["lm_head"], None)  # (B, T, V)

    cache = (tok_cache, pos_cache, blk_cache, n_cache, head_cache)
    return logits, cache


def transformer_backward(
    dlogits: np.ndarray, cache: tuple
) -> dict[str, np.ndarray]:
    tok_cache, pos_cache, blk_cache, n_cache, head_cache = cache

    dh, dlm_head, _ = linear_backward(dlogits, head_cache)
    dh, dnf = rmsnorm_backward(dh, n_cache)
    dh, blk_grads = block_backward(dh, blk_cache, prefix="block_0_")

    # h = tok + pos  →  两条 embedding 都收到 dh
    dtok_emb = embedding_backward(dh, tok_cache)
    dpos_emb = embedding_backward(dh, pos_cache)

    return {
        "tok_emb": dtok_emb,
        "pos_emb": dpos_emb,
        "norm_f_g": dnf,
        "lm_head": dlm_head,
        **blk_grads,
    }


# ============================================================
# 9. Cross-entropy 损失 + 反向（fused）
# ============================================================
def cross_entropy_forward_backward(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    logits:  (B, T, V)
    targets: (B, T) int

    把 softmax + NLL 合在一起：dlogits = (probs - onehot(target)) / N
    这样数值稳定，又避免显式构造 one-hot 矩阵。
    """
    B, T, V = logits.shape
    N = B * T
    z = logits.reshape(N, V)
    y = targets.reshape(N)

    # 数值稳定 softmax → log p
    z_shift = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z_shift)
    probs = e / e.sum(axis=-1, keepdims=True)            # (N, V)

    # NLL：负对数似然
    nll = -np.log(probs[np.arange(N), y] + 1e-12)
    loss = float(nll.mean())

    # 梯度
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    return loss, dlogits.reshape(B, T, V)


# ============================================================
# 10. 参数初始化
# ============================================================
def init_weights(
    config: dict[str, Any], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """所有线性层用 std=0.02 的高斯（GPT-2 风格），RMSNorm gain=1，bias=0。"""
    V = config["vocab_size"]
    D = config["dim"]
    H = config["hidden_dim"]
    T = config["max_seq_len"]
    std = 0.02

    def randn(*shape):
        return (rng.standard_normal(shape) * std).astype(DTYPE)

    return {
        "tok_emb": randn(V, D),
        "pos_emb": randn(T, D),
        "block_0_norm1_g": np.ones(D, dtype=DTYPE),
        "block_0_attn_Wq": randn(D, D),
        "block_0_attn_Wk": randn(D, D),
        "block_0_attn_Wv": randn(D, D),
        "block_0_attn_Wo": randn(D, D),
        "block_0_norm2_g": np.ones(D, dtype=DTYPE),
        "block_0_mlp_W1": randn(D, H),
        "block_0_mlp_b1": np.zeros(H, dtype=DTYPE),
        "block_0_mlp_W2": randn(H, D),
        "block_0_mlp_b2": np.zeros(D, dtype=DTYPE),
        "norm_f_g": np.ones(D, dtype=DTYPE),
        "lm_head": randn(D, V),
    }

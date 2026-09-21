"""
model.py — 纯 numpy 的极简 GPT：前向 + 反向全部手写，没有自动微分。

解决什么问题：torch 的 loss.backward() 是黑盒；这里每个算子都成对出现
    xxx_forward(...) -> (out, cache)      xxx_backward(dout, cache) -> (dx, *dparams)
forward 把反向要用的中间量塞进 cache，backward 严格按 forward 的逆序取用。

结构（pre-norm，n_layer 层，单头）：
    ids(B,T) → tok_emb + pos_emb → [x + Attn(RMSNorm(x)); h + MLP(RMSNorm(h))] × n_layer
             → RMSNorm → lm_head → logits(B,T,V)
关键公式：softmax 反向 ds = a ⊙ (da − Σ_j a_j da_j)；CE 反向 dlogits = (p − onehot)/N。

形状记号：B=batch, T=序列长, D=模型维(单头, head_dim=D), H=MLP 中间维, V=词表。
读代码时盯住：每个 backward 里 dout 的形状——它永远等于对应 forward 输出的形状。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# 全程 float64：gradcheck 的中心差分需要这个精度；模型很小，慢一点无所谓
DTYPE = np.float64


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定 softmax：先减 max 再 exp（结果不变，但 exp 不会溢出）。"""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T: int) -> np.ndarray:
    """(T, T) bool，True = 屏蔽。(i, j) 表示 query i 看 key j；j > i 是未来 → 上三角为 True。"""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


# ============================================================
# 1. Embedding：查表
# ============================================================
def embedding_forward(ids: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, tuple]:
    out = W[ids]                 # ids [B, T] 查 W [V, D] → [B, T, D]
    return out, (ids, W.shape)


def embedding_backward(dout: np.ndarray, cache: tuple) -> np.ndarray:
    ids, shape = cache
    dW = np.zeros(shape, dtype=DTYPE)          # [V, D]
    # 同一个 id 在 batch 里出现多次 → 梯度要累加。dW[ids] += dout 遇到重复索引只加一次，必须用 add.at
    np.add.at(dW, ids, dout)                   # dout [B, T, D] 散射回 [V, D]
    return dW


# ============================================================
# 2. Linear: y = x @ W + b
# ============================================================
def linear_forward(
    x: np.ndarray, W: np.ndarray, b: np.ndarray | None
) -> tuple[np.ndarray, tuple]:
    out = x @ W                                # [..., in] @ [in, out] → [..., out]
    if b is not None:
        out = out + b
    return out, (x, W, b is not None)


def linear_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """dx = dout @ Wᵀ;  dW = xᵀ @ dout;  db = Σ dout（W、b 被所有 batch/时间步共享 → 求和）。"""
    x, W, has_b = cache
    x_flat = x.reshape(-1, x.shape[-1])            # [N, in]，N = B*T
    dout_flat = dout.reshape(-1, dout.shape[-1])   # [N, out]
    dW = x_flat.T @ dout_flat                      # [in, N] @ [N, out] → [in, out]
    db = dout_flat.sum(axis=0) if has_b else None  # [out]
    dx = dout @ W.T                                # [..., out] @ [out, in] → [..., in]
    return dx, dW, db


# ============================================================
# 3. RMSNorm: y = g ⊙ x / rms(x),  rms = sqrt(mean(x²) + eps)
# ============================================================
# 反向：dx_i = g_i·dy_i / rms − x_i · Σ_j(dy_j g_j x_j) / (D · rms³)
# 第二项来自 rms 本身依赖整行 x：一行里每个元素的梯度互相耦合。
def rmsnorm_forward(
    x: np.ndarray, g: np.ndarray, eps: float = 1e-5
) -> tuple[np.ndarray, tuple]:
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)   # [..., 1]
    out = x / rms * g                                           # [..., D]
    return out, (x, g, rms)


def rmsnorm_backward(dout: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray]:
    x, g, rms = cache
    D = x.shape[-1]
    dg = (dout * x / rms).reshape(-1, D).sum(axis=0)   # [D]，g 被所有位置共享 → 求和
    c = dout * g                                       # [..., D]
    s = (c * x).sum(axis=-1, keepdims=True)            # [..., 1]  耦合项里的 Σ_j
    dx = c / rms - x * s / (D * rms ** 3)              # [..., D]
    return dx, dg


# ============================================================
# 4. ReLU
# ============================================================
def relu_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(0.0, x), x


def relu_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return dout * (cache > 0)      # x ≤ 0 的位置梯度被截断


# ============================================================
# 5. 单头 causal self-attention
# ============================================================
def attention_forward(
    x: np.ndarray,
    Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    """x [B, T, D]，四个权重都是 [D, D]，输出 [B, T, D]。"""
    B, T, D = x.shape
    Q, q_cache = linear_forward(x, Wq, None)       # [B, T, D]
    K, k_cache = linear_forward(x, Wk, None)       # [B, T, D]
    V, v_cache = linear_forward(x, Wv, None)       # [B, T, D]

    scale = 1.0 / np.sqrt(D)                       # 不缩放的话点积方差 ∝ D，softmax 会饱和
    scores = Q @ K.transpose(0, 2, 1) * scale      # [B, T, D] @ [B, D, T] → [B, T, T]
    mask = causal_mask(T)
    scores = np.where(mask, -np.inf, scores)       # 未来位置 → -inf → softmax 后权重为 0

    attn = softmax(scores, axis=-1)                # [B, T, T]，每行和为 1
    ctx = attn @ V                                 # [B, T, T] @ [B, T, D] → [B, T, D]
    out, o_cache = linear_forward(ctx, Wo, None)   # [B, T, D]

    return out, (Q, K, V, attn, scale, q_cache, k_cache, v_cache, o_cache)


def attention_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """forward 的逐行镜像。被 mask 的位置 attn=0，softmax 反向自动给出 dscores=0，无需再 mask。"""
    Q, K, V, attn, scale, q_cache, k_cache, v_cache, o_cache = cache

    dctx, dWo, _ = linear_backward(dout, o_cache)            # [B, T, D]

    # ctx = attn @ V
    dattn = dctx @ V.transpose(0, 2, 1)                      # [B, T, D] @ [B, D, T] → [B, T, T]
    dV = attn.transpose(0, 2, 1) @ dctx                      # [B, T, T] @ [B, T, D] → [B, T, D]

    # softmax 反向（逐行独立）：ds = a ⊙ (da − Σ_j a_j da_j)
    sum_term = (dattn * attn).sum(axis=-1, keepdims=True)    # [B, T, 1]
    dscores = attn * (dattn - sum_term) * scale              # [B, T, T]

    # scores = Q @ Kᵀ
    dQ = dscores @ K                                         # [B, T, T] @ [B, T, D] → [B, T, D]
    dK = dscores.transpose(0, 2, 1) @ Q                      # [B, T, T] @ [B, T, D] → [B, T, D]

    # x 同时喂给了 Q/K/V 三条支路 → 三路 dx 相加
    dx_q, dWq, _ = linear_backward(dQ, q_cache)
    dx_k, dWk, _ = linear_backward(dK, k_cache)
    dx_v, dWv, _ = linear_backward(dV, v_cache)
    return dx_q + dx_k + dx_v, dWq, dWk, dWv, dWo


# ============================================================
# 6. MLP: Linear(D→H) → ReLU → Linear(H→D)
# ============================================================
def mlp_forward(
    x: np.ndarray,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    h, l1_cache = linear_forward(x, W1, b1)        # [B, T, D] → [B, T, H]
    a, r_cache = relu_forward(h)                   # [B, T, H]
    out, l2_cache = linear_forward(a, W2, b2)      # [B, T, H] → [B, T, D]
    return out, (l1_cache, r_cache, l2_cache)


def mlp_backward(
    dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l1_cache, r_cache, l2_cache = cache
    da, dW2, db2 = linear_backward(dout, l2_cache)   # [B, T, H]
    dh = relu_backward(da, r_cache)                  # [B, T, H]
    dx, dW1, db1 = linear_backward(dh, l1_cache)     # [B, T, D]
    return dx, dW1, db1, dW2, db2


# ============================================================
# 7. Transformer Block（pre-norm + 残差）
#    h = x + Attn(RMSNorm(x));   out = h + MLP(RMSNorm(h))
# ============================================================
def block_forward(
    x: np.ndarray, W: dict[str, np.ndarray], prefix: str
) -> tuple[np.ndarray, tuple]:
    n1, n1_cache = rmsnorm_forward(x, W[prefix + "norm1_g"])
    a, a_cache = attention_forward(
        n1,
        W[prefix + "attn_Wq"], W[prefix + "attn_Wk"],
        W[prefix + "attn_Wv"], W[prefix + "attn_Wo"],
    )
    h = x + a                                      # 残差 1  [B, T, D]

    n2, n2_cache = rmsnorm_forward(h, W[prefix + "norm2_g"])
    m, m_cache = mlp_forward(
        n2,
        W[prefix + "mlp_W1"], W[prefix + "mlp_b1"],
        W[prefix + "mlp_W2"], W[prefix + "mlp_b2"],
    )
    return h + m, (n1_cache, a_cache, n2_cache, m_cache)   # 残差 2


def block_backward(
    dout: np.ndarray, cache: tuple, prefix: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """残差 out = h + m 的反向：梯度原样走捷径，再加上穿过子层回来的那一路。"""
    n1_cache, a_cache, n2_cache, m_cache = cache

    dn2, dW1, db1, dW2, db2 = mlp_backward(dout, m_cache)
    dh_from_n2, dgn2 = rmsnorm_backward(dn2, n2_cache)
    dh = dout + dh_from_n2                         # 捷径 + MLP 支路

    dn1, dWq, dWk, dWv, dWo = attention_backward(dh, a_cache)
    dx_from_n1, dgn1 = rmsnorm_backward(dn1, n1_cache)
    dx = dh + dx_from_n1                           # 捷径 + attention 支路

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
# 8. 完整 Transformer：多层 = 对 block 的一个 for 循环
# ============================================================
def num_layers(W: dict[str, np.ndarray]) -> int:
    """层数直接从参数名数出来（block_{i}_norm1_g），所以不带 n_layer 的旧 ckpt 也能加载。"""
    return sum(k.endswith("_norm1_g") for k in W)


def transformer_forward(
    W: dict[str, np.ndarray], ids: np.ndarray
) -> tuple[np.ndarray, tuple]:
    """ids [B, T] int → logits [B, T, V] 和反向所需的 cache。"""
    B, T = ids.shape
    tok, tok_cache = embedding_forward(ids, W["tok_emb"])          # [B, T, D]
    pos_ids = np.broadcast_to(np.arange(T), (B, T))                # [B, T]，每行都是 0..T-1
    pos, pos_cache = embedding_forward(pos_ids, W["pos_emb"])      # [B, T, D]
    h = tok + pos                                                  # [B, T, D]

    blk_caches = []
    for i in range(num_layers(W)):
        h, c = block_forward(h, W, prefix=f"block_{i}_")           # [B, T, D]
        blk_caches.append(c)

    h, n_cache = rmsnorm_forward(h, W["norm_f_g"])
    logits, head_cache = linear_forward(h, W["lm_head"], None)     # [B, T, D] @ [D, V] → [B, T, V]
    return logits, (tok_cache, pos_cache, blk_caches, n_cache, head_cache)


def transformer_backward(dlogits: np.ndarray, cache: tuple) -> dict[str, np.ndarray]:
    tok_cache, pos_cache, blk_caches, n_cache, head_cache = cache

    dh, dlm_head, _ = linear_backward(dlogits, head_cache)         # [B, T, D]
    dh, dnf = rmsnorm_backward(dh, n_cache)
    grads = {"norm_f_g": dnf, "lm_head": dlm_head}

    for i in reversed(range(len(blk_caches))):                     # 反向：从最后一层往回走
        dh, g = block_backward(dh, blk_caches[i], prefix=f"block_{i}_")
        grads.update(g)

    # h = tok + pos → 两张 embedding 表收到同一个 dh
    grads["tok_emb"] = embedding_backward(dh, tok_cache)
    grads["pos_emb"] = embedding_backward(dh, pos_cache)
    return grads


# ============================================================
# 9. Cross-entropy（softmax + NLL 融合）
# ============================================================
def cross_entropy_forward_backward(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """logits [B, T, V], targets [B, T] → (loss, dlogits [B, T, V])。

    融合后梯度极简：dlogits = (p − onehot(y)) / N，也不用显式构造 one-hot。
    """
    B, T, V = logits.shape
    N = B * T
    z = logits.reshape(N, V)                                   # [N, V]
    y = targets.reshape(N)                                     # [N]

    # log-softmax = z − logsumexp(z)：全程在 log 域，不会出现 log(0)
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))   # [N, V]
    loss = float(-logp[np.arange(N), y].mean())

    dlogits = np.exp(logp)                                     # p  [N, V]
    dlogits[np.arange(N), y] -= 1.0                            # p − onehot
    dlogits /= N                                               # loss 是 N 个位置的平均
    return loss, dlogits.reshape(B, T, V)


# ============================================================
# 10. 参数初始化
# ============================================================
def init_weights(
    config: dict[str, Any], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """线性层/embedding ~ N(0, 0.02²)（GPT-2 风格），RMSNorm gain=1，bias=0。

    logits 初始 ≈ 0 → 预测近似均匀 → 初始 loss ≈ ln V（train.py 里有断言）。
    config["n_layer"] 缺省为 1；参数名 block_{i}_* 与旧 ckpt 完全一致。
    """
    V = config["vocab_size"]
    D = config["dim"]
    H = config["hidden_dim"]
    T = config["max_seq_len"]

    def randn(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(DTYPE)

    W = {"tok_emb": randn(V, D), "pos_emb": randn(T, D)}
    for i in range(config.get("n_layer", 1)):
        p = f"block_{i}_"
        W[p + "norm1_g"] = np.ones(D, dtype=DTYPE)
        W[p + "attn_Wq"] = randn(D, D)
        W[p + "attn_Wk"] = randn(D, D)
        W[p + "attn_Wv"] = randn(D, D)
        W[p + "attn_Wo"] = randn(D, D)
        W[p + "norm2_g"] = np.ones(D, dtype=DTYPE)
        W[p + "mlp_W1"] = randn(D, H)
        W[p + "mlp_b1"] = np.zeros(H, dtype=DTYPE)
        W[p + "mlp_W2"] = randn(H, D)
        W[p + "mlp_b2"] = np.zeros(D, dtype=DTYPE)
    W["norm_f_g"] = np.ones(D, dtype=DTYPE)
    W["lm_head"] = randn(D, V)
    return W

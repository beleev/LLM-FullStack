"""
tiny_model.py — 极简 Decoder-only Transformer (numpy, 仅推理前向)

设计目标
========
1. 让每个 mXX 推理优化模块都有"一个能跑的模型"做实验
2. 暴露两种前向接口:
   - prefill(ids)        : 处理一段 prompt, 返回 logits 与新生成的 KV cache
   - decode_step(id, kv) : 给一个 token + 已有 KV, 返回 logits 与更新后的 KV
3. 所有数学运算用 numpy 写明, 不调用任何 torch / jax

模型结构 (LLaMA 风格简化版, 单头注意力)
======================================
    ids (T,)
        │
        ▼
    tok_emb(V, D) + RoPE                位置编码
        │
        ▼
    ┌─ Block × N_LAYER ────────────────┐
    │   x → RMSNorm → Attn(causal) ──┐ │
    │                                + │
    │                                ↓ │
    │   h → RMSNorm → SwiGLU MLP ────┐ │
    │                                + │
    └────────────────────────────────┘
        │
        ▼
    RMSNorm
        │
        ▼
    lm_head(D, V) → logits (T, V)

形状约定 (无 batch, 教学最简; 推理引擎里 batch 由 scheduler 拼接)
=====================================================================
    T  = 序列长度
    D  = hidden dim         (default 32)
    H  = mlp hidden         (default 64)
    V  = vocab size
    N  = num layers         (default 4)

KV cache 结构
=============
    kv_cache: List[Tuple[K, V]] 长度 = N
        K, V shape = (T, D)        (单头, 否则 (T, n_head, head_dim))
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from llm_infer.core.utils import softmax, rms_norm, silu, causal_mask


# --------------------------------------------------------------------- #
# 配置                                                                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 32
    d_mlp: int = 64
    n_layer: int = 4
    max_seq_len: int = 512
    rope_base: float = 10000.0


# --------------------------------------------------------------------- #
# 权重容器                                                              #
# --------------------------------------------------------------------- #

@dataclass
class LayerWeights:
    """单层权重。命名对齐 HuggingFace 习惯, 方便后续模块替换为真实权重。"""
    # attention
    wq: np.ndarray  # (D, D)
    wk: np.ndarray  # (D, D)
    wv: np.ndarray  # (D, D)
    wo: np.ndarray  # (D, D)
    norm1_g: np.ndarray  # (D,)
    # mlp (SwiGLU: gate + up + down)
    w_gate: np.ndarray   # (D, H)
    w_up: np.ndarray     # (D, H)
    w_down: np.ndarray   # (H, D)
    norm2_g: np.ndarray  # (D,)


@dataclass
class ModelWeights:
    tok_emb: np.ndarray        # (V, D)
    layers: List[LayerWeights] = field(default_factory=list)
    norm_f_g: np.ndarray = None  # (D,)
    lm_head: np.ndarray = None   # (D, V)


def init_weights(cfg: ModelConfig, seed: int = 42) -> ModelWeights:
    """随机初始化权重 (xavier-ish)。教学用, 输出是乱码但前向流程正确。"""
    rs = np.random.RandomState(seed)
    D, H, V, N = cfg.d_model, cfg.d_mlp, cfg.vocab_size, cfg.n_layer

    def rand(shape):
        # 简化的 xavier: std = 1/sqrt(fan_in)
        fan_in = shape[0]
        return rs.randn(*shape).astype(np.float32) * (1.0 / np.sqrt(fan_in))

    layers = []
    for _ in range(N):
        layers.append(LayerWeights(
            wq=rand((D, D)), wk=rand((D, D)), wv=rand((D, D)), wo=rand((D, D)),
            norm1_g=np.ones(D, dtype=np.float32),
            w_gate=rand((D, H)), w_up=rand((D, H)), w_down=rand((H, D)),
            norm2_g=np.ones(D, dtype=np.float32),
        ))

    return ModelWeights(
        tok_emb=rand((V, D)),
        layers=layers,
        norm_f_g=np.ones(D, dtype=np.float32),
        lm_head=rand((D, V)),
    )


# --------------------------------------------------------------------- #
# RoPE (precompute)                                                     #
# --------------------------------------------------------------------- #

def precompute_rope(d: int, max_t: int, base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """预计算 cos / sin 表, shape = (max_t, d/2)。

    RoPE 公式: 把 d 维向量看作 d/2 个 2D 复数, 第 k 个复数旋转角度
        theta_k(pos) = pos / base^(2k/d)
    """
    half = d // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half).astype(np.float32) * 2.0 / d))
    pos = np.arange(max_t).astype(np.float32)
    freqs = np.outer(pos, inv_freq)         # (max_t, d/2)
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, start_pos: int = 0) -> np.ndarray:
    """对 x 应用 RoPE。x: (T, D) 或 (T, n_head, head_dim)。"""
    t = x.shape[0]
    c = cos[start_pos:start_pos + t]   # (T, d/2)
    s = sin[start_pos:start_pos + t]
    # 把最后一维 D 切成两半: x = [x1 ; x2], 各 d/2
    x1, x2 = np.split(x, 2, axis=-1)
    # 旋转: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    # 广播: c/s 是 (T, d/2), x1/x2 是 (T, d/2) → 直接乘
    if x1.ndim == 3:  # 多头, 在 head 维广播
        c = c[:, None, :]
        s = s[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


# --------------------------------------------------------------------- #
# 单层前向                                                              #
# --------------------------------------------------------------------- #

def attn_forward(
    x: np.ndarray,                     # (T_q, D)
    layer: LayerWeights,
    cos: np.ndarray, sin: np.ndarray,
    kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    start_pos: int = 0,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """单头注意力 + RoPE + KV cache。

    若传入 kv_cache, 则把新算的 K/V 拼到尾部, 实现增量计算 (decode);
    否则当作 prefill, kv_cache 从头建立。

    返回 (out, (K_full, V_full))。
    """
    # 1) Q K V 投影
    q = x @ layer.wq                       # (T_q, D)
    k = x @ layer.wk
    v = x @ layer.wv

    # 2) RoPE 加在 Q / K 上 (V 不加)
    q = apply_rope(q, cos, sin, start_pos=start_pos)
    k = apply_rope(k, cos, sin, start_pos=start_pos)

    # 3) 拼接历史 KV
    if kv_cache is not None:
        K_prev, V_prev = kv_cache
        K = np.concatenate([K_prev, k], axis=0)
        V = np.concatenate([V_prev, v], axis=0)
    else:
        K, V = k, v

    # 4) 注意力分数 + 因果 mask + softmax
    d = q.shape[-1]
    scores = (q @ K.T) / np.sqrt(d)        # (T_q, T_k)
    mask = causal_mask(q.shape[0], K.shape[0])
    attn = softmax(scores + mask, axis=-1)
    out = attn @ V                          # (T_q, D)

    # 5) 输出投影
    out = out @ layer.wo
    return out, (K, V)


def mlp_forward(x: np.ndarray, layer: LayerWeights) -> np.ndarray:
    """SwiGLU MLP: down( silu(gate(x)) * up(x) )"""
    gate = silu(x @ layer.w_gate)
    up = x @ layer.w_up
    return (gate * up) @ layer.w_down


def block_forward(
    x: np.ndarray,
    layer: LayerWeights,
    cos: np.ndarray, sin: np.ndarray,
    kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    start_pos: int = 0,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """PreLN 风格: norm → sublayer → residual"""
    # attention sublayer
    h = rms_norm(x, layer.norm1_g)
    h, new_kv = attn_forward(h, layer, cos, sin, kv_cache, start_pos)
    x = x + h
    # mlp sublayer
    h = rms_norm(x, layer.norm2_g)
    h = mlp_forward(h, layer)
    x = x + h
    return x, new_kv


# --------------------------------------------------------------------- #
# 完整模型                                                              #
# --------------------------------------------------------------------- #

class TinyLM:
    """numpy 实现的极简 LM, 同时支持 prefill 与 decode_step。

    KVCache 类型: List[Tuple[K, V]], 每层一个; K/V shape (T, D)
    本类不持有 KV state, KV 由调用者保管 (推理引擎需要这种解耦)。
    """

    def __init__(self, cfg: ModelConfig, weights: Optional[ModelWeights] = None):
        self.cfg = cfg
        self.w = weights if weights is not None else init_weights(cfg)
        self.cos, self.sin = precompute_rope(cfg.d_model, cfg.max_seq_len, cfg.rope_base)

    # ------------------------------------------------------------- #
    # prefill: 一次处理整段 prompt                                  #
    # ------------------------------------------------------------- #

    def prefill(self, ids: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """ids: (T,) int → (logits (T, V), kv_cache list)"""
        x = self.w.tok_emb[ids]                  # (T, D)
        kv_cache = []
        for layer in self.w.layers:
            x, kv = block_forward(x, layer, self.cos, self.sin, None, start_pos=0)
            kv_cache.append(kv)
        x = rms_norm(x, self.w.norm_f_g)
        logits = x @ self.w.lm_head              # (T, V)
        return logits, kv_cache

    # ------------------------------------------------------------- #
    # decode_step: 单步, 拼接到已有 KV                              #
    # ------------------------------------------------------------- #

    def decode_step(
        self,
        token_id: int,
        kv_cache: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """token_id 单个 → (logits (V,), updated kv_cache)"""
        x = self.w.tok_emb[np.array([token_id])]  # (1, D)
        start = kv_cache[0][0].shape[0]            # 当前 context 长度
        new_kv = []
        for layer, kv in zip(self.w.layers, kv_cache):
            x, kv_new = block_forward(x, layer, self.cos, self.sin, kv, start_pos=start)
            new_kv.append(kv_new)
        x = rms_norm(x, self.w.norm_f_g)
        logits = (x @ self.w.lm_head).squeeze(0)   # (V,)
        return logits, new_kv

    # ------------------------------------------------------------- #
    # 简易完整生成 (greedy), 仅做基线对比用                          #
    # ------------------------------------------------------------- #

    def generate_greedy(self, prompt_ids: np.ndarray, max_new: int = 16) -> List[int]:
        out_ids: List[int] = list(prompt_ids)
        logits, kv = self.prefill(prompt_ids)
        next_id = int(np.argmax(logits[-1]))
        out_ids.append(next_id)
        for _ in range(max_new - 1):
            logits, kv = self.decode_step(next_id, kv)
            next_id = int(np.argmax(logits))
            out_ids.append(next_id)
        return out_ids


if __name__ == "__main__":
    # 自检
    cfg = ModelConfig()
    lm = TinyLM(cfg)
    print(f"model: {cfg.n_layer} layer, d={cfg.d_model}, V={cfg.vocab_size}")

    prompt = np.array([1, 10, 20, 30], dtype=np.int64)
    logits, kv = lm.prefill(prompt)
    print(f"prefill: logits {logits.shape}, kv layers={len(kv)}, K[0] {kv[0][0].shape}")

    logits1, kv1 = lm.decode_step(int(np.argmax(logits[-1])), kv)
    print(f"decode:  logits {logits1.shape}, K[0] grown to {kv1[0][0].shape}")

    out = lm.generate_greedy(prompt, max_new=8)
    print(f"greedy out ids: {out}")

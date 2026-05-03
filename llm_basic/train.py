"""
train.py — 训练循环。

整体流程
========
1) 读 train.bin / val.bin / meta.npz
2) 用 init_weights 初始化参数
3) 循环：
     - get_batch  随机采样 (B, T) token block
     - forward    前向 + cache
     - loss       cross-entropy
     - backward   反向，得到 grads
     - adam_step  优化器更新
     - 每 N 步算一次 val loss，打印
4) 训练完保存 ckpt.npz（参数 + 模型超参）
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from model import (
    cross_entropy_forward_backward,
    init_weights,
    transformer_backward,
    transformer_forward,
)
from optim import adam_init, adam_step

HERE = Path(__file__).parent
TRAIN_BIN = HERE / "train.bin"
VAL_BIN = HERE / "val.bin"
META_NPZ = HERE / "meta.npz"
CKPT_NPZ = HERE / "ckpt.npz"

# ============================================================
# 超参
# ============================================================
# 模型
DIM = 64           # hidden size
HIDDEN_DIM = 128   # MLP 中间层
SEQ_LEN = 64       # 上下文长度

# 训练
BATCH_SIZE = 32
LR = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20  # 算 val loss 时取多少个 batch 平均
SEED = 1337


# ============================================================
# 数据：随机采样 (B, T) 大小的 block
# ============================================================
def get_batch(
    data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    随机起点采样 batch_size 段长 seq_len 的 token 序列。
    x: (B, T)        — 输入
    y: (B, T)        — 目标，即 x 向右平移 1 位（标准 LM 训练目标）
    """
    # 起点范围：[0, len-seq_len-1]，预留一位给目标的最后一个字符
    ix = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix]).astype(np.int64)
    return x, y


def estimate_val_loss(
    W: dict[str, np.ndarray],
    val_data: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """在 val 上跑 EVAL_BATCHES 个 batch 求平均 loss。"""
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, BATCH_SIZE, SEQ_LEN, rng)
        logits, _ = transformer_forward(W, x)
        loss, _ = cross_entropy_forward_backward(logits, y)
        losses.append(loss)
    return float(np.mean(losses))


# ============================================================
# Checkpoint：把 W 和 config 一起塞进一个 npz
# ============================================================
def save_ckpt(W: dict[str, np.ndarray], config: dict, path: Path) -> None:
    """用 _config_<key> 前缀存 config 标量；其余键名直接是参数名。"""
    payload = {f"_config_{k}": np.array(v) for k, v in config.items()}
    payload.update(W)
    np.savez(path, **payload)


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    # 数据
    print(f"loading data from {TRAIN_BIN.name} / {VAL_BIN.name} / {META_NPZ.name}")
    train_data = np.fromfile(TRAIN_BIN, dtype=np.uint8)
    val_data = np.fromfile(VAL_BIN, dtype=np.uint8)
    meta = np.load(META_NPZ, allow_pickle=False)
    vocab_size = int(meta["vocab_size"])
    print(
        f"  train: {len(train_data):,} tokens"
        f" | val: {len(val_data):,} tokens"
        f" | vocab: {vocab_size}"
    )

    config = {
        "vocab_size": vocab_size,
        "dim": DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_seq_len": SEQ_LEN,
    }

    rng = np.random.default_rng(SEED)
    W = init_weights(config, rng)
    opt_state = adam_init(W)

    # 参数量
    n_params = sum(v.size for v in W.values())
    print(f"  model params: {n_params:,}")
    print(
        f"  config: dim={DIM}, hidden={HIDDEN_DIM}, seq_len={SEQ_LEN}, "
        f"batch={BATCH_SIZE}, lr={LR}, iters={MAX_ITERS}"
    )
    print()

    t0 = time.time()
    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, rng)

        # forward → loss
        logits, cache = transformer_forward(W, x)
        loss, dlogits = cross_entropy_forward_backward(logits, y)

        # backward
        grads = transformer_backward(dlogits, cache)

        # 优化器
        W, opt_state = adam_step(W, grads, opt_state, lr=LR)

        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            val_loss = estimate_val_loss(W, val_data, rng)
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | "
                f"train_loss {loss:.4f} | val_loss {val_loss:.4f} | "
                f"{elapsed:.1f}s elapsed"
            )

    save_ckpt(W, config, CKPT_NPZ)
    print(f"\ncheckpoint saved to {CKPT_NPZ.name}")


if __name__ == "__main__":
    main()

"""
train.py — 训练循环：get_batch → forward → loss → backward → (clip) → adam_step。

这就是 torch 里 `loss.backward(); opt.step()` 背后的全部内容，只是每一步都是显式函数调用。
目标：next-token 预测，y 是 x 右移一位；loss = 平均交叉熵，初始应 ≈ ln V = ln 65 ≈ 4.17
（均匀瞎猜），字符级 Tiny Shakespeare 上 2000 步约降到 2.0 左右。

用法（在 llm_basic/ 目录下）：
    python train.py                                   # 默认 2000 步，覆盖写 ckpt.npz
    python train.py --max-iters 200 --out /tmp/c.npz  # 快速试跑，不碰自带的 ckpt.npz
    python train.py --n-layer 2 --weight-decay 0.1 --grad-clip 1.0 --cosine --warmup 100
AdamW / 梯度裁剪 / cosine 调度默认全关。读代码时盯住：main() 里循环体那 5 行。
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from model import (
    cross_entropy_forward_backward,
    init_weights,
    transformer_backward,
    transformer_forward,
)
from optim import adam_init, adam_step, clip_grad_norm, cosine_lr

HERE = Path(__file__).parent
TRAIN_BIN = HERE / "train.bin"
VAL_BIN = HERE / "val.bin"
META_NPZ = HERE / "meta.npz"
CKPT_NPZ = HERE / "ckpt.npz"

# 模型
DIM = 64           # D
HIDDEN_DIM = 128   # H
SEQ_LEN = 64       # T，也是位置 embedding 的上限

# 训练
BATCH_SIZE = 32
LR = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20  # val loss 取多少个 batch 平均
SEED = 1337


def get_batch(
    data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """随机起点切 batch_size 段。x [B, T]；y [B, T] = x 右移一位（每个位置预测下一个 token）。"""
    ix = rng.integers(0, len(data) - seq_len - 1, size=batch_size)   # 留 1 位给 y 的最后一个
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)          # [B, T]
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix]).astype(np.int64)  # [B, T]
    return x, y


def estimate_val_loss(
    W: dict[str, np.ndarray], val_data: np.ndarray, rng: np.random.Generator
) -> float:
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, BATCH_SIZE, SEQ_LEN, rng)
        logits, _ = transformer_forward(W, x)
        losses.append(cross_entropy_forward_backward(logits, y)[0])
    return float(np.mean(losses))


def save_ckpt(W: dict[str, np.ndarray], config: dict, path: Path) -> None:
    """config 标量用 _config_<key> 前缀存，其余键名就是参数名（sample.py 的 load_ckpt 与之对应）。"""
    payload = {f"_config_{k}": np.array(v) for k, v in config.items()}
    payload.update(W)
    np.savez(path, **payload)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=MAX_ITERS)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--out", type=Path, default=CKPT_NPZ, help="checkpoint 输出路径")
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.0, help=">0 即 AdamW")
    p.add_argument("--grad-clip", type=float, default=0.0, help=">0 时按全局范数裁剪")
    p.add_argument("--cosine", action="store_true", help="warmup + cosine，终点 lr/10")
    p.add_argument("--warmup", type=int, default=100)
    args = p.parse_args()

    # memmap：不把文件读进内存，get_batch 切到哪一段才读哪一段（语料再大启动也是瞬间）
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint8, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint8, mode="r")
    vocab_size = int(np.load(META_NPZ, allow_pickle=False)["vocab_size"])
    print(f"train: {len(train_data):,} tokens | val: {len(val_data):,} tokens | vocab: {vocab_size}")

    config = {
        "vocab_size": vocab_size,
        "dim": DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_seq_len": SEQ_LEN,
        "n_layer": args.n_layer,
    }
    rng = np.random.default_rng(SEED)
    W = init_weights(config, rng)
    opt_state = adam_init(W)
    print(f"params: {sum(v.size for v in W.values()):,} | {config} | "
          f"batch={BATCH_SIZE}, lr={LR}, iters={args.max_iters}\n")

    t0 = time.time()
    for step in range(1, args.max_iters + 1):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, rng)

        logits, cache = transformer_forward(W, x)                  # [B, T, V]
        loss, dlogits = cross_entropy_forward_backward(logits, y)
        grads = transformer_backward(dlogits, cache)
        if args.grad_clip > 0:
            grads, _ = clip_grad_norm(grads, args.grad_clip)
        lr = cosine_lr(step, LR, LR / 10, args.warmup, args.max_iters) if args.cosine else LR
        W, opt_state = adam_step(W, grads, opt_state, lr=lr, weight_decay=args.weight_decay)

        if step == 1:
            # 初始化正确 ⇒ logits≈0 ⇒ 均匀分布 ⇒ loss ≈ ln V。差得远说明 init 或 loss 写错了
            assert abs(loss - math.log(vocab_size)) < 0.5, f"初始 loss {loss:.3f} ≠ ln V"
        if step == 1 or step % args.eval_interval == 0 or step == args.max_iters:
            val_loss = estimate_val_loss(W, val_data, rng)
            print(f"step {step:5d} | train_loss {loss:.4f} | val_loss {val_loss:.4f} | "
                  f"lr {lr:.2e} | {time.time() - t0:.1f}s")

    save_ckpt(W, config, args.out)
    print(f"\ncheckpoint saved to {args.out}")


if __name__ == "__main__":
    main()

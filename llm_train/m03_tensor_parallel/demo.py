"""
M03 — 张量并行 (Megatron-style TP)

是什么: 把一层的权重矩阵切给 N 个 rank。MLP 的标准切法: W1 按列切, W2 按行切。
解决的瓶颈: 单层参数/激活放不下一张卡。代价是 **每层** 前向 1 次 + 反向 1 次 all-reduce,
           阻塞在关键路径上, 所以 TP 只在 NVLink 机内用 (通常 ≤ 8)。
关键公式: 前向  Y = Σ_r relu(X·W1_r)·W2_r          ← g 算子: 前向 all-reduce, 反向恒等
          反向  dX = Σ_r dZ_r · W1_rᵀ               ← f 算子: 前向恒等, 反向 all-reduce
读代码盯住: 两次 `all_reduce_sum` 的位置; relu 夹在列切和行切之间不需要通信 (逐元素)。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_reduce_sum, banner, comm, kv, max_abs_diff, relu


def dense_mlp_grads(x, target, W1, b1, W2, b2):
    z = x @ W1 + b1
    h = relu(z)
    out = h @ W2 + b2
    diff = out - target
    loss = float(np.mean(diff * diff))
    d_out = (2.0 / diff.size) * diff
    d_z = (d_out @ W2.T) * (z > 0)
    grads = {
        "W2": h.T @ d_out, "b2": d_out.sum(0),
        "W1": x.T @ d_z, "b1": d_z.sum(0),
        "x": d_z @ W1.T,                      # 传给上一层的梯度 —— TP 的反向通信就为它
    }
    return loss, grads, out


def main() -> None:
    banner("M03 - Tensor Parallel MLP")

    rs = np.random.RandomState(4)
    B, D, H, O, world = 3, 4, 8, 2, 2
    x = rs.randn(B, D).astype(np.float32)
    target = rs.randn(B, O).astype(np.float32)
    W1 = (rs.randn(D, H) * 0.1).astype(np.float32)
    b1 = (rs.randn(H) * 0.1).astype(np.float32)
    W2 = (rs.randn(H, O) * 0.1).astype(np.float32)
    b2 = np.zeros(O, dtype=np.float32)

    dense_loss, dense, dense_out = dense_mlp_grads(x, target, W1, b1, W2, b2)

    W1_s = np.split(W1, world, axis=1)        # 列切: [D, H] -> N × [D, H/N]
    b1_s = np.split(b1, world)                #        [H]    -> N × [H/N]
    W2_s = np.split(W2, world, axis=0)        # 行切: [H, O] -> N × [H/N, O]

    comm.reset()
    # ---- 前向: x 在每个 rank 上是完整副本 (f 算子前向 = 恒等) ----
    z_s = [x @ w + b for w, b in zip(W1_s, b1_s)]            # N × [B, H/N]
    h_s = [relu(z) for z in z_s]                             # 逐元素, 无需通信
    partial = [h @ w for h, w in zip(h_s, W2_s)]             # N × [B, O], 每份只是部分和
    tp_out = all_reduce_sum(partial)[0] + b2                 # g 算子; b2 只加一次, 不能每 rank 都加

    # ---- 反向: d_out 在每个 rank 上相同 (g 算子反向 = 恒等) ----
    diff = tp_out - target
    tp_loss = float(np.mean(diff * diff))
    d_out = (2.0 / diff.size) * diff                         # [B, O]
    gW2_s = [h.T @ d_out for h in h_s]                       # N × [H/N, O]
    d_z_s = [(d_out @ w.T) * (z > 0) for w, z in zip(W2_s, z_s)]   # N × [B, H/N]
    gW1_s = [x.T @ dz for dz in d_z_s]                       # N × [D, H/N]
    gb1_s = [dz.sum(0) for dz in d_z_s]
    dx_partial = [dz @ w.T for dz, w in zip(d_z_s, W1_s)]    # N × [B, D], 每份只含本 rank 那 H/N 列的贡献
    tp_dx = all_reduce_sum(dx_partial)[0]                    # f 算子反向: 不做这一步, 上一层拿到的梯度就是错的

    tp = {
        "W1": np.concatenate(gW1_s, axis=1), "b1": np.concatenate(gb1_s),
        "W2": np.concatenate(gW2_s, axis=0), "b2": d_out.sum(0), "x": tp_dx,
    }

    kv("dense / tp loss", f"{dense_loss:.6f} / {tp_loss:.6f}")
    kv("max output diff", f"{max_abs_diff(dense_out, tp_out):.2e}")
    for name in ("W1", "b1", "W2", "b2", "x"):
        kv(f"max grad {name} diff", f"{max_abs_diff(dense[name], tp[name]):.2e}")
    kv("单 rank 的 dX 与真值差", f"{max_abs_diff(dense['x'], dx_partial[0]):.2e}  (未 all-reduce → 错)")
    kv("每 rank 权重", f"{W1_s[0].size + W2_s[0].size} / {W1.size + W2.size} 个参数")
    kv("通信量 (1 层 fwd+bwd)", comm.summary())

    assert max_abs_diff(dense_out, tp_out) < 1e-6
    assert all(max_abs_diff(dense[k], tp[k]) < 1e-6 for k in dense)
    assert max_abs_diff(dense["x"], dx_partial[0]) > 1e-4, "局部 dX 不等于真 dX, 反向 all-reduce 不可省"
    assert comm.calls["all_reduce"] == 2
    print("\n  OK: 输出、全部参数梯度、dX 都与 dense 一致; 每个 MLP 块 = 前向 1 次 + 反向 1 次 all-reduce。")


if __name__ == "__main__":
    main()

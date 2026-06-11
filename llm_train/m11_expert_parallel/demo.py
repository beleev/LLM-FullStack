"""
M11 — Expert Parallelism (MoE 专家并行)

MoE 模型 (Mixtral / DeepSeek-V3) 的参数大头在专家 FFN 上, 单卡放不下所有专家。
EP 把 E 个专家切到 D 张卡上 (每卡 E/D 个), 于是出现一种新的通信模式:

    token 在哪张卡  ≠  它选中的专家在哪张卡
    → all-to-all #1 (dispatch): 把每个 token 发到它的专家所在的卡
    → 本地专家计算 (每卡只算自己持有的专家)
    → all-to-all #2 (combine):  把结果按原路寄回, 按 gate 权重加权求和

与 DDP 的 all-reduce 不同, all-to-all 的通信量取决于 **路由结果**:
路由越不均衡, 热点卡越慢 (落后者效应), 还会触发容量溢出丢 token。
所以 MoE 训练必须带 load-balancing 辅助损失 (Switch/Mixtral 的 aux loss,
DeepSeek-V3 用 aux-loss-free 的 bias 调节, 思想相同: 把路由"推平")。

说明: 本 demo 用 list 模拟 D 张卡, all_to_all 是纯 numpy 数组重排。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import all_to_all, banner, kv, set_seed


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def expert_forward(weights: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    """每个专家是一个独立的 2 层 MLP: relu(x W1) W2。"""
    w1, w2 = weights
    return np.maximum(x @ w1, 0.0) @ w2


def route(x: np.ndarray, router_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """top-1 (Switch 式) 路由: 返回 (选中专家 id, gate 权重, 全量概率)。"""
    probs = softmax(x @ router_w)        # [N, E]
    expert_of = probs.argmax(axis=1)     # [N]
    gate = probs[np.arange(len(x)), expert_of]
    return expert_of, gate, probs


def moe_forward_ep(
    x: np.ndarray,
    expert_of: np.ndarray,
    gate: np.ndarray,
    experts: list[tuple[np.ndarray, np.ndarray]],
    world: int,
    capacity: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    专家并行版 MoE 前向: dispatch all-to-all → 本地专家计算 → combine all-to-all。

    token 按行号均分在 D 张卡上; 专家按 id 均分在 D 张卡上 (expert e 在卡 e // (E/D))。
    capacity 不为 None 时, 每个专家最多接收 capacity 个 token, 溢出的 token
    直接走残差 (输出 0), 这就是 "token dropping"。
    """
    n_tokens, e_total = len(x), len(experts)
    per_dev_experts = e_total // world
    token_dev = np.arange(n_tokens) // (n_tokens // world)   # token 所在卡
    expert_dev = expert_of // per_dev_experts                # token 目标卡

    # ---- all-to-all #1: dispatch ----
    # shards[src][dst] = (token 行号, token 向量) — 行号随行李一起寄, 回程要用
    idx_shards = [
        [np.where((token_dev == src) & (expert_dev == dst))[0] for dst in range(world)]
        for src in range(world)
    ]
    tok_shards = [[x[idx] for idx in row] for row in idx_shards]
    recv_idx = all_to_all(idx_shards)    # 每张卡收到: 来自各卡的 token 行号
    recv_tok = all_to_all(tok_shards)    # 与之对齐的 token 向量

    dispatch_matrix = np.array(
        [[len(idx_shards[s][d]) for d in range(world)] for s in range(world)]
    )

    # ---- 本地专家计算 (每张卡只持有自己的专家权重) ----
    out = np.zeros_like(x)
    dropped = 0
    for dev in range(world):
        ids = np.concatenate(recv_idx[dev]).astype(int)
        toks = np.concatenate(recv_tok[dev])   # 与 ids 按同样的 src 顺序拼接, 行行对齐
        for local_e in range(per_dev_experts):
            e = dev * per_dev_experts + local_e
            sel = expert_of[ids] == e
            e_ids, e_toks = ids[sel], toks[sel]
            if capacity is not None and len(e_ids) > capacity:
                dropped += len(e_ids) - capacity
                e_ids, e_toks = e_ids[:capacity], e_toks[:capacity]   # 溢出丢弃
            if len(e_ids):
                # all-to-all #2 (combine): 教学上直接按行号写回全局输出
                out[e_ids] = expert_forward(experts[e], e_toks) * gate[e_ids, None]

    stats = {"dispatch": dispatch_matrix, "dropped": dropped}
    return out, stats


def balance_router(x: np.ndarray, router_w: np.ndarray, steps: int, lr: float) -> np.ndarray:
    """
    用 Switch 风格的 load-balancing aux loss 训练 router:
        L_aux = E · Σ_e f_e · P_e
    f_e: 派给专家 e 的 token 比例 (不可导, 视为常数);  P_e: 平均路由概率 (可导)。
    f 与 P 同向, 最小化点积会把"热门专家"的概率压下去 → 路由变平。
    """
    e_total = router_w.shape[1]
    w = router_w.copy()
    for _ in range(steps):
        probs = softmax(x @ w)                       # [N, E]
        f = np.bincount(probs.argmax(1), minlength=e_total) / len(x)
        # dL/d logits = E/N · (diag(p) - p p^T) f   (softmax 的 Jacobian 乘 f)
        grad_p = np.tile(f, (len(x), 1)) * e_total / len(x)
        grad_logits = probs * (grad_p - (probs * grad_p).sum(1, keepdims=True))
        w -= lr * x.T @ grad_logits
    return w


def main() -> None:
    banner("M11 - Expert Parallelism (MoE all-to-all)")

    rs = set_seed(7)
    world, e_total, n_tokens, d, d_ff = 4, 8, 64, 8, 16
    x = rs.randn(n_tokens, d).astype(np.float64)
    experts = [
        (rs.randn(d, d_ff) * 0.3, rs.randn(d_ff, d) * 0.3) for _ in range(e_total)
    ]
    # 刻意制造倾斜的 router: 真实训练初期 / 数据分布漂移时就是这副样子
    router_w = rs.randn(d, e_total) + np.linspace(1.2, -1.2, e_total)[None, :]

    expert_of, gate, _ = route(x, router_w)

    # ---- 1) dispatch 计划: 谁给谁发多少 token ----
    out_ep, stats = moe_forward_ep(x, expert_of, gate, experts, world)
    print(f"\n[1] all-to-all 发送矩阵 (行=源卡, 列=目标卡, {world} 卡 × 每卡 {e_total // world} 专家)")
    for row in stats["dispatch"]:
        print("    " + "  ".join(f"{v:>3}" for v in row))

    # ---- 2) 正确性: EP 输出 == 单卡稠密计算 ----
    dense = np.zeros_like(x)
    for e in range(e_total):
        sel = expert_of == e
        if sel.any():
            dense[sel] = expert_forward(experts[e], x[sel]) * gate[sel, None]
    print("\n[2] 正确性")
    kv("max |dense - EP|", f"{np.abs(dense - out_ep).max():.2e}")
    assert np.allclose(dense, out_ep), "EP 前向必须与单卡稠密计算一致"

    # ---- 3) 负载不均衡 → 容量溢出丢 token ----
    counts = np.bincount(expert_of, minlength=e_total)
    capacity = int(np.ceil(n_tokens / e_total * 1.25))   # capacity factor 1.25
    _, stats_cap = moe_forward_ep(x, expert_of, gate, experts, world, capacity=capacity)
    print("\n[3] 倾斜路由下的负载")
    kv("每个专家收到的 token 数", counts.tolist())
    kv("负载不均衡度 max/mean", f"{counts.max() / counts.mean():.2f}x")
    kv(f"capacity={capacity} 时丢弃 token", f"{stats_cap['dropped']} / {n_tokens}")

    # ---- 4) aux loss 把路由推平 ----
    router_balanced = balance_router(x, router_w, steps=60, lr=0.5)
    expert_of2, gate2, _ = route(x, router_balanced)
    counts2 = np.bincount(expert_of2, minlength=e_total)
    _, stats2 = moe_forward_ep(x, expert_of2, gate2, experts, world, capacity=capacity)
    print("\n[4] 用 load-balancing aux loss 训练 router 60 步后")
    kv("每个专家收到的 token 数", counts2.tolist())
    kv("负载不均衡度 max/mean", f"{counts2.max() / counts2.mean():.2f}x")
    kv("丢弃 token", f"{stats2['dropped']} / {n_tokens}")

    assert counts2.max() / counts2.mean() < counts.max() / counts.mean()
    print("\n  OK: EP 的代价是两次 all-to-all + 路由均衡问题;")
    print("      aux loss (或 DeepSeek-V3 的 bias 调节) 把热点专家压平, 丢 token 减少。")


if __name__ == "__main__":
    main()

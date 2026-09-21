"""
m12 demo — CUDA Graph capture / replay (**模拟**: launch 开销是人为 busy-wait 注入的, 见 graph.py)。

[1] replay 输出 == eager 输出; host 提交次数 16 → 1
[2] 扫 LAUNCH_OVERHEAD_US ∈ {0, 10, 50, 200}: 开销为 0 时加速比塌到 ~1.3×, 证明收益只来自省掉的 launch
[3] 静态 buffer 语义: 重新绑定输入变量 (bug) vs 拷进 static_input (正确); 输出 buffer 会被下次 replay 覆盖
[4] batch size 分桶 [1,2,4,8]: B=3 padding 到 4, 输出切片后 == eager; B=9 回退 eager

运行: python -m llm_infer.m12_cuda_graph.demo
"""
from __future__ import annotations
from functools import partial
import numpy as np

from llm_infer.core.utils import banner, kv, Timer
from llm_infer.m12_cuda_graph.graph import FakeGPU, CudaGraph, BucketedGraphRunner, forward, N_LAYER

LAUNCH_OVERHEAD_US_SWEEP = [0, 10, 50, 200]   # 模拟参数; 真实 GPU 上每个 kernel 的 host 端开销约 10 µs 量级 (含框架 dispatcher 更高)
N_STEPS = 100
OPS_PER_STEP = 4 * N_LAYER


def us_per_step(step) -> float:
    """3 轮取最快, 压掉系统抖动。"""
    best = float("inf")
    for _ in range(3):
        with Timer() as t:
            for _ in range(N_STEPS):
                step()
        best = min(best, t.ms * 1e3 / N_STEPS)
    return best


def main():
    banner("M12 - CUDA Graph capture / replay (模拟)")
    print("  注意: 本机没有 GPU。'launch 开销' 是 FakeGPU 里 busy-wait 注入的, 下面所有加速比都是这个模型的产物。")

    rs = np.random.RandomState(0)
    B, D, H = 4, 64, 128
    W1 = rs.randn(D, H).astype(np.float32) * 0.05
    W2 = rs.randn(H, D).astype(np.float32) * 0.05
    fwd = partial(forward, W1=W1, W2=W2)
    x = rs.randn(B, D).astype(np.float32)

    print("\n[1] 正确性 + host 提交次数")
    gpu = FakeGPU(launch_overhead_us=0)
    out_eager = fwd(gpu, x)
    n_eager = gpu.n_launch
    graph = CudaGraph(gpu, fwd, np.zeros_like(x))        # 用全 0 的样例输入 capture: 录的是地址, 样例数值无所谓
    gpu.n_launch = 0
    out_replay = graph.replay(x)
    diff = float(np.abs(out_eager - out_replay).max())
    kv("host 提交次数 eager / replay", f"{n_eager} / {gpu.n_launch}")
    kv("max |eager - replay|", f"{diff:.2e}")
    assert (n_eager, gpu.n_launch) == (OPS_PER_STEP, 1) and len(graph.ops) == OPS_PER_STEP
    assert diff == 0.0                                   # 同样的 numpy kernel、同样的顺序 → 逐位相同

    print(f"\n[2] 扫 launch 开销 (每格 {N_STEPS} 步 ×3 轮取最快, B={B}, 每步 {OPS_PER_STEP} 个 kernel)")
    print(f"  {'LAUNCH_OVERHEAD_US':>18} {'eager µs/步':>12} {'replay µs/步':>13} {'加速比':>7} {'模型预测':>8}")
    speedup = {}
    for us in LAUNCH_OVERHEAD_US_SWEEP:
        gpu = FakeGPU(us)
        graph = CudaGraph(gpu, fwd, x)
        e, r = us_per_step(lambda: fwd(gpu, x)), us_per_step(lambda: graph.replay(x))
        if us == 0:
            c_e, c_r = e, r                              # 开销为 0 时测到的就是纯计算 (+Python) 时间
        pred = (c_e + OPS_PER_STEP * us) / (c_r + us)    # eager 付 16 次, replay 付 1 次
        speedup[us] = e / r
        print(f"  {us:>18} {e:>12.1f} {r:>13.1f} {e / r:>6.1f}x {pred:>7.1f}x")
    print("  ↑ 开销=0 时剩下的一点差距来自 eager 每步重新分配 4 个中间 buffer + Python 包装, 与 'CUDA graph' 无关。")
    assert speedup[0] < 2.0, speedup                     # 没有 launch 开销 → 没有数量级收益
    assert speedup[50] > 3.0 and speedup[200] > speedup[50] > speedup[10]

    print("\n[3] 静态 buffer 语义")
    gpu = FakeGPU(0)
    x_old, x_new = x, rs.randn(B, D).astype(np.float32)
    ref_old, ref_new = fwd(gpu, x_old).copy(), fwd(gpu, x_new).copy()
    graph = CudaGraph(gpu, fwd, x_old)
    out = graph.replay_buggy(x_new)                      # bug: static_input = x_new 只是换了名字的指向
    kv("bug 版 |out - eager(新输入)|", f"{np.abs(out - ref_new).max():.3f}  ← 错")
    kv("bug 版 |out - eager(旧输入)|", f"{np.abs(out - ref_old).max():.1e}  ← 算的还是旧输入")
    assert np.abs(out - ref_new).max() > 0.1 and np.array_equal(out, ref_old)

    graph = CudaGraph(gpu, fwd, x_old)
    out1 = graph.replay(x_new)                           # 正确: 拷进 static_input
    assert np.array_equal(out1, ref_new)
    x_new[:] = 0                                         # replay 之后改调用方的数组, 不影响图 (图读的是自己的 buffer)
    out2 = graph.replay(x_old)
    kv("两次 replay 返回同一块内存", np.shares_memory(out1, out2))
    assert out1 is out2 and np.array_equal(out1, ref_old)   # out1 已被第二次 replay 覆盖 — 要留结果必须 .copy()

    print("\n[4] batch size 分桶 + padding")
    gpu = FakeGPU(0)
    runner = BucketedGraphRunner(gpu, fwd, D, buckets=[1, 2, 4, 8])
    for b in (1, 3, 5, 8, 9):
        xb = rs.randn(b, D).astype(np.float32)
        ref = fwd(gpu, xb).copy()
        gpu.n_launch = 0
        out = runner.run(xb)
        bucket = next((k for k in runner.graphs if k >= b), None)
        how = f"桶 {bucket}, padding {bucket - b} 行" if bucket else "超出最大桶 → eager"
        print(f"  B={b}: {how:<24} host 提交 {gpu.n_launch:>2} 次, 输出 {out.shape}, max-abs-diff {np.abs(out - ref).max():.1e}")
        assert out.shape == (b, D) and np.allclose(out, ref, atol=1e-5)   # 不同 B 下 BLAS 分块不同, 允许 1e-5
        assert gpu.n_launch == (1 if bucket else OPS_PER_STEP)


if __name__ == "__main__":
    main()

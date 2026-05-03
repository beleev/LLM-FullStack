"""
m12 demo — CUDA Graph capture / replay 思想模拟

我们用 Python 模拟"control flow 开销 vs replay 开销":
- 每个 op 在 eager 里都要走 Python 解释器 + 函数调用 + 参数检查 (假装的 launch overhead)
- replay 时所有 op 已被记录成一个 list, 直接跑一遍数值, 跳过 Python 控制流

实际 CUDA graph 的 replay 是 GPU 端串好的 dag, host 端零参与;
这里我们用纯 Python 演示这个"省去 control flow"的思想。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np

from llm_infer.core.utils import banner, kv, Timer


# --------------------------------------------------------------------- #
# 假装的"GPU op" — 调用前有固定 launch 开销 (模拟)                       #
# --------------------------------------------------------------------- #

LAUNCH_OVERHEAD_US = 50           # 每 op "调度开销" 50 微秒


def _simulate_launch():
    # 真实 launch ~10us; 这里放大到 50us 让对比明显, 又不至于太慢
    time.sleep(LAUNCH_OVERHEAD_US / 1e6)


def gpu_matmul(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    _simulate_launch()
    np.matmul(a, b, out=out)


def gpu_add(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    _simulate_launch()
    np.add(a, b, out=out)


def gpu_relu(x: np.ndarray, out: np.ndarray) -> None:
    _simulate_launch()
    np.maximum(x, 0, out=out)


# --------------------------------------------------------------------- #
# 一个"模型 forward" — 故意做很多小 op, 模拟 LLaMA decode 的 kernel 流  #
# --------------------------------------------------------------------- #

def eager_forward(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """每个 op 都走完整 Python 控制流 + 假 launch overhead。"""
    out_buf1 = np.empty((x.shape[0], W1.shape[1]), dtype=np.float32)
    out_buf2 = np.empty_like(out_buf1)
    out_buf3 = np.empty((x.shape[0], W2.shape[1]), dtype=np.float32)
    out_final = np.empty_like(out_buf3)

    gpu_matmul(x, W1, out_buf1)         # x @ W1
    gpu_relu(out_buf1, out_buf2)        # relu
    gpu_matmul(out_buf2, W2, out_buf3)  # @ W2
    gpu_add(out_buf3, x[:, :W2.shape[1]], out_final)  # 残差 (维度不对就截一下)

    # 多 layer 模拟: 重复 4 次
    for _ in range(3):
        gpu_matmul(out_final, W1, out_buf1)
        gpu_relu(out_buf1, out_buf2)
        gpu_matmul(out_buf2, W2, out_buf3)
        gpu_add(out_buf3, out_final, out_final)

    return out_final


# --------------------------------------------------------------------- #
# Graph capture / replay                                                #
# --------------------------------------------------------------------- #

@dataclass
class CapturedOp:
    """记录一个 op 的"形状", replay 时复用 buffer 引用。"""
    fn: Callable
    inputs: list                # list of np.ndarray (引用, 不复制)
    output: np.ndarray


@dataclass
class CapturedGraph:
    ops: List[CapturedOp] = field(default_factory=list)
    input_buffer: np.ndarray = None
    output_buffer: np.ndarray = None

    def replay(self, new_input: np.ndarray) -> np.ndarray:
        """把新输入写到 input_buffer, 按记录跑所有 op (跳过 Python 控制流)。"""
        np.copyto(self.input_buffer, new_input)
        for op in self.ops:
            # 注意: 我们不再走 gpu_xxx 的 wrapper (那有 _simulate_launch),
            # 因为 graph 已经把 dag 提交给 "GPU"; 这里直接调底层 numpy。
            # 真实 CUDA graph: GPU 端 dag 调度, host 不参与。
            if op.fn is gpu_matmul:
                np.matmul(op.inputs[0], op.inputs[1], out=op.output)
            elif op.fn is gpu_add:
                np.add(op.inputs[0], op.inputs[1], out=op.output)
            elif op.fn is gpu_relu:
                np.maximum(op.inputs[0], 0, out=op.output)
        return self.output_buffer


def capture_forward(x_template: np.ndarray, W1, W2) -> CapturedGraph:
    """把 forward 跑一次, 同时记录 op 序列。"""
    g = CapturedGraph(input_buffer=x_template.copy())

    out_buf1 = np.empty((x_template.shape[0], W1.shape[1]), dtype=np.float32)
    out_buf2 = np.empty_like(out_buf1)
    out_buf3 = np.empty((x_template.shape[0], W2.shape[1]), dtype=np.float32)
    out_final = np.empty_like(out_buf3)

    def rec(fn, inputs, output):
        fn(*inputs, output)             # 真跑一次, 让 buffer 有值
        g.ops.append(CapturedOp(fn=fn, inputs=list(inputs), output=output))

    rec(gpu_matmul, (g.input_buffer, W1), out_buf1)
    rec(gpu_relu,   (out_buf1,),         out_buf2)
    rec(gpu_matmul, (out_buf2, W2),      out_buf3)
    rec(gpu_add,    (out_buf3, g.input_buffer[:, :W2.shape[1]]), out_final)
    for _ in range(3):
        rec(gpu_matmul, (out_final, W1), out_buf1)
        rec(gpu_relu,   (out_buf1,),     out_buf2)
        rec(gpu_matmul, (out_buf2, W2),  out_buf3)
        rec(gpu_add,    (out_buf3, out_final), out_final)

    g.output_buffer = out_final
    return g


# --------------------------------------------------------------------- #
# main                                                                  #
# --------------------------------------------------------------------- #

def main():
    banner("M12 - CUDA Graph capture / replay (Python 模拟)")

    rs = np.random.RandomState(0)
    B, D = 4, 64
    W1 = rs.randn(D, D).astype(np.float32) * 0.05
    W2 = rs.randn(D, D).astype(np.float32) * 0.05
    x = rs.randn(B, D).astype(np.float32)

    print(f"\n[1] eager forward: 每 op 都付 launch_overhead={LAUNCH_OVERHEAD_US}us")
    N = 50
    with Timer() as t_eager:
        for _ in range(N):
            out_e = eager_forward(x, W1, W2)
    kv(f"eager × {N} 次", f"{t_eager.ms:.1f} ms")

    print("\n[2] capture 一次, 后续 replay")
    with Timer() as t_capture:
        graph = capture_forward(x, W1, W2)
    kv("capture 一次", f"{t_capture.ms:.1f} ms (≈ 1 次 eager)")

    with Timer() as t_replay:
        for _ in range(N):
            out_g = graph.replay(x)
    kv(f"replay × {N} 次", f"{t_replay.ms:.1f} ms")

    print("\n[3] 数值正确性")
    diff = np.max(np.abs(out_e - out_g))
    kv("max |eager - replay|", f"{diff:.2e}")
    assert diff < 1e-5

    print("\n[4] 对比")
    kv("加速比", f"{t_eager.ms / t_replay.ms:.1f}x")
    kv("每步 launch 节省", f"~{LAUNCH_OVERHEAD_US * 16}us / step (16 ops)")
    print("\n  ✓ 真实 vLLM/SGLang 在 decode 阶段几乎全靠 graph, ")
    print("    H100 上 batch=8 decode latency 能从 ~5ms 降到 ~1.5ms。")
    print("  ⚠ 限制: capture 时形状必须固定, 所以要为多 batch_size 分别 capture。")


if __name__ == "__main__":
    main()

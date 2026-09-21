"""
graph.py — CUDA Graph 的 capture / replay 语义, 用 numpy + 一个**人为注入的 launch 开销**来模拟。

是什么: 把一次 forward 的全部 kernel 调用录下来, 之后 host 只发 1 次 "replay", 不再逐个 launch。
瓶颈: decode 每步算得很少, GPU kernel 本身几十 µs, 而 host 端每次 launch (Python → 框架 dispatcher → 驱动)
      也要 ~10 µs 量级; 一步几百个 kernel → 延迟被 host 开销主导, GPU 在空等。
**这是模拟**: 没有 GPU。`FakeGPU.launch` 里用 busy-wait 烧掉 `launch_overhead_us` 微秒来充当 host 开销,
      demo 的加速比完全来自这个参数 (设为 0 时实测加速比从 ~12× 塌到 ~1.3×, 剩下的 0.3 来自 eager 每步重新分配 buffer)。
读代码盯住: `static_input` / `static_output` — 图里录的是 buffer **地址**, replay 只认地址不认变量名。
真实系统: `torch.cuda.CUDAGraph`; vLLM `CUDAGraphRunner` / SGLang `CudaGraphRunner` (按 batch size 分桶 capture);
      TensorRT-LLM 同样在 decode 阶段用 CUDA graph。
"""
from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional
import numpy as np


class FakeGPU:
    """模拟 "host 提交 kernel" 这件事: 每次提交烧掉 launch_overhead_us 微秒 (busy-wait, 比 sleep 准)。"""

    def __init__(self, launch_overhead_us: float):
        self.launch_overhead_us = launch_overhead_us
        self.n_launch = 0                      # host 端提交次数: eager 每 op 1 次, graph 每次 replay 1 次
        self.tape: Optional[list] = None       # 非 None 表示正在 capture

    def _host_overhead(self) -> None:
        self.n_launch += 1
        if not self.launch_overhead_us:
            return
        end = time.perf_counter() + self.launch_overhead_us * 1e-6
        while time.perf_counter() < end:
            pass

    def launch(self, fn: Callable, *args, out: np.ndarray) -> None:
        """eager 路径: 付一次 host 开销, 再执行 kernel。capture 期间顺便把 (fn, args, out) 录到 tape。"""
        self._host_overhead()
        fn(*args, out=out)
        if self.tape is not None:
            self.tape.append((fn, args, out))  # 录的是数组对象本身 (= 地址), 不是数值

    def launch_graph(self, ops: list) -> None:
        """replay 路径: 整张图只付 1 次 host 开销; 图内各 kernel 由 "GPU" 自己连续执行。"""
        self._host_overhead()
        for fn, args, out in ops:
            fn(*args, out=out)


N_LAYER = 4   # 每层 4 个 kernel (matmul, relu, matmul, add) → 一次 forward 16 次 launch


def forward(gpu: FakeGPU, x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """4 层残差 MLP: y = x + relu(x·W1)·W2。eager 与 capture 共用这一份代码 (和 PyTorch 一样: capture 就是"录着跑一遍")。

    x (B, D), W1 (D, H), W2 (H, D) → (B, D)。各行互不相关, 所以 padding 行不影响真实行。
    """
    B, (D, H) = x.shape[0], W1.shape
    h1, h2 = np.empty((B, H), np.float32), np.empty((B, H), np.float32)
    h, y = np.empty((B, D), np.float32), np.empty((B, D), np.float32)
    src = x
    for _ in range(N_LAYER):
        gpu.launch(np.matmul, src, W1, out=h1)      # (B, D) @ (D, H) → (B, H)
        gpu.launch(np.maximum, h1, 0, out=h2)       # relu
        gpu.launch(np.matmul, h2, W2, out=h)        # (B, H) @ (H, D) → (B, D)
        gpu.launch(np.add, h, src, out=y)           # 残差; 第 1 层读 x 写 y, 之后 y 原地累加 — 不能写回 x, 那会污染输入 buffer
        src = y
    return y


class CudaGraph:
    """capture 一次, 之后 replay。形状在 capture 时定死 (static_input.shape)。"""

    def __init__(self, gpu: FakeGPU, fn: Callable[[FakeGPU, np.ndarray], np.ndarray], example_input: np.ndarray):
        self.gpu = gpu
        self.static_input = example_input.copy()    # 图私有的输入 buffer, 地址从此固定
        gpu.tape = []
        self.static_output = fn(gpu, self.static_input)
        self.ops, gpu.tape = gpu.tape, None

    def replay(self, x: np.ndarray) -> np.ndarray:
        self.static_input[...] = x                  # 必须**拷进**静态 buffer; 返回值是 static_output 的引用, 下次 replay 会被覆盖
        self.gpu.launch_graph(self.ops)
        return self.static_output

    def replay_buggy(self, x: np.ndarray) -> np.ndarray:
        """经典 bug: 只是把名字 static_input 重新绑定到 x, 图里录的旧地址没变 → 算的还是旧输入。"""
        self.static_input = x
        self.gpu.launch_graph(self.ops)
        return self.static_output


class BucketedGraphRunner:
    """为一组 batch size 各 capture 一张图; 运行时向上取最近的桶, 不足的行 padding, 输出切片。"""

    def __init__(self, gpu: FakeGPU, fn, d_model: int, buckets: List[int]):
        self.gpu, self.fn = gpu, fn
        self.graphs: Dict[int, CudaGraph] = {
            b: CudaGraph(gpu, fn, np.zeros((b, d_model), np.float32)) for b in sorted(buckets)
        }

    def run(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        bucket = next((b for b in self.graphs if b >= B), None)
        if bucket is None:
            return self.fn(self.gpu, x)             # 超过最大桶 → 回退 eager (vLLM 同样如此)
        g = self.graphs[bucket]
        g.static_input[:B] = x                      # padding 行留着上次的旧值也无妨: 各行独立, 且输出会被切掉
        self.gpu.launch_graph(g.ops)
        return g.static_output[:B]                  # (B, D) 视图

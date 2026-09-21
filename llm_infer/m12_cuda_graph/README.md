# M12 — CUDA Graph: 把每步几百次 kernel launch 压成 1 次 (模拟)

> **本模块是模拟。** 没有 GPU; "launch 开销" 是 `FakeGPU` 里 busy-wait 人为注入的参数
> `launch_overhead_us`, demo 里的加速比完全是这个模型的产物。模块要教的是 capture/replay 的**语义**
> (静态 buffer、固定形状、分桶 padding), 不是性能数字。

## 直觉
decode 每步只算 1 个 token, 单个 kernel 在 GPU 上几十 µs 就跑完, 但 host 每提交一个 kernel
(Python → 框架 dispatcher → 驱动) 也要 ~10 µs 量级。一步几百个 kernel → GPU 大部分时间在等 host。
CUDA Graph: 把一次 forward 提交的 kernel 序列 (含参数和**显存地址**) 录下来, 之后 host 每步只发 1 次 replay。

## 核心数据结构或公式
```
CudaGraph.ops           = [(kernel, 输入数组们, 输出数组), ...]   录的是数组对象 (地址), 不是数值
CudaGraph.static_input   capture 时分配, 地址固定; replay 前必须 static_input[...] = x  (拷贝, 不是赋值)
CudaGraph.static_output  每次 replay 原地覆盖; 要保留结果必须 .copy()
```
- 开销模型: `eager = c + n_ops · o`, `replay = c' + 1 · o` (o = launch_overhead_us, c = 纯计算)。
  o → 0 时加速比 → c/c' ≈ 1; o ≫ c 时加速比 → n_ops (本 demo 16)。
- 分桶: 形状在 capture 时定死, 所以为 batch size `[1,2,4,8]` 各录一张图; B=3 → 取桶 4,
  `static_input[:3] = x`, replay, 返回 `static_output[:3]`; B 超过最大桶 → 回退 eager。

## 运行后应该看到什么
```bash
python -m llm_infer.m12_cuda_graph.demo      # ~1.5 s
```
```
[1] host 提交次数 eager / replay = 16 / 1;  max |eager - replay| = 0.00e+00
[2] LAUNCH_OVERHEAD_US   eager µs/步  replay µs/步   加速比   模型预测
                     0        24.5         18.9     1.3x     1.3x
                    10       193.9         30.5     6.4x     6.4x
                    50       855.7         70.1    12.2x    12.0x
                   200      3303.5        219.9    15.0x    14.7x
[3] bug 版 (重新绑定输入): |out - eager(新输入)| = 3.705, |out - eager(旧输入)| = 0 ← 算的还是旧输入
    两次 replay 返回同一块内存 = True
[4] B=3: 桶 4, padding 1 行, host 提交 1 次, 输出 (3, 64), max-abs-diff 0
    B=9: 超出最大桶 → eager, host 提交 16 次
```
断言: replay 与 eager 逐位相等; 提交次数 16 vs 1; 开销=0 时加速比 < 2 且随开销单调上升;
bug 版输出 == 旧输入的结果; 正确版在调用方改掉输入数组后仍不受影响; 两次 replay 返回同一对象;
分桶输出切片后 == eager。µs 数字随机器变化 (3 轮取最快), 0 开销那行的 1.3× 来自 eager 每步重新分配中间 buffer。

## 与真实系统的差距
- 真实 CUDA graph 在**驱动层**录制 GPU 命令流, replay 时 host 一次调用、GPU 端连续执行; 这里只是 Python
  list 里存 numpy 调用, "GPU 执行" 和 "host" 其实是同一个线程。
- 真实收益还包括省掉 Python / dispatcher / 显存分配器开销; 真实限制还包括: 图内不能有 host 同步、
  不能有依赖数值的控制流、不能动态分配显存 (用私有 memory pool)、每张图占一份激活显存。
- vLLM / SGLang 只对 **decode** 用 graph (prefill 形状多变且计算本身够大), 桶一般是 1,2,4,8,16,…,
  attention 的 KV 长度变化靠把 block table / seq_lens 也放进静态 buffer 解决 — 本 demo 的 MLP 没有这个问题。
- padding 行在真实系统里要小心: 要指向合法的 dummy KV 槽位, 否则 attention kernel 会越界读。

## 常见误区
- "CUDA graph 让 kernel 算得更快": 不会。kernel 一模一样, 省的只是 host 端提交开销 — [2] 的第 0 行就是证据。
- "replay 时把新 tensor 传进去就行": 图只认地址。`static_input = x` 是重新绑定名字, 图读的仍是旧 buffer ([3])。
- "replay 的返回值可以攒起来": 每次返回的是同一块 static_output, 下一次 replay 就被覆盖。
- "batch 大了收益更大": 相反。batch 越大单 kernel 计算越久, launch 开销占比越小, 所以大 batch / prefill 常回退 eager。

## 自测题
1. **一步 forward 有 300 个 kernel, 每个 launch 开销 10 µs, GPU 纯计算 2 ms。eager 和 graph 的单步延迟大约各是多少?**
   假设 host 串行提交且 GPU 要等: eager ≈ max(3 ms 提交, 2 ms 计算) ≈ 3 ms 起 (host-bound); graph ≈ 2 ms + 10 µs。
2. **为什么要分桶而不是为每个 batch size 都 capture 一张图?**
   每张图占 capture 时间和一份激活显存; 分桶用少量 padding 浪费换来 O(log B) 张图。
3. **B=3 用桶 4 时, 第 4 行 padding 是上次遗留的旧数据, 为什么输出仍然正确?**
   本模型各行独立 (逐行 matmul / relu / add), padding 行只影响被切掉的第 4 行输出; 若有跨行算子 (如 batch norm、共享 KV 的 attention) 就必须保证 padding 行无害。

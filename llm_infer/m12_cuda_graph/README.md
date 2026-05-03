# M12 — CUDA Graph: 把 launch 开销压到零

## 痛点
decode 阶段每步只算 1 个 token, **kernel 计算时间 < kernel launch 时间**:
- 一个 forward 串了几百个小 kernel (qkv proj, attention, mlp, norm...)
- 每个 launch 在 host 端开销 ~10us
- T=512 个 token 的 decode → 几百 × ~10us = 几 ms 纯 launch 浪费

## 思路
**捕获一次完整 forward 的所有 kernel + 它们的依赖 (DAG), 后续只 replay 这个图。**
GPU 直接按图调度, host 端零参与。
- capture: 真正跑一次, 记录所有 op
- replay: 把"输入张量"换掉, 复用图

## 限制
- 输入张量的**形状必须固定** (CUDA graph 不支持动态形状)
- 解决: 为常用 batch_size 各捕获一张图 (1, 2, 4, 8, ..., 256), replay 时挑最近的
- prefill 因 prompt 长度任意, 用 eager; decode 因 batch_size 有限, 用 graph

## 本模块如何"模拟" CUDA graph?
没有 GPU, 但可以用 Python 实现"trace + replay"的同样思路:
1. **trace 阶段**: 调一次 forward, 在每个 op 处记录 `(op_name, input_buffers, output_buffers)`
2. **replay 阶段**: 输入张量原地写入 (in-place), 按记录的顺序 replay op, 不再跑 Python 控制流
3. 对比 eager vs graph 的"控制流开销" (用人工 sleep 模拟 launch 开销)

## 运行
```bash
python -m llm_infer.m12_cuda_graph.demo
```

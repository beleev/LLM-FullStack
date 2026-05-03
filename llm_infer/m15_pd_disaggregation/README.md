# M15 — Prefill / Decode Disaggregation: 把两类负载分开

## 痛点
prefill 与 decode 是性质完全不同的负载:
| 维度 | prefill | decode |
|---|---|---|
| 计算 | compute-bound (T 个 token 一起) | memory-bound (1 token, 全权重要读一遍) |
| 显存 | 临时大 (attn 矩阵) | 持续大 (KV cache) |
| 时延 SLO | TTFT (用户看到第一个 token 的时间) | ITL (token 间隔) |
| GPU 适合 | 高算力 (H100, B100) | 高带宽 (H200, MI300) |

把它们**塞进同一个 batch / 同一张卡**会互相干扰:
- 长 prefill 阻塞 decode → 抖动
- decode batch 不愿等 prefill, 但又要复用相同权重

## 思路 (Splitwise / DistServe / Mooncake)
**两组节点**: prefill 节点专门跑 prefill, decode 节点专门跑 decode。
请求路径:
```
  user request
        │
        ▼
   ┌─────────────────┐         ┌─────────────────┐
   │ Prefill Cluster │  ───→   │ Decode Cluster  │
   │ - 高算力        │   KV    │ - 高带宽         │
   │ - prompt → KV   │ transfer│ - KV → tokens   │
   └─────────────────┘         └─────────────────┘
                                       │
                                       ▼
                                 streaming output
```

关键: **KV cache 跨节点传输**, 走 RDMA / NVLink。
延迟 ~ KV 大小 / 带宽; LLaMA-7B 4k context 约 2 GB, 200 GB/s NVLink → 10 ms,
可接受。

## 收益
- TTFT 与 ITL 解耦, 可独立 SLO 优化
- prefill 节点 batch 大, 算力打满
- decode 节点 batch 大, 带宽打满
- 整体吞吐 + TTFT/ITL 都更优 (vs 不分离)

## 本模块
单进程模拟两个"节点"对象 (`PrefillEngine`, `DecodeEngine`),
中间用一个"KV transport" 模拟传输 (Python 直接 copy)。
重点演示 **接口** 与 **数值正确性**: 分离后输出与单引擎一致。

## 运行
```bash
python -m llm_infer.m15_pd_disaggregation.demo
```

# M13 — Multi-LoRA Serving: 一个底模, N 个适配器同 batch 服务

## 场景
线上一个底模 (LLaMA-7B) 同时服务多个客户, 每个客户有自己的 LoRA adapter
(几 MB)。希望:
- 不为每个客户复制底模 (省显存)
- 不同客户的请求**同 batch 跑** (高吞吐)

## 朴素做法的问题
**串行**: 一次只服务一个客户, 切换 adapter, 吞吐惨淡。
**Batch 内同 adapter**: 把同 adapter 的请求攒一批, 但 idle 时间长 (谁的客户多就服务谁)。

## Punica / S-LoRA 思路 (本模块演示)
**Batch 内不同请求带不同 adapter**, 计算时把"基矩阵 + 各自 LoRA"融合:
```
  base:   y = x @ W                    所有请求共享
  LoRA:   y_i = x_i @ A_i @ B_i        每请求一对 (A_i, B_i)
  最终:   out_i = y + (alpha/r) * y_i

  关键: A 与 B 按"请求"维度分组, 用 segmented gemm (SGMV) 一次算完
```

## 简化实现 (本模块)
- 不实现真正的 SGMV kernel (那需要 Triton)
- 用 numpy "批量但显式循环 over 请求" 演示语义
- 重点展示**接口**: 一个 batch 中每条请求带 adapter id, forward 自动应用对应 LoRA

## 收益
- 一份底模显存, 服务任意多 adapter
- 同 batch 内 adapter 不同也能并行
- 真实 S-LoRA 一卡上 1000+ adapter 同时服务

## 运行
```bash
python -m llm_infer.m13_lora_serving.demo
```

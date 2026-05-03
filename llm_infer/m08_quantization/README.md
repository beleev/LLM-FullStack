# M08 — Quantization: 砍显存的核武器

两类量化, 各有侧重:

## A. 权重量化 (Weight-only quant)
**只量化模型权重, 激活仍 fp16/fp32; matmul 时 dequant 回 fp16 再算。**
- 收益: 模型体积 fp16→int8 砍半, fp16→int4 砍 4 倍
- 代价: 单步 dequant 开销 (但 decode 是 memory-bound, 加快显存读取反而更快)
- 算法: AWQ / GPTQ / SmoothQuant / RTN
- 本模块用最简单的 **per-channel symmetric INT8 RTN** (Round-To-Nearest)

## B. KV Cache 量化
**KV cache 在长 context 下比模型权重还大** (LLaMA-7B fp16, T=8k → 4 GB KV)
- 收益: KV 显存砍半 (INT8) 到 1/4 (INT4)
- 代价: 写入时 quant, 读取时 dequant; 误差影响 attention
- 算法: per-token / per-channel scale
- 本模块用 **per-token symmetric INT8** (每个 token 一个 scale, 分桶最简)

## 量化公式 (对称 INT8)
```
  scale = max(|x|) / 127
  q     = round(x / scale)              ∈ [-127, 127]
  x'    = q * scale                     反量化
```
误差: |x - x'| ≤ scale / 2

## 文件
- `int8_weight.py` — 权重 quantize / dequantize / 量化 matmul
- `kv_quant.py`    — KV cache per-token quantize / dequantize / 量化 attention
- `demo.py`        — 同时演示两者, 看精度损失与显存收益

## 运行
```bash
python -m llm_infer.m08_quantization.demo
```

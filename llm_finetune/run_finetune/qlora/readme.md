# QLoRA — 4-bit 基座 + LoRA

```bash
python -m llm_finetune.run_finetune.qlora.train_qlora  # ~8 s
```

## 直觉
LoRA 之后显存大头变成了冻结的基座本身。它只读不写 → 压成 4 bit 存, 算的时候现场反量化; 梯度只流进高精度的 LoRA 支路。

## 核心公式
`y = dequant(W_nf4) x + (α/r) B A x`。NF4: 16 个格点取 N(0,1) 的等概率分位数; 每 64 个权重共用一个 absmax scale。
存储 = 0.5 B/参数 + 4 B/64 参数 = 0.5625 B/参数, 对 fp32 的理论上限 4/0.5625 = 7.1×。

## 运行后应该看到什么
| | 数值 |
|---|---|
| 被量化的层 | 14 个 = 2 block × (4 注意力 + 3 SwiGLU); embedding / lm_head / RMSNorm 保持 fp32 |
| **整模型**权重 | fp32 389 KB → NF4 基座 59 KB (**6.57×**) + LoRA 76 KB |
| 单层相对量化误差 | 9.0% |
| 基座原任务 (copy) 留出集 EM | fp32 1.000 → NF4 1.000 |
| QLoRA 适配 sort, 300 步 | loss 4.190 → 0.328, 留出集 EM 0.402 (同配置 fp32 LoRA: 0.352) |
| 合并 (反量化 → 加 ΔW → Linear) 前后 logits 最大差 | 3e-06 |

6.57× < 7.1× 的差额就是没量化的 embedding 和 norm。本模型词表只有 16, embedding 占比很小; 真实 LLM 里 embedding + lm_head 占几个百分点, 整模型压缩比会更低一些。

## 常见误区
- 只量化注意力投影却报 "7.1× 压缩": 那只是被量化那几层的数字 (旧版的问题)。
- 量化 lm_head / embedding: 二者共享权重; embedding 是查表, 行分布不像正态; lm_head 的误差直接进 logits。bitsandbytes 默认也跳过。
- 合并后再量化回 4 bit: 刚学到的 ΔW 会被量化噪声抹掉一部分, 所以合并结果保持高精度。
- 打包时假设索引个数是偶数: block_size 为奇数时会错位 (脚本开头有这个单测)。

## 自测题
1. 为什么量化基座不影响 LoRA 的梯度正确性? — 基座是常数 buffer; ∂L/∂A、∂L/∂B 只依赖输入激活和上游梯度, 反量化误差只是让 "被适配的函数" 略有不同。
2. block_size 调小有什么得失? — scale 更贴合局部 → 误差更小; 但 scale 开销 4/bs 字节/参数变大 (所以论文再对 scale 做 double quantization)。
3. NF4 为什么比均匀 INT4 好? — 权重近似正态, 分位数格点让每个格子接住等量的权重, 期望量化误差更小。

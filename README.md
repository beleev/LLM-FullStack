# LLM 全栈教具库

从 numpy 手写反向传播 → PyTorch 现代架构 → 规模化训练 → 微调对齐 → 推理优化 → Agent 应用层 — **六个递进阶段**，把"训出一个能用、能部署、能行动的大模型系统"拆成可独立运行的教学模块。

每个模块都能 `python -m xxx.demo` 单独跑，CPU 几秒到几十秒跑完，零 GPU 依赖。附带一个 Vue 3 交互式 Web 教程，代码与讲解逐行对照。

## 六阶段总览

```
阶段 1 · llm_basic       用 numpy 手写 forward/backward/采样/BPE，看清数学怎么流过模型
    ↓
阶段 2 · llm_models      PyTorch 实现 20 个模型架构 (2017 Transformer → 2025 Qwen3-Next)
    ↓
阶段 3 · llm_train        纯 numpy 模拟分布式训练：DDP / TP / PP / ZeRO / EP / Ring Attention / FP8
    ↓
阶段 4 · llm_finetune    SFT / LoRA / QLoRA / DPO / RM / GRPO / 蒸馏 七大微调范式，复用 llm_models 的 LLaMA + Trainer
    ↓
阶段 5 · llm_infer        17 个推理优化模块 + mini-vLLM 引擎，覆盖业界 80% 推理技术
    ↓
阶段 6 · llm_agent        Agent 循环 / 工具 / 权限 / 记忆 / Hook / 持久化 / 子智能体 / 检索
```

## 安装

```bash
pip install -e .          # 需要 torch>=1.10 (阶段 2/4 用 PyTorch，其余只需 numpy)
python main.py            # 冒烟测试：跑 10 个核心模型的最小前向
```

Web 教程 (可选)：

```bash
cd web && npm install && npm run dev
```

## 目录结构

```
llm/
├── llm_basic/             阶段 1 · 最小可跑闭环 (纯 numpy，~45K 参数)
│   ├── model.py           forward/backward 成对实现，每条链式法则都看得见
│   ├── gradcheck.py       数值梯度 vs 解析梯度验证
│   ├── train.py           训练循环 (2000 步，loss 4.17 → ~1.9)
│   ├── sample.py          自回归采样 (temperature + top-k)
│   └── bpe.py             手写 byte-level BPE (GPT-2 同款思想)
│
├── llm_models/            阶段 2 · 架构家族 (PyTorch)
│   ├── layers/            可复用零件：attention / ffn / norm / pos / moe / ssm / adaln / vq
│   ├── models/            20 个模型：Transformer → BERT → GPT-3 → LLaMA → Mistral(SWA)
│   │                      → MTP → Qwen3-Next(混合线性) → Mixtral → Mamba → DeepSeek-V3/V3.2
│   │                      → CLIP → Whisper → Qwen2-VL → Qwen2.5-Omni
│   │                      → ImageVAE → DiT → MM-DiT → Video DiT → VAR
│   ├── training/          通用训练框架 (策略模式)：Trainer / Loss / Data / Diffusion
│   └── run_models/        前向 + 训练示例 (共 28 组)
│
├── llm_train/             阶段 3 · 规模化训练 (纯 numpy 模拟)
│   ├── m01–m13/           梯度累积 → DDP → TP → PP → ZeRO → 混合精度
│   │                      → 激活检查点 → checkpoint → 通信原语 → 训练稳定性
│   │                      → 专家并行 (EP) → 序列并行 (Ring Attention) → FP8 训练
│   └── full_loop/         多技术组合的训练主循环
│
├── llm_finetune/          阶段 4 · 微调对齐
│   ├── methods/           SFT / LoRA / QLoRA(NF4) / DPO / RewardModel / GRPO / 蒸馏 核心算法
│   ├── data/              指令数据 + 偏好对 + prompt 数据生成器
│   └── run_finetune/      端到端训练脚本 (CPU 一分钟内跑完)
│
├── llm_infer/             阶段 5 · 推理与部署优化 (纯 numpy)
│   ├── m01–m17/           KV Cache → PagedAttention → Continuous Batching
│   │                      → Prefix Cache → Radix Cache → Chunked Prefill
│   │                      → Speculative Decoding → Quantization → Tensor Parallel
│   │                      → Sampling → FlashAttention → CUDA Graph
│   │                      → Multi-LoRA → Structured Output → P/D Disaggregation
│   │                      → Attention Sinks (StreamingLLM) → EAGLE 投机解码
│   └── full_engine/       集成 mini-vLLM 引擎
│
├── llm_agent/             阶段 6 · Agent 应用层 (纯 Python stdlib)
│   ├── m01–m08/           Agent 循环 → 工具调用 → 权限门 → 上下文与记忆
│   │                      → Hook/Skill/MCP → 持久化 → 子智能体 → 向量检索 (RAG-lite)
│   └── full_loop/         多机制组合的 mini Agent harness
│
├── web/                   Vue 3 交互式教程 (Vite + vue-router)
│   └── src/views/         Home / Basic / Attention / Position / Blocks / MoE
│                          / Diffusion / Train / Finetune / Infer / Agent / Compare
│
├── main.py                冒烟测试入口
└── ref/                   参考项目 (nano-vllm / mini-sglang)
```

## 快速运行

### 阶段 1 · llm_basic — 手写反向传播

```bash
cd llm_basic
python prepare.py                                   # 下载 Tiny Shakespeare (~5s)
python gradcheck.py                                  # 验证 backward 正确性
python train.py                                      # 训练 2000 步 (~2 min)
python sample.py "ROMEO:" --max-new 300 --temperature 0.8
python bpe.py --merges 300                           # 手写 BPE：看词表怎么长出来
```

### 阶段 2 · llm_models — 架构家族

```bash
# 前向示例
python -m llm_models.run_models.language_models.gpt3.infer_gpt3
python -m llm_models.run_models.language_models.llama.infer_llama
python -m llm_models.run_models.language_models.mistral.infer_mistral   # SWA 滑动窗口
python -m llm_models.run_models.language_models.mtp.infer_mtp           # 多 token 预测
python -m llm_models.run_models.language_models.qwen3_next.infer_qwen3_next  # 混合线性注意力
python -m llm_models.run_models.moe.deepseek.infer_deepseek

# 合成数据训练
python -m llm_models.run_models.language_models.llama.train_llama
python -m llm_models.run_models.generative.dit.train_dit
python -m llm_models.run_models.multimodal.clip.train_clip
```

### 阶段 3 · llm_train — 规模化训练

```bash
python -m llm_train.m01_gradient_accumulation.demo
python -m llm_train.m02_data_parallel.demo
python -m llm_train.m05_zero_fsdp.demo
python -m llm_train.m11_expert_parallel.demo          # MoE all-to-all + 路由均衡
python -m llm_train.m12_sequence_parallel.demo        # Ring Attention
python -m llm_train.m13_fp8_training.demo             # FP8 + block scaling
python -m llm_train.full_loop.demo                   # 组合闭环
python -m llm_train.run_all                           # 全部跑一遍
```

### 阶段 4 · llm_finetune — 微调对齐

```bash
python -m llm_finetune.run_finetune.sft.train_sft    # 全参 SFT
python -m llm_finetune.run_finetune.lora.train_lora   # LoRA (PEFT, <1% 参数)
python -m llm_finetune.run_finetune.qlora.train_qlora  # QLoRA (NF4 4-bit 基座 + LoRA)
python -m llm_finetune.run_finetune.dpo.train_dpo     # DPO 偏好对齐
python -m llm_finetune.run_finetune.rm.train_rm       # Reward Model (RLHF 第二阶段)
python -m llm_finetune.run_finetune.grpo.train_grpo   # GRPO (R1 式 RLVR 在线 RL)
python -m llm_finetune.run_finetune.distill.train_distill  # 知识蒸馏
```

### 阶段 5 · llm_infer — 推理优化

```bash
python -m llm_infer.m01_kv_cache.demo
python -m llm_infer.m02_paged_attention.demo
python -m llm_infer.m03_continuous_batching.demo
python -m llm_infer.m11_flash_attention.demo
python -m llm_infer.m16_attention_sinks.demo          # StreamingLLM 有界 KV
python -m llm_infer.m17_eagle_speculative.demo        # EAGLE 特征级投机解码
python -m llm_infer.full_engine.demo                  # mini-vLLM 引擎
```

### 阶段 6 · llm_agent — Agent 应用层

```bash
python -m llm_agent.m01_agent_loop.demo
python -m llm_agent.m03_permissions.demo
python -m llm_agent.m07_subagents.demo
python -m llm_agent.m08_retrieval.demo                # TF-IDF 向量检索 (RAG-lite)
python -m llm_agent.full_loop.demo                    # mini Agent harness
python -m llm_agent.run_all                           # 全部跑一遍
```

## 架构演进时间线

### 左脑：语言 LLM

| 年份 | 模型 | 核心组件 |
|------|------|---------|
| 2017 | Transformer | `MultiHeadAttention` + `FeedForward`(ReLU) + `LayerNorm` + `SinPositionalEncoding` |
| 2018 | BERT | 双向注意力 + `BERTEmbeddings` + MLM head |
| 2020 | GPT-3 | `MultiHeadAttention` + `GeLUFeedForward` + Weight Tying |
| 2023 | LLaMA | `GroupedQueryAttention` + `SwiGLUFeedForward` + `RMSNorm` + `RotaryPositionalEncoding` |
| 2023 | Mistral | LLaMA 骨架 + `build_sliding_window_mask` (SWA，KV cache 封顶 O(W)) |
| 2024 | Mixtral | LLaMA 骨架 + `MixtralMoE` (softmax top-k) |
| 2024 | MTP | LLaMA 骨架 + `MTPModule` 级联 (多 token 预测，DeepSeek-V3 训练目标) |
| 2025 | Qwen3-Next | `GatedDeltaNet` (线性注意力 + delta rule) 3:1 混合 `GroupedQueryAttention` |
| 2023 | Mamba | `SelectiveSSM` + 1D conv + gate，线性复杂度 O(T) |
| 2024 | DeepSeek-V3 | `MultiHeadLatentAttention` + `DeepSeekMoE` (sigmoid + shared experts + aux-loss-free) |
| 2025 | DeepSeek-V3.2 | V3 + `MultiHeadLatentSparseAttention` (MLA + `LightningIndexer`) |

### 眼耳：多模态理解

| 年份 | 模型 | 关键设计 |
|------|------|---------|
| 2021 | CLIP | 对比学习双塔 + `logit_scale` 可学习温度 |
| 2022 | Whisper | Conv1D stem + Transformer Encoder + Cross-Attention Decoder |
| 2024 | Qwen2-VL | ViT + `PerceiverResampler` + M-RoPE (三轴) |
| 2025 | Qwen2.5-Omni | 双脑 Thinker-Talker + 流式语音 codec |

### 右脑：扩散 / 自回归生成

| 模型 | 训练目标 | 代码位置 |
|------|---------|---------|
| `ImageVAE` / `CausalVideoVAE` | MSE + KL | `models/generative/vae.py` `vae3d.py` |
| `DiT` (2023) | DDPM ε-pred | `models/generative/dit.py` |
| `MMDiT` (SD3/FLUX) | Rectified Flow v-pred | `models/generative/mmdit.py` |
| `VideoDiT` (Sora-lite) | DDPM ε-pred + spacetime patches | `models/generative/video_dit.py` |
| `VARModel` (VAR) | next-token CE | `models/generative/var.py` |

## 业界技术覆盖度

### 训练技术 (llm_train)

| 技术 | 覆盖 | 模块 |
|------|:---:|------|
| Gradient Accumulation | yes | m01, full_loop |
| Data Parallel / DDP | yes | m02, full_loop |
| Tensor Parallel (Megatron) | yes | m03 |
| Pipeline Parallel | yes | m04 |
| ZeRO / FSDP | yes | m05, full_loop |
| Mixed Precision / Loss Scaling | yes | m06, full_loop |
| Activation Checkpointing | yes | m07 |
| Checkpoint / Resume | yes | m08 |
| Communication Collectives | yes | m09 |
| Warmup / Cosine / Grad Clip / NaN Guard | yes | m10 |
| Expert Parallel / MoE all-to-all | yes | m11 |
| Sequence Parallel / Ring Attention | yes | m12 |
| FP8 Training (E4M3/E5M2 + block scaling) | yes | m13 |

### 推理技术 (llm_infer)

| 技术 | 覆盖 | 模块 |
|------|:---:|------|
| KV Cache | yes | m01 |
| PagedAttention | yes | m02 |
| Continuous Batching | yes | m03 |
| Prefix Cache (hash) | yes | m04 |
| Radix Cache (SGLang) | yes | m05 |
| Chunked Prefill | yes | m06 |
| Speculative Decoding | yes | m07 |
| Weight / KV Quantization (INT8) | yes | m08 |
| Tensor Parallel | yes | m09 |
| Sampling (greedy/temp/top-k/top-p/min-p) | yes | m10 |
| FlashAttention | yes | m11 |
| CUDA Graph | yes | m12 |
| Multi-LoRA Serving | yes | m13 |
| Structured Output (JSON/Grammar) | yes | m14 |
| P/D Disaggregation | yes | m15 |
| Attention Sinks / StreamingLLM | yes | m16 |
| EAGLE 特征级投机解码 | yes | m17 |

### 微调技术 (llm_finetune)

| 技术 | 覆盖 | 说明 |
|------|:---:|------|
| SFT (Supervised Fine-Tuning) | yes | 全参微调，prompt masking |
| LoRA (Low-Rank Adaptation) | yes | <1% 参数，极快收敛 |
| QLoRA (NF4 4-bit 基座 + LoRA) | yes | 真 4bit 打包，基座 ~7x 压缩 |
| DPO (Direct Preference Optimization) | yes | 双前向 + KL 约束，跳过 reward model |
| Reward Model (RLHF 第二阶段) | yes | value head + Bradley-Terry 偏好损失 |
| GRPO (R1 式在线 RL) | yes | 组内相对优势替代 critic + RLVR 可验证奖励 |
| Knowledge Distillation | yes | 软标签 + 温度 T² 补偿，teacher→student |

### Agent 技术 (llm_agent)

| 技术 | 覆盖 | 模块 |
|------|:---:|------|
| Agent Loop / ReAct | yes | m01 |
| Tool Calling | yes | m02 |
| Permission Gate | yes | m03 |
| Context Compaction & File Memory | yes | m04 |
| Hooks / Skills / MCP | yes | m05 |
| Session Persistence & Resume | yes | m06 |
| Subagents (隔离 + summary return) | yes | m07 |
| Retrieval / RAG-lite (TF-IDF 余弦) | yes | m08 |

## 设计取舍速查表

| 主题 | 早期 | 现代 LLM | 代码位置 |
|------|------|----------|---------|
| Normalization | Post-LN (依赖 warmup) | Pre-LN + RMSNorm | `layers/core/blocks.py`, `normalization.py` |
| FFN 激活 | ReLU | GELU → SwiGLU | `layers/core/feedforward.py` |
| 位置编码 | Sin/Learned 绝对 | RoPE → M-RoPE | `layers/core/position_encoding.py` |
| KV Cache | MHA (最大) | MQA → GQA → MLA → MLA+DSA | `layers/core/attention.py` |
| 注意力范围 | 全因果 O(T²) | SWA 带状 O(T·W) + sink | `utils/masks.py`, `models/.../mistral.py` |
| 训练目标 | next-token CE | + MTP 多 token 预测 | `models/language_models/mtp.py` |
| 偏好对齐 | RLHF (RM+PPO) | DPO (离线) / GRPO (在线, 无 critic) | `llm_finetune/methods/` |
| FFN 结构 | 单 FFN | Mixtral MoE → DeepSeekMoE | `layers/sparse/moe.py` |
| 序列建模 | Attention O(T^2) | Mamba SSM / DeltaNet+Attn 混合 | `layers/sparse/ssm.py`, `linear_attention.py` |
| 扩散骨架 | UNet (SD 1.5) | DiT + adaLN-Zero | `layers/diffusion/adaln.py` |
| 扩散目标 | ε-prediction | Rectified Flow v-prediction | `training/diffusion.py` |

## 参考文献

<details>
<summary>展开完整参考文献列表</summary>

### 基础 / 理解类
- *Attention Is All You Need* — Vaswani et al., 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- *BERT* — Devlin et al., 2019 ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805))
- *GPT-3* — Brown et al., 2020 ([arXiv:2005.14165](https://arxiv.org/abs/2005.14165))
- *LLaMA / LLaMA 2 / Llama 3* — Touvron et al. / Meta AI, 2023-2024
- *Mistral 7B* — Jiang et al., 2023 ([arXiv:2310.06825](https://arxiv.org/abs/2310.06825))
- *Mixtral of Experts* — Jiang et al., 2024 ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088))
- *Mamba* — Gu & Dao, 2023 ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))
- *Gated Delta Networks* — Yang et al., 2024 ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464))
- *DeepSeek-V2 / V3 / V3.2 Technical Reports* — DeepSeek-AI, 2024-2025

### 组件层面
- *RoFormer* (RoPE) — Su et al., 2021 ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))
- *RMSNorm* — Zhang & Sennrich, 2019 ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467))
- *GLU Variants* — Shazeer, 2020 ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202))
- *GQA* — Ainslie et al., 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245))
- *Switch Transformer* — Fedus et al., 2021 ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961))
- *Multi-token Prediction* — Gloeckle et al., 2024 ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
- *Ring Attention* — Liu et al., 2023 ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889))

### 多模态
- *CLIP* — Radford et al., 2021 ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
- *Whisper* — Radford et al., 2022 ([arXiv:2212.04356](https://arxiv.org/abs/2212.04356))
- *Flamingo* — Alayrac et al., 2022 ([arXiv:2204.14198](https://arxiv.org/abs/2204.14198))
- *Qwen2-VL* — Wang et al., 2024 ([arXiv:2409.12191](https://arxiv.org/abs/2409.12191))
- *Qwen2.5-Omni Technical Report* — Xu et al., 2025

### 生成模型
- *VAE* — Kingma & Welling, 2013 ([arXiv:1312.6114](https://arxiv.org/abs/1312.6114))
- *DDPM* — Ho et al., 2020 ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239))
- *DDIM* — Song et al., 2021 ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502))
- *Latent Diffusion / Stable Diffusion* — Rombach et al., 2022 ([arXiv:2112.10752](https://arxiv.org/abs/2112.10752))
- *Classifier-Free Diffusion Guidance* — Ho & Salimans, 2022 ([arXiv:2207.12598](https://arxiv.org/abs/2207.12598))
- *DiT* — Peebles & Xie, 2023 ([arXiv:2212.09748](https://arxiv.org/abs/2212.09748))
- *SD3 / Rectified Flow* — Esser et al., 2024 ([arXiv:2403.03206](https://arxiv.org/abs/2403.03206))
- *Flow Matching* — Lipman et al., 2023 ([arXiv:2210.02747](https://arxiv.org/abs/2210.02747))
- *Sora* — OpenAI, 2024
- *VAR* — Tian et al., NeurIPS 2024 ([arXiv:2404.02905](https://arxiv.org/abs/2404.02905))
- *VQ-VAE* — van den Oord et al., 2017 ([arXiv:1711.00937](https://arxiv.org/abs/1711.00937))

### 推理优化
- *vLLM / PagedAttention* — Kwon et al., 2023 ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180))
- *FlashAttention* — Dao et al., 2022 ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
- *Speculative Decoding* — Leviathan et al., 2023 ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192))
- *EAGLE* — Li et al., 2024 ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077))
- *SGLang / RadixAttention* — Zheng et al., 2024 ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104))
- *StreamingLLM / Attention Sinks* — Xiao et al., 2023 ([arXiv:2309.17453](https://arxiv.org/abs/2309.17453))

### 微调 / 对齐
- *LoRA* — Hu et al., 2021 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- *QLoRA* — Dettmers et al., 2023 ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
- *DPO* — Rafailov et al., 2023 ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290))
- *InstructGPT* — Ouyang et al., 2022 ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155))
- *GRPO / DeepSeekMath* — Shao et al., 2024 ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
- *DeepSeek-R1* — DeepSeek-AI, 2025 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948))
- *Knowledge Distillation* — Hinton et al., 2015 ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531))

</details>

## License

MIT

# LLM 全栈教具库

把"训出一个能用、能部署、能行动的大模型系统"拆成**六个递进阶段**，每个阶段都能单独跑。

numpy 手写反向传播 → PyTorch 现代架构 → 规模化训练 → 微调对齐 → 推理优化 → Agent 应用层。

- 每个模块 `python -m xxx.demo` 单独跑，CPU 几秒到几十秒，零 GPU 依赖。
- 每个 demo 末尾都用 `assert` 验证它声称的结论（与单卡基线逐位相等、与朴素解码逐 token 相同……），不是打印一句 OK。
- 配套 **交互式 Web 教程**：84 个可拖、可点、可单步播放的实验台，218 道章末自测，还有术语速查。页面上的代码直接取自仓库里的 Python 源文件。

**在线教程**: https://beleev.github.io/LLM-FullStack/

## 怎么学

```
阶段 1 · llm_basic       numpy 手写 forward / backward / Adam / 采样 / BPE —— 看清梯度怎么流过 Transformer
    ↓
阶段 2 · llm_models      PyTorch 实现 23 个模型: 2017 Transformer → 2025 GPT-OSS / LLaDA / Qwen3-Next
    ↓
阶段 3 · llm_train       纯 numpy 模拟大规模训练: DDP / TP / PP / ZeRO / EP / Ring + Ulysses / FP8 / FP4 / Muon
    ↓
阶段 4 · llm_finetune    SFT / LoRA / QLoRA / DoRA / DPO / SimPO / ORPO / RM / GRPO 及变体 / 离线与在线蒸馏
    ↓
阶段 5 · llm_infer       22 个推理优化模块 + 真分页的 mini-vLLM 引擎
    ↓
阶段 6 · llm_agent       Agent 循环 / 工具 / 权限 / 上下文工程 / Hook / Skill / MCP / 子智能体 / 护栏 / 评测
```

推荐节奏：**先在网页上拖实验台建立直觉 → 做章末自测 → 再跑对应的 Python demo 读源码**。每个模块目录下的 README 都是同一个结构：

> 直觉 → 核心公式 → 运行后应该看到什么（真实数字）→ 与真实系统的差距 → 常见误区 → 3 道自测题

## 安装与运行

```bash
pip install -e ".[dev]"      # Python ≥ 3.10; 阶段 2/4 需要 torch, 阶段 1/3/5 只要 numpy, 阶段 6 纯标准库
python main.py               # 冒烟: 每个核心模型跑一次前向, 断言初始 loss ≈ ln V、KV cache 有无一致

pytest                       # 所有 demo / infer 脚本 (约 1.5 分钟)
pytest -m slow               # 所有训练脚本
```

每个阶段一条命令全跑：

```bash
python -m llm_train.run_all
python -m llm_finetune.run_all
python -m llm_infer.run_all
python -m llm_agent.run_all
cd llm_basic && python gradcheck.py && python train.py && python sample.py "ROMEO:"
```

单个模块：

```bash
python -m llm_models.run_models.language_models.llama.train_llama
python -m llm_models.run_models.moe.gpt_oss.infer_gpt_oss
python -m llm_train.m05_zero_fsdp.demo
python -m llm_finetune.run_finetune.grpo.train_grpo
python -m llm_infer.full_engine.demo
python -m llm_agent.m09_mcp.demo
```

Web 教程：

```bash
cd web && npm install && npm run dev
npm run check:sources        # 校验页面引用的每个 Python 符号都还存在
```

## 技术覆盖

### 阶段 2 · 模型架构 (llm_models)

| 主题 | 内容 |
|------|------|
| 注意力 | MHA → MQA/GQA → MLA（含只缓存 latent 的解码路径）→ DSA（LightningIndexer + KL 对齐损失）；QK-Norm；可学习 attention sink |
| 位置 | Sinusoidal / Learned → RoPE → NTK / YaRN 长上下文缩放 → M-RoPE |
| 序列建模 | 滑动窗口（Mistral）、滑窗/全局交替（GPT-OSS-mini）、Gated DeltaNet 混合（Qwen3-Next）、Mamba 选择性 SSM |
| MoE | Mixtral softmax top-k；DeepSeekMoE sigmoid + 共享专家 + **真正会更新的** aux-loss-free 路由偏置 |
| 训练目标 | next-token、MTP 多 token 预测、LLaDA 掩码扩散语言模型 |
| 生成 | 所有 decoder LM 共用带 KV cache 的 `GenerationMixin`（有/无 cache 输出 `torch.equal`） |
| 多模态 | CLIP、Whisper、Qwen2-VL、Qwen2.5-Omni |
| 视觉生成 | VAE / 因果 3D VAE、DiT、MM-DiT（Rectified Flow）、Video DiT、VAR（真 next-scale + 多尺度残差 VQ） |

### 阶段 3 · 规模化训练 (llm_train)

| 模块 | 技术 | 验证方式 |
|------|------|---------|
| m01–m02 | 梯度累积、DDP | 与单卡整 batch 一致 |
| m03 | Megatron 张量并行（前向 + 反向 all-reduce） | W、dX 均与 dense 一致 |
| m04 | GPipe / 1F1B / 交错 1F1B | 气泡 = 公式；1F1B 峰值精确为 `[PP, …, 1]` |
| m05 | ZeRO-1/2/3 + FSDP gather→compute→free | 三档与 DDP 逐位相同；字节数 = 2+2+12 公式 |
| m06 | FP16 / BF16、动态 loss scaling | 真实 fp16 溢出触发跳步 |
| m07 | 激活重算（真反向） | 梯度与全存基线逐位相同；峰值 = L/k + k |
| m08 | 断点续训（含 RNG、数据游标） | 漏恢复任何一项都有断言量化偏差 |
| m09 | 集合通信 + ring all-reduce + 通信量计数 | all_reduce = reduce_scatter + all_gather |
| m10 | warmup / cosine / WSD、裁剪、NaN guard | — |
| m11 | 专家并行（两次真 all-to-all）+ aux-loss-free 均衡 | 与 dense MoE 一致 |
| m12 / m16 | Ring Attention（zigzag 均衡）/ Ulysses | 与完整 attention 一致 |
| m13 / m15 | FP8 配方消融 / MXFP4 · NVFP4 | 四臂消融各动一个变量 |
| m14 | Muon（Newton–Schulz）+ QK-clip | 含 Adam 反而更好的反例 |
| full_loop | DDP + 累积 + AMP + 分片 Adam + 裁剪 + 续训 | 续训逐位相同；fp32 下与单卡差 6e-8 |

### 阶段 4 · 微调对齐 (llm_finetune)

| 类别 | 方法 |
|------|------|
| 指令微调 | SFT（prompt mask） |
| 参数高效 | LoRA · QLoRA（NF4 真 4-bit 打包）· DoRA |
| 偏好对齐 | DPO · SimPO · ORPO · Reward Model（Bradley–Terry） |
| 在线 RL | GRPO（重要性比率 + clip），及 DAPO / Dr.GRPO / GSPO 变体 |
| 蒸馏 | 离线（forward KL）· on-policy（reverse KL） |

合成任务是**依赖 prompt 的可学习任务**。验证看留出集指标，不看"loss 下降了"。

### 阶段 5 · 推理优化 (llm_infer)

| 方向 | 模块 |
|------|------|
| 省 KV 显存 | KV Cache · PagedAttention · Prefix Cache · Radix Cache · KV 占用对比 / MLA 解码 · 分层 KV offload |
| 调度 | Continuous Batching（含抢占）· Chunked Prefill（Sarathi 混批）· P/D 分离 |
| 少算几次 | Speculative Decoding（拒绝采样 + 分布保持检验）· EAGLE · 树状投机 · 稀疏注意力解码 |
| 压缩 | INT8 / group-wise INT4 / AWQ · KIVI KV 量化 |
| 算子与并行 | FlashAttention（Q/KV 双向分块 + LSE）· Tensor Parallel · CUDA Graph · MoE 推理 EP + EPLB |
| 输出控制 | Sampling · Structured Output（token 级预编译语法 mask）· Attention Sinks（SinkCache）· Multi-LoRA |

`full_engine` 是真分页的 mini-vLLM：物理 KV block 池、前缀命中真的跳过计算、分块 prefill 混批、recompute 抢占，**每条输出与朴素 greedy 逐 token 相同**。

### 阶段 6 · Agent (llm_agent)

| 层 | 内容 |
|------|------|
| 循环 | Agent 循环（`tool_use` / `tool_result` content block）· JSON Schema 工具 + 并行调用 |
| 安全 | 权限门（deny > ask > allow，六种模式，命令归一化与分段）· 护栏（注入、路径围栏、脱敏） |
| 上下文 | 压缩与文件记忆 · 上下文工程 · BM25 检索（中文 bigram） |
| 扩展 | Hooks · Skills 渐进式披露 · 真 MCP（stdio JSON-RPC） |
| 编排 | 计划模式 · 子智能体 · orchestrator–workers |
| 运维 | JSONL 持久化与恢复 · Agent 评测（pass@k / pass^k） |
| 接真模型 | 可选的 Claude API 适配器，默认不联网 |

工具全部是模拟或沙箱内的，可以放心跑。

## Web 教程

### 73 章不用全读 —— 先看 [速成路线](https://beleev.github.io/LLM-FullStack/#/fast-track)

| 档位 | 章数 | 用时 (粗估) | 读完能干什么 |
|------|-----:|------------|--------------|
| ★ 冲刺 | 12 | 约 1.4 小时 | 每个阶段最核心的那一两页, 接得上下一阶段 |
| ● 主干 | 36 | 约 3.5 小时 | 讲清一个大模型怎么训出来、怎么上线、怎么变成会行动的系统 |
| ○ 全部 | 73 | 约 7.6 小时 | 加上 2026 年的前沿技术、同一问题的其他解法和生成模型分支 |

分层只是阅读建议, 不影响任何章节的内容。侧栏可以按档位收起扩展章, 每章开头标着它属于哪一层,
章内最该带走的那条结论有「重点」徽章。用时按内容字数粗估, 只用来排计划。

- **实验台**：每章"先动手再读字"。拖图上的点、点格子、步进播放；右侧 2–4 个会变的数字；每个实验台配一道"先预测再看答案"的挑战题。模拟与 Python 模块算的是同一件事，关键数字对得上。
- **真源码**：章节里的代码块在构建期直接读 `llm_*/**/*.py` 并按符号截取，CI 校验每个引用都有效，不会和仓库漂移。
- **学习辅助**：章末自测（全对侧栏打 ✓）、学习进度与"接着上次学"（只存本机）、术语速查页、键盘 ← / → 翻章、移动端可用。
- **加内容只加文件**：实验台、章节、自测、术语都按阶段分文件自动注册，见 [`web/LABS.md`](web/LABS.md)。章节正文全部住在 `web/src/data/topics/<stage>.js`。
- **说人话**：73 章正文按四条规矩写成——先说具体的再说抽象的、拆掉名词堆、一个数字胜过一个形容词、先说为什么疼再说怎么治。每章的三条要点里，最该带走的那条有「重点」徽章。

## 这个教具的边界

教具最容易骗人的地方，是演示了一个名字却没演示那件事。

所以这里的每个 demo 都在末尾 `assert` 它声称的结论：与单卡基线逐位相等、与朴素解码逐 token 相同、梯度确实非空。"看起来在跑"和"真的在做"因此是分开的。

有几个结论在玩具规模上就是不成立，各模块 README 的"与真实系统的差距"一节写得更细：

- 随机权重的小模型**没有 attention sink 现象**，稀疏解码的块打分也**不优于随机选块**。这两个效应只在植入了 sink / needle 的合成数据上演示。
- FP8 的 per-tensor 与 block scaling 在玩具任务上**打平**，outlier 要超过约 1e5 倍才拉开差距。
- `llm_basic` 的 2 层配置**不优于** 1 层（val 2.045 vs 1.981）。
- CUDA Graph 的加速比来自**显式注入的 launch 开销模型**，不是实测。
- 同步数下**全参微调优于 LoRA**（留出集 EM 0.809 vs 0.352）。LoRA 买的是显存和分发，不是收敛速度。

这些都如实写在程序输出和文档里，没有构造数据让结论好看。

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


### 长上下文 / 低精度 / 新架构
- *YaRN* — Peng et al., 2023 ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071))
- *QK-Norm / OLMo 2* — Team OLMo, 2024 ([arXiv:2501.00656](https://arxiv.org/abs/2501.00656)); *Qwen3 Technical Report*, 2025
- *gpt-oss model card* — OpenAI, 2025
- *LLaDA: Large Language Diffusion Models* — Nie et al., 2025 ([arXiv:2502.09992](https://arxiv.org/abs/2502.09992))
- *Auxiliary-Loss-Free Load Balancing* — Wang et al., 2024 ([arXiv:2408.15664](https://arxiv.org/abs/2408.15664))
- *Muon* — Jordan et al., 2024; *Muon is Scalable for LLM Training* — Moonshot AI, 2025 ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982))
- *OCP Microscaling Formats (MX) Specification*, 2023; *NVFP4* — NVIDIA, 2025
- *DeepSpeed-Ulysses* — Jacobs et al., 2023 ([arXiv:2309.14509](https://arxiv.org/abs/2309.14509))
- *ZeRO* — Rajbhandari et al., 2019 ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054))
- *DoRA* — Liu et al., 2024 ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353))
- *SimPO* — Meng et al., 2024 ([arXiv:2405.14734](https://arxiv.org/abs/2405.14734)); *ORPO* — Hong et al., 2024 ([arXiv:2403.07691](https://arxiv.org/abs/2403.07691))
- *DAPO* — Yu et al., 2025 ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)); *Dr. GRPO* — Liu et al., 2025 ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)); *GSPO* — Zheng et al., 2025 ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071))
- *On-Policy Distillation of Language Models (GKD)* — Agarwal et al., 2023 ([arXiv:2306.13649](https://arxiv.org/abs/2306.13649))
- *Sarathi-Serve (chunked prefill)* — Agrawal et al., 2024 ([arXiv:2403.02310](https://arxiv.org/abs/2403.02310))
- *AWQ* — Lin et al., 2023 ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)); *KIVI* — Liu et al., 2024 ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750))
- *Medusa* — Cai et al., 2024 ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)); *EAGLE-2* — Li et al., 2024 ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858))
- *XGrammar* — Dong et al., 2024 ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100))
- *Quest* — Tang et al., 2024 ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)); *Native Sparse Attention* — Yuan et al., 2025 ([arXiv:2502.11089](https://arxiv.org/abs/2502.11089))
- *Mooncake* — Qin et al., 2024 ([arXiv:2407.00079](https://arxiv.org/abs/2407.00079)); *DeepSeek EPLB*, 2025
- *Model Context Protocol* — modelcontextprotocol.io; *Building Effective Agents* — Anthropic, 2024
- *τ-bench (pass^k)* — Yao et al., 2024 ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045))

</details>

## License

MIT

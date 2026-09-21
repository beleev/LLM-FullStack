# Whisper (语音 → 文本的 Encoder-Decoder)

## 直觉
把声谱图当成一张"时间 × 频率"的图: 两层 1D 卷积把它切成 50 Hz 的帧 token, Encoder 双向读完整段音频;
Decoder 就是一个普通的因果语言模型, 只是每层多一个 cross-attention 去"听" Encoder 的输出。

## 核心公式
```
mel [B, 80, T_mel] → Conv1d(k=3) → GELU → Conv1d(k=3, stride=2) → GELU → [B, T_mel/2, D] + sinPE → Encoder
Decoder 第 l 层:  x += SelfAttn_causal(x);  x += CrossAttn(Q=x, K=V=encoder_hidden);  x += FFN(x)
loss = CE(logits[:, t], token[t+1])      # teacher forcing; decoder 开头的 task token 决定转写 / 翻译
```

## 运行命令
```bash
python -m llm_models.run_models.multimodal.whisper.infer_whisper
python -m llm_models.run_models.multimodal.whisper.train_whisper
```

## 运行后应该看到什么 (CPU 实测)
- infer: `mel (2, 80, 100) -> encoder (2, 50, 128) | 换音频 logits 变化 0.0305 | 改未来 token 过去变化 0.0e+00`。
- train: 2 条样本**文本输入相同、音频不同、标签不同**。初始 loss 6.157 (ln 500 = 6.215) → 150 步后 0.041,
  远低于"不听音频"的理论下界 ln 2 = 0.693 ⇒ cross-attention 在起作用。初始化修复前初始 loss 是 **82.30**。
- 数据是固定随机 batch: 这是记忆, 不是语音识别。

## 常见误区
- "Whisper 需要专门的语音架构" —— 除了 Conv stem, 其余与 2017 Transformer 完全相同; 能力来自 68 万小时弱监督数据。
- "音频和文本拼成一条序列" —— 那是 Qwen2-Audio / Omni 的 prefix 路线; Whisper 的音频只通过 cross-attn 的 K/V 进入 decoder。
- "cross-attn 也要因果 mask" —— 不要, 整段音频在解码前已完整可得。

## 自测题
1. 30 秒音频进入 Encoder 的序列长度? —— 3000 帧 mel, stride=2 后 1500。
2. 为什么训练脚本里 loss 低于 ln 2 才算数? —— 两条样本文本前缀相同, 只看文本时每个位置最多排除到 2 个候选, CE 下界为 ln 2。
3. 推理时哪部分只算一次? —— Encoder (及 cross-attn 的 K/V); Decoder 自回归逐 token 运行。

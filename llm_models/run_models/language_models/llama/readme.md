# LLaMA — 现代开源 LLM 的模板

## 直觉
LLaMA = GPT-3 的骨架 + 四个换代零件: GQA (省 KV cache)、SwiGLU (门控 FFN)、RMSNorm (不减均值)、RoPE (相对位置)。
本库里它几乎是纯组装, 也是 KV cache / QK-Norm / attention sink / RoPE 缩放这些 "可选零件" 的试验台。

## 核心公式
- GQA: H 个 Q head 共享 Hkv 对 K/V, cache 每 token 每层 `2·Hkv·Dh` (MHA 是 `2·H·Dh`)。
- QK-Norm: `q̂ = RMSNorm(q), k̂ = RMSNorm(k)` → `|q̂·k̂|/sqrt(Dh) ≤ sqrt(Dh)`, logit 有界。
- attention sink: `softmax([scores, s_h])[..., :-1]`, 每 head 一个可学 `s_h`; 行和 < 1, head 可以 "谁都不看"。
- 左 padding: 全屏蔽行 softmax 出 NaN → `combine_causal_and_padding_mask` 让这种行只看自己。

## 运行命令
```bash
python -m llm_models.run_models.language_models.llama.train_llama
python -m llm_models.run_models.language_models.llama.infer_llama
```

## 运行后应该看到什么 (实测, CPU)
- train: `初始 loss 7.073 vs ln V = 6.908 | 最终 loss 0.060`。
- infer:
  - `[1]` 生成 200 token 有/无 cache 完全一致, 加速约 5.7x; 每 token 每层缓存 128 个数 (MHA 要 512)。
  - `[2]` 左 padding 5 位: 无 NaN, 真实位置 logits 最大偏差 5.4e-07。
  - `[3]` `qk_norm=False: 1.20 → 120.31` (权重×10 → logit×100); `qk_norm=True: 3.44 → 3.44`。
  - `[4]` sink→-∞ 等价普通 GQA; 逐 token cache 解码与整段前向一致。

## 常见误区
- "loss 降到 0.06 说明模型好": 数据是固定的一个随机 batch, 这只是 "能背下来"。
- "左 padding 时 pad 位置输出什么都行, 所以 NaN 无所谓": NaN 会经残差和 loss 的梯度污染整个 batch。
- "QK-Norm 在 RoPE 之后做也一样": 旋转不改范数, 数值上界一样, 但约定是 RoPE 之前 (Qwen3/OLMo-2), 与公开权重对齐需要一致。
- "GQA 减少了计算量": 主要减少的是 KV cache 显存/带宽, Q 侧的注意力计算没变。

## 自测题
1. 8 个 Q head、2 个 KV head, cache 比 MHA 小几倍? —— 4 倍 (脚本打印 128 vs 512)。
2. 为什么左 padding + RoPE 时真实位置的输出与不 padding 时一致, 而 Sin-PE 不行? —— RoPE 的内积只依赖相对距离, 整体平移 5 位不影响; Sin-PE 是绝对位置, 平移后 embedding 变了。
3. sink logit 固定为 0 时, 一个所有真实 score 都是 -5 的 query 行, 注意力总和约多少? —— `T·e^{-5} / (T·e^{-5} + 1)`, T=16 时 ≈ 0.10; 其余 0.90 进了 sink。

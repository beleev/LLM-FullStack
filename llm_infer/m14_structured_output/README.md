# M14 — Structured Output: 让模型只能说"合法"的话

## 场景
- 要求 JSON 输出, 又怕模型漏个引号
- function call: 第一个 token 必须是 `{`, 最后必须是 `}`
- 类型化输出: 性别只能 `male/female/other`

## 思路
**在采样前, 把"非法 token"的 logits 置 -inf, 强制模型只能选合法 token。**
合法集合由 grammar / regex / FSM (有限状态机) 实时算出。

```
  current state                      可选下一 token (合法集)
  ───────────                       ────────────────────────
  start                              "{"
  after "{"                          '"' or whitespace
  inside string                      any char
  after closing "                    ":" or "," or "}"
  ...
```

每一步:
1. FSM 根据已生成 token, 给出"合法 token id 集合"
2. mask = (legal? 0 : -inf), apply 到 logits
3. 正常 sample
4. FSM 吃掉新 token, 状态前进

## 主流库
- **outlines** — 用 regex / context-free grammar 编译成 FSM
- **lm-format-enforcer** — JSON Schema 直接转 FSM
- **xgrammar** — Apache 2.0, 与 vLLM 集成最紧

## 本模块
实现一个**JSON 合法性 FSM**, 强制模型输出形如 `{"name":"abc","age":42}` 的内容。
配合任意 logits (随机或真实模型), 都能保证 100% 合法。

## 运行
```bash
python -m llm_infer.m14_structured_output.demo
```

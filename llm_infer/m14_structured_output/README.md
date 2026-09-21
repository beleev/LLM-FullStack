# M14 — Structured Output: FSM 约束解码与 token mask 预编译

## 直觉
要 JSON / function call / 枚举值时, 不靠"求模型别写错", 而是**采样前把会违反语法的 token 的
logit 置 -inf**。合法集合由一个状态机给出: 每步 `logits → mask → 采样 → 状态机吃掉新 token`。
难点在于真实词表是**多字符 token** (`{"`、`":`、`true`), 一个 token 要在字符级 FSM 上连走几步;
每步对 V 个 token 逐字符试走是 O(V·len) 的 CPU 开销, 夹在两次 GPU 前向之间。FSM 状态有限,
所以可以离线把"每个状态 × 每个 token"全部试走一遍存成表, 在线只查表 —— 这就是
outlines 的 index 和 xgrammar 的 token mask cache。

## 核心数据结构或公式
- `JsonFSM` (`grammar.py`): 字符级状态机, `legal_chars()` / `advance(ch)`; 计数器有上界, 所以
  配置 `key()` 是有限集。`compile_char_dfa` 用 BFS 枚举成显式 DFA `trans[s] = {ch: s'}`。
- `next_state[s, t]` (S,V) int: 从状态 s 把 token t 的**整段字符**走完的落点, 走不通 = -1。
  `mask_table = next_state >= 0` (S,V) bool。EOS 只在接受态合法。
- `need[s, t]` (S,V): 选了 t 之后最少还要几个 token 才能以 EOS 收尾 (token 图最短路)。
  预算感知 mask: `ok = mask_table[s] & (need[s] <= 剩余 token 数 - 1)`。
- 每步: `logits[~ok] = -inf` → greedy/采样 → `s = next_state[s, tok]`。

## 运行后应该看到什么
```bash
python -m llm_infer.m14_structured_output.demo
```
```
[1] 字符级 FSM + 随机 logits (200 个 seed)   pair 数分布 = {1: 99, 2: 47, 3: 54}
[2] 词表 V = 96 (多字符 48), DFA 状态 S = 101, 表 9696 项, 合法占比 21.0%, 预编译 7.9 ms
    {"thd":7100,"fvalue":3300,"axxxe":2584}
    约束采样 100 次, 合法 JSON = 100/100        (真实 TinyLM logits, 随机权重)
[3] 无约束采样 100 次, 合法 JSON = 0/100
[4] max_tokens=12:  mode=mask 33/100 (例 '{ "smfmpr": true, "')   mode=budget 100/100
[5] 在线现算 18.4 µs/step   查表 0.086 µs/step (215x)
```
断言: 全部输出 `json.loads` 通过且 pair 数 ∈ [1,3]; 无约束合法率 ≤ 5%; budget 模式 100% 闭合而
仅 mask 模式会被截断; 查表比现算快。耗时数字随机器浮动, 量级不变。

## 与真实系统的差距
- 本 grammar 是正则语言 (无嵌套) → 纯 DFA。真实 JSON Schema 有嵌套, 需要下推自动机 (栈):
  xgrammar 只对"与栈无关"的 token 预编译, 其余 token 运行时查; outlines 把 regex 编译成 DFA。
- 真实表是 S × 128k 的 bitmask, 编译要几百 ms~数秒, 所以按 schema 缓存, 且与 GPU 前向重叠执行。
- **max_tokens**: vLLM / OpenAI 等默认**不**强制收尾 —— 撞上 max_tokens 就截断, 返回
  `finish_reason="length"`, JSON 不完整, 由调用方检查重试 (即 [4] 的 mode=mask)。本模块的
  `need` 表是"预算内必闭合"的做法, 代价是结尾被硬掰, 内容可能不是模型想说的。
- batch 内每条请求各有自己的 FSM 状态, mask 要拼成 (B,V) 一次性 apply 到 GPU logits 上。

## 常见误区
- "mask 保证内容正确": 只保证**语法**合法; 随机权重下 key 全是乱码, 照样 100% 通过。
- "token 合法 = 首字符合法": 必须整段字符都走通。`true` 在 value 起始处合法, 在 key 里还剩 ≥4
  字符额度时也合法 (当作 4 个字母), 额度不够就非法 —— demo 里它在 24 个状态下合法。
- "约束不改变分布": mask 后重新归一化, 会放大模型本来很小的概率; 同一字符串有多种分词
  (`{"` vs `{`+`"`), 强制走模型不习惯的分词会伤质量 (token healing 要解决的问题)。
- 容易写错的地方: value 写完后能否接 `,`/`}` 的判断, 若 string 分支用 `n_pairs`、number 分支用
  `n_pairs+1` → string 结尾时被迫 ≥2 对, 且能写出 max_pairs+1 对。现统一到 `_separators()`。

## 自测题
1. 为什么 `next_state` 必须和 `mask_table` 一起预编译?
   答: 在线若还要逐字符走 token 来更新状态, 又回到 O(len) 且依赖字符级 FSM; 存下落点后每步只有两次查表。
2. 词表 128k、DFA 1000 个状态, mask 表用 bitmask 存多大? 为什么可接受?
   答: 1000×128k bit ≈ 16 MB; 按 schema 编译一次并缓存, 远小于模型权重。
3. 只做 mask 不做预算控制, max_tokens 到了会怎样? 真实系统怎么办?
   答: 输出是合法**前缀**但不是完整 JSON (demo 里 67% 被截断); 真实系统返回 finish_reason=length 让调用方处理, 或像 `need` 表那样在预算将尽时只放行能及时收尾的 token。

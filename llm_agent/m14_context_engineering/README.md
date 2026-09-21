# M14 — Context Engineering

把"每次调用到底放什么进上下文"当成工程问题: 便宜的先清, 贵的再压, 能现取的不预存, 要跨会话的写文件。

## 直觉

Agent loop 每一轮都要把整个上下文重发给模型, 所以上下文只增不减意味着: 你一直在为几十轮前的工具输出付费, 模型的注意力被稀释, 最后撑爆窗口。
"爆了再截断"是最差的补救: 它会把最早的用户目标和 tool_use / tool_result 的配对结构一起切坏。
更好的做法是分档降级: 先零成本地清掉旧工具结果的正文, 还不够再花一次模型调用写摘要, 并且真的用摘要替换历史。
另外两条腿是"不放进来": 上下文里只放文件索引、内容用到才读 (just-in-time); 跨会话的知识写进文件, 不占任何一轮的上下文。

## 核心数据结构与控制流

| 机制 | 位置 |
|------|------|
| 组装与分档 | `core/agent.py`: `Agent._assemble_context()` / `Agent._compact()` |
| 三个瘦身函数 | `core/memory.py`: `clear_tool_results` / `summarize_with_llm` / `truncate_messages` |
| 压缩边界 | `core/persistence.py`: `JsonlSessionStore.append_compact()` / `load()` / `load_all()` |
| 压缩前 hook | `core/hooks.py`: `pre_compact` 事件, 返回"摘要里必须保留什么" |
| 记忆工具 / 按需读文件 | `core/sandbox.py`: `MemoryTool` / `ReadFileTool` / `confine()` |
| toy 摘要器 | `core/toy_llm.py`: `RuleBasedLLM._summarize()` (滚动合并上一份摘要) |

每次问模型之前 (`_assemble_context`):

```
budget = context_budget_chars - len(system prompt + 记忆)
view = self.messages
超预算? ─ 否 ─→ 原样发送
   │是
第 1 档  view = clear_tool_results(view, keep_last=1)      只改视图, transcript 与磁盘不动
   │仍超
第 2 档  compaction == "summary" → _compact():
           start = 最后一条真正的用户 prompt 的下标
           old, tail = messages[:start], messages[start:]   当前轮原样保留
           keep = hooks.on_pre_compact(old)
           summary = summarize_with_llm(llm, old, keep)     再调一次模型
           self.messages = [summary] + tail                 内存里的历史被替换
           store.append_compact(summary, kept=len(tail))    JSONL 追加 compact_boundary
         compaction == "truncate" → truncate_messages(view) 仅视图, 反例
```

设计取舍:

- 第 1 档只改视图: 清理是可逆的、零模型开销的, 占位符 `[cleared: N chars]` 保留了配对结构; 原文仍在 transcript 里, 审计与摘要都还能看到。
- 第 2 档必须替换历史并落盘边界: 否则每一轮都要重新压一次, 而且 resume 读回来的仍是完整历史。`load()` 回放到 boundary 时把视图重置为"摘要 + 最后 kept 条", `load_all()` 跳过 boundary 返回全部 —— 文件只增不减, 恢复出来的上下文却真的变小。
- 切点选在"当前用户轮"之前: `tail` 以用户 prompt 开头, 本轮的 tool_use 与 tool_result 都在 tail 里, 配对永远不会被切断。
- 摘要由模型写, 人通过 `pre_compact` hook 指定必留信息: 什么重要是任务相关的, harness 猜不出来。
- `MemoryTool` 沿用 Claude API memory tool 的 `/memories` 目录约定: 路径必须是 `/memories` 或以 `/memories/` 开头, 围栏立在 `root/memories` 上 (`confine()` 先 `resolve()` 再判断), 所以 `/memories/../x` 也出不去。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m14_context_engineering.demo
```

```
[1] 5 轮检索, 预算 900 字符: 清理 → 压缩 逐级触发
  [compact summary of 5 messages]
  目标: 检索 kv cache; 检索 lora; 检索 dpo; 检索 sampling
  已完成: search_docs(检索 kv cache); search_docs(检索 lora); search_docs(检索 dpo); search_docs(检索 sampling)
  关键结果: search_docs: kv_cache: kv cache key fact: FACT-1. long backg; search_docs: lora: ...
  保留: 用户只关心 FACT 编号
  compactions             : 3
  peak context tokens     : 246

[2] 压缩真的缩小了'恢复出来的会话', 审计日志一条没少
  resume 视图               : 5 条 / 1026 字符
  审计全量                    : 20 条 / 3049 字符

[3] 对照: truncate 策略只裁剪视图 —— 每轮重新裁, 历史和 resume 都不会变小
  truncate resume 视图      : 20 条 (没有变小)
  truncate 下答非所问的轮数       : 3/5

[4] 即时检索 vs 预加载
  预加载: 每次调用都要带的 tokens    : 331
  即时检索: 峰值上下文 tokens      : 107
```

`assert` 验证的内容:

- [1] 模型实际看到的某个上下文里出现过 `[cleared:` (第 1 档生效); `compactions >= 1` 且 `pre_compact` hook 的触发次数等于压缩次数; `messages[0]` 是 `compact_summary`, 其中既有最早的目标"检索 kv cache", 也有 hook 要求保留的那句话。
- [2] `load()` 的字符数小于 `load_all()` 的 60%; 审计全量恰好 20 条 (5 轮 × 4 条); 磁盘恢复出的视图与内存里压缩后的 `agent.messages` 逐条相同; 恢复视图通过 `validate_transcript` (没有孤儿 tool_result); 用它 resume 的新 agent 还能正常完成一次检索。
- [3] `truncate` 下 `compactions == 0`, 且 `load()`、`load_all()`、内存历史三者长度相同 —— 截断只影响单次视图, 不缩小持久化与恢复出来的历史; 并且断言 truncate 会话里出现了答非所问的轮次 (tool_result 被拍平成 user 文本), 而 summary 会话没有。
- [4] 即时检索拿到了 FACT-2, 且峰值上下文小于预加载全文 token 数的一半。注意 331 是对"全文塞进 system prompt"的估算值, demo 没有真的跑一个预加载 agent。
- [5] 会话 1 在磁盘上写出了 `memories/notes.md`; 全新的会话 2 (4 条消息, 无任何历史) 通过 `memory view` 读到了 "pytest"。

## 与真实系统的差距

- 预算按字符计, `estimate_tokens` 是粗估 (英文约 4 字符/token, 中文约 1 字/token)。真实系统用 API 的 token 计数 (`messages.count_tokens`) 和响应里的 `usage`。
- Claude API 提供服务端的 context editing (`clear_tool_uses`) 与 compaction; 这里是客户端自己做, 触发阈值、保留策略都简化到一个字符预算。
- 没有 prompt caching。真实系统里改写前缀 (清理、压缩) 会让缓存失效, 清理的收益要和缓存损失一起算; 本仓库看不到这笔账。
- toy LLM 的"摘要"是按结构抽取 (用户 prompt、工具调用名与参数、助手结论或工具结果的前 60 字符), 不是真的读懂后重写; 摘要质量问题 (丢关键细节、幻觉) 在这里观察不到。
- 压缩会把 `session_start` 消息本身摘要掉, 但其文本会和 `pre_compact` hook 的要求一起作为"必须保留"交给摘要; 下次 resume 发现历史里没有 `session_start` 时会重新注入。Claude Code 是在 compact 之后立刻重跑 SessionStart hook, 这里没有这一步。
- `MemoryTool` 只实现 view / create / str_replace / delete, 参数名也简化了 (`text` / `old`); 真实 memory tool 还有 insert、rename。没有记忆的过期、冲突与投毒防护。
- 即时检索只有"按路径读文件"一种; Claude Code 靠 glob / grep 先定位再读, 并对大文件分页。

## 常见误区

- "清理工具结果会破坏 transcript。" 不会: `clear_tool_results` 返回新列表, 只替换发给模型的视图里旧 tool_result 的正文, 内存历史和 JSONL 原文都不动, 配对结构也保留。
- "截断也是一种压缩, 只是粗糙一点。" 两者效果不同类: `truncate` 每轮重裁视图, 历史与 resume 永远不变小; 它还把消息拍平成文本、丢掉 block 结构 (自己打印 truncate 会话的回答会看到后几轮退化成"无需工具的直接回答")。`summary` 替换历史并落盘边界, 是一次性的、可恢复的。
- "压缩之后旧消息就没了, 无法审计。" JSONL 是 append-only, 压缩只追加一条 `compact_boundary`; `load_all()` 仍返回全部 20 条, 变小的只是 `load()` 给 resume 的视图。

## 自测题

1. 压缩时为什么以"最后一条真正的用户 prompt"为切点, 而不是"保留最后 N 条消息"?

<details><summary>答案</summary>
按条数切可能正好落在 assistant 的 tool_use 和随后 user 的 tool_result 之间, 留下孤儿 tool_result —— 真实 API 会直接拒绝这种序列, `validate_transcript` 也会报错。以用户 prompt 为切点时, tail 必然以一条用户输入开头, 本轮所有 tool_use / tool_result 成对留在 tail 里; 同时当前任务的原话逐字保留, 不经过有损摘要。`is_user_prompt` 专门排除了携带 tool_result 的 user 消息, 就是为了找对这个切点。
</details>

2. 进程在第 4 轮压缩后崩溃。新进程用 `load_history=True` 恢复时读到什么? `store.load()` 是如何从一个只增不减的文件里算出变小的视图的?

<details><summary>答案</summary>
`load()` 顺序回放记录: 普通行追加进 view; 遇到 `compact_boundary` 就把 view 重置为 `[summary] + view[-kept:]` (kept 是压缩当时原样保留的尾部条数, 即当前轮已写入的消息), 之后继续追加。多次压缩就多次重置, 最终得到"最后一份摘要 + 当轮尾部 + 其后的消息", 与崩溃前内存里的 `agent.messages` 一致 (demo 断言了逐条相等)。tool_use 的 id 续号用的是 `load_all()` 的全量计数, 所以压缩后新 id 也不会和旧的撞号。
</details>

3. 某团队说"我们的知识库只有 300 token, 直接放进 system prompt 比做即时检索省事"。这个判断在什么条件下成立, 什么条件下会反噬?

<details><summary>答案</summary>
成立条件: 内容小而稳定、几乎每个任务都用得到 —— 常驻的代价是"每次调用 × 300 token", 换来省掉一次工具往返。反噬条件: 知识库会增长, 或大部分任务只用到其中一小块; 常驻成本随 调用次数 × 全文大小 线性增长, 还稀释注意力。即时检索把常驻成本降为一份索引 (路径列表), 只在用到时付一次读取的钱, 读进来的内容之后还能被第 1 档清理掉。demo 里 5 篇文档时是 331 对 107; 文档数增长时差距线性扩大。
</details>

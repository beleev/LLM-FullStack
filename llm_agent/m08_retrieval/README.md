# M08 — Retrieval (RAG-lite): 知识放在上下文之外, 按需取 top-k

用纯 stdlib 实现 TF-IDF 稀疏向量 + 余弦相似度检索 (支持中文), 并以同名工具 `search_docs` 热替换掉关键词计数版, agent loop 零改动。

## 直觉

知识库装不进上下文, 全塞进去也贵; 不给知识, 模型就凭记忆编。检索的做法是把知识留在外面, 每次只取与当前问题最相关的 k 条。
难点在"相关"怎么算。最朴素的关键词计数给每个命中词 1 分, 于是一篇凑满 how / does / the / model 的文档能挤掉唯一提到 fragmentation 的正确答案。
TF-IDF 的修正是: 几乎每篇都有的词权重趋近 0, 罕见词主导排序。
另一个坑是分词: 只认 `[a-z0-9]+` 的分词器会把中文整段丢掉, 中文查询得分恒为 0。
检索错了, agent 后面的推理再好也是在错误证据上推理。

## 核心数据结构与控制流

- `core/utils.py: tokenize` — 英文 / 数字按词 (先 lower); 连续 CJK 字符 (U+4E00–U+9FFF) 切成字符 bigram, 单字则保留该字。
- `core/retrieval.py: TfidfIndex` — 构建时对每篇 `title + body` 分词, 统计 df, 算出 `idf` 字典和每篇文档的归一化稀疏向量 `_doc_vec`。
- `TfidfIndex.embed(text)` — 把任意文本变成同一空间的向量; 这是换成神经 embedding 时唯一要替换的点。
- `TfidfIndex.search(query, k)` — 返回 `[(score, title)]`。
- `core/retrieval.py: VectorSearchTool` — `name = "search_docs"`, schema 与 `core/tools.py: SearchDocsTool` 完全相同; 输出 `[score] title: body`, 只保留 score > 0 的命中。

```
建索引:  docs ─tokenize→ tf(每篇), df(全局)
         idf(w)    = ln(1 + (N - df + 0.5) / (df + 0.5))        # BM25 同款
         weight(w) = (1 + ln tf) * idf(w)                       # 亚线性 tf
         doc_vec   = weight / ||weight||                        # L2 归一化
查询:    query ─tokenize→ 同样加权、丢弃语料外的词、归一化 → q
         score(doc) = Σ_w q[w] * doc_vec[w]                     # 两边已归一化, 稀疏点积 = 余弦
         按 score 降序取前 k
```

关键设计决策:

- **亚线性 tf `1 + ln(tf)`**: 同一个词重复 3 次不该有 3 倍话语权, 否则堆砌关键词的文档占便宜。
- **BM25 式 idf**: 加 0.5 平滑并包一层 `ln(1 + ·)`, 保证恒为正; 8 篇里出现 6 篇的 `the` 只有 0.33, 只出现 1 篇的 `fragmentation` 是 1.79。
- **字符 bigram 而非中文分词器**: 零依赖、无词典, 是 Lucene CJKAnalyzer 的同款折中; 代价是产生"片怎""么解"这类跨词边界的无意义 token; 它们多半不在语料词表里, 会被当作语料外的词丢掉, 偶尔碰巧命中则成为噪声。
- **工具名和 schema 不变**: 模型侧看到的仍是 `search_docs(query)`, 所以检索质量升级对 loop、权限规则、prompt 全透明。关键词 → TF-IDF → 神经 embedding 走的是同一个接口。

## 运行后应该看到什么

```bash
cd <仓库根目录> && python3 -m llm_agent.m08_retrieval.demo
```

```
[1] 同一查询, 两种排序
  query                   : how does the model manage memory fragmentation
  关键词命中数                  : {'paged_attention': 4, 'kv_cache': 5, 'faq': 5, 'lora': 5, 'dpo': 5, 'sampling': 4}
  关键词计数 top-1             : lora
    tf-idf [0.33] paged_attention
    tf-idf [0.20] faq
    tf-idf [0.16] kv_cache

[2] 为什么: idf 给每个词的权重
    idf(the          ) = 0.33
    idf(model        ) = 0.33
    idf(how          ) = 0.49
    idf(memory       ) = 1.28
    idf(fragmentation) = 1.79

[3] 中文查询: bigram 分词
  旧分词 [a-z0-9]+           : []
  bigram                  : ['显存', '存碎', '碎片', '片怎', '怎么', '么解', '解决']
  top-1                   : [0.30] 分页注意力

[4] 热替换进 agent: 工具名不变, loop 零改动
  [m08] turn 1: model -> tool_use toolu_0001 search_docs {'query': '检索: how does the model manage memory fragmentation'}
  [m08] tool_result toolu_0001 -> [0.33] paged_attention: paged attention fixes memory fragmentation in the kv cache with...
```

断言验证的内容:

- [1] 关键词计数下正确答案 `paged_attention` 的命中数 (4) 低于 top-1 (5), 且 top-1 不是它; TF-IDF 下它排第一, 且分数超过第二名的 1.5 倍。
- [2] `idf["fragmentation"] > 4 * idf["model"]`。
- [3] 旧分词对中文查询返回空列表; bigram 分词后 top-1 是 `分页注意力` 且分数 > 0。
- [4] agent 通过 `search_docs` 拿到的第一行结果以 `[分数]` 开头并包含 `paged_attention`。

注意: 关键词版里 `kv_cache / faq / lora / dpo` 四篇并列 5 分, `lora` 排第一只是因为 `(score, title, body)` 元组逆序排序时标题字母序最大 —— 并列时的名次是偶然的, 这本身也是等权计分的毛病。

## 与真实系统的差距

- 仍是词面匹配: 同义词、改写、跨语言都匹配不上 (英文查询 "memory fragmentation" 找不到中文的"显存碎片"那篇)。生产 RAG 用稠密 embedding 解决语义匹配, 常与 BM25 做混合检索。
- 没有 ANN 索引: `search` 对所有文档逐篇算点积, O(N); 连倒排索引都没有。规模上去后需要 HNSW / IVF 之类的近似最近邻或搜索引擎。
- 没有 reranker: 生产系统常先召回几十条, 再用 cross-encoder 或 LLM 精排取 top-k。
- 没有 chunking: 这里一篇文档就是一句话, 整篇返回。真实文档要切块、处理重叠与元数据, 返回的是片段加出处。
- 只借了 BM25 的 idf, 不是 BM25: 没有 k1 饱和参数和 b 长度归一化, 用的是余弦归一化。
- 索引是静态的内存对象: 增删文档要整体重建 (df 变了所有 idf 都变), 没有持久化和增量更新。
- 没有词干化 / 停用词 / 拼写容错 (`manage` 与 `manages` 是两个词); CJK 范围只覆盖 U+4E00–U+9FFF, 假名、谚文等会被直接丢弃。
- 没有检索评测 (recall@k、MRR), 只有一条手工查询; 也没有"分数过低就回答不知道"的阈值, 只过滤 score = 0。
- `VectorSearchTool` 的 `untrusted_output` 是默认的 False: 检索回来的文档没有被当作不可信数据标记, 而真实 RAG 的语料正是 prompt injection 的常见入口 (m12)。

## 常见误区

- **"向量检索就等于语义检索"** — TF-IDF 也是向量 + 余弦, 但维度是词表, 只能匹配字面相同的 token。"语义"来自神经 embedding, 而不是来自"用了向量"。
- **"命中的查询词越多, 文档越相关"** — 这正是关键词计数翻车的原因: 命中 5 个烂大街的词不如命中 1 个罕见词。相关性取决于词的区分度 (idf), 不是命中个数。
- **"TF-IDF 分数只有 0.33, 说明匹配得很差"** — 余弦分数受文档向量长度影响, 文档里与查询无关的词越多分数越低; 它只用于同一查询下的排序比较, 不是可跨查询比较的置信度。

## 自测题

1. 查询里有 `the`、`model` 这类每篇都有的词, 为什么几乎不影响 TF-IDF 排序, 却能主导关键词计数的排序?
<details><summary>答案</summary>

关键词计数里每个命中词固定 1 分, 常见词和罕见词等权, 于是凑齐 how / does / the / model / manage 的文档拿到 5 分, 压过只命中 4 个词但含 fragmentation 的正确答案。TF-IDF 里词的权重乘了 idf: df 越大 idf 越接近 0 (`the` 0.33 vs `fragmentation` 1.79), 而且查询向量和文档向量两边都乘 idf, 点积里常见词的贡献按平方级缩小。

</details>

2. agent 实际发出的查询是 `检索: how does the model ...`, 多出来的"检索"两个字为什么没有拉低排序质量?
<details><summary>答案</summary>

`tokenize` 把"检索"切成一个 bigram `检索`, 它不在语料词表里。`_vectorize` 只保留 `w in self.idf` 的词, 语料外的词没有方向, 在归一化之前就被丢弃, 所以查询向量与不带前缀时完全相同, 分数仍是 0.33。

</details>

3. 要把本模块升级成稠密语义检索, 需要改哪些地方, 哪些不用动?
<details><summary>答案</summary>

要改的是 `TfidfIndex.embed` (换成调用 embedding 模型, 返回稠密向量) 以及相应的文档向量构建和点积实现 (稀疏 dict 换成稠密数组, 规模大时再加 ANN 索引)。不用动的是 `search()` 的返回形状、`VectorSearchTool` 的名字与 schema、agent loop、权限规则和模型侧的任何 prompt —— 因为它们只依赖 `search_docs(query) -> 文本` 这个接口。

</details>

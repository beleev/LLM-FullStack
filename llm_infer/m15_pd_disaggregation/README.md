# M15 — Prefill / Decode Disaggregation: 两类负载分到两组节点

## 直觉
prefill 一次吃几千 token, **算力瓶颈**, 用户关心 TTFT; decode 一次 1 token, **访存瓶颈**, 用户关心 ITL。
同卡混跑时, 一个长 prefill 插进来, 同 batch 所有 decode 请求的这一步都要等它算完 → ITL 抖动。
chunked prefill (m06) 是把 prefill 切碎来缓解; P/D 分离是干脆分家: P 节点只 prefill, 算完把 KV cache
通过网络发给 D 节点, D 节点只 decode。两边可以各自选并行策略、batch 大小、甚至不同型号的卡。代价: **KV 要过网络**。

## 核心数据结构或公式
```
PrefillNode.run(prompt) → first_token, (shape, dtype, payload: bytes)     KV 堆成 (n_layer, 2, T, D) 再 tobytes
KVLink(link_gbps).send(payload)                                           代价模型, 不 sleep
DecodeNode.run(first_token, wire_kv, max_new) → 后续 token                 np.frombuffer 还原, 接着 decode_step
```
- KV 字节数 `= 2 · n_layer · T · D · sizeof(dtype)`。
- **单位**: 链路按 Gbps (bit) 标, KV 按 byte 算 → `bytes_per_s = link_gbps · 1e9 / 8`,
  `transfer_ms = latency_ms + nbytes / bytes_per_s · 1e3`。(NVLink 例外, 厂商按 GB/s 标。)
- 首 token 由 P 节点产出 (TTFT 不含传输); 传输时间加在**第 2 个 token** 的延迟上。

## 运行后应该看到什么
```bash
python -m llm_infer.m15_pd_disaggregation.demo      # ~0.2 s
```
```
[1] 单机 baseline 生成 = [103, 33, 126, 113, ...]   P→D 分离 生成 = 完全相同 (16/16 token)
[2] 公式 2·4·48·64·4 = 98304 B = 实际 nbytes 98304 B = 线上 payload 98304 B
[3] LLaMA-7B fp16, T=4096: KV = 2.147 GB (2.00 GiB)                       [公式计算, 非实测]
      100GbE / RoCE            100 Gbps   12.5 GB/s   171.8 ms
      IB NDR 400G              400 Gbps   50.0 GB/s    42.9 ms
      8×400G (DGX H100 节点间)  3200 Gbps  400.0 GB/s     5.4 ms
      NVLink4 单向 (节点内)     3600 Gbps  450.0 GB/s     4.8 ms
    400 Gbps: 正确 42.9 ms / 把 Gbps 当 GB/s 5.4 ms (低估 8x)
[4] TinyLM 实测: 400-token prefill 4.36 ms, decode step 0.250 ms → 混跑时被插队那一步 ITL 4.61 ms (18x 抖动)
```
断言: 两个节点的权重不共享内存、链路上传的是 `bytes`; 分离输出与 `generate_greedy` 逐 token 相同;
公式字节数 == nbytes == payload 长度 == 链路计数; 400 Gbps == 50e9 B/s, 错误换算恰好低估 8×; prefill > 5× decode step。
[3] 是公式算出来的代价模型, 不是测出来的; [4] 的毫秒数随机器变化。

## 与真实系统的差距
- 两个 "节点" 是同一进程里的两个对象 (各自 `TinyLM(cfg)`), 没有真实网络; 序列化是一次性整块 `tobytes`。
  Mooncake / DistServe **按层流式传输**: 第 l 层 prefill 算完就发, 与后续层的计算重叠, 传输几乎被藏进 prefill 时间里。
- 真实传输走 GPUDirect RDMA / NVLink, 直接写进 D 节点的 paged KV block (vLLM `KVConnector` + NIXL / LMCache,
  SGLang PD), 不经过 CPU bytes; 还要处理 P、D 两边 TP 度不同时的 KV 重新切分。
- 没有调度器: 真实系统要决定 P:D 节点配比、D 节点的 KV 显存准入、短 prompt 是否值得分离 (传输 > 重算时不如本地 prefill)。
- GQA / MLA 会让 KV 小 4–8× 甚至更多 (LLaMA-3-8B 同上下文只有 ~0.5 GiB), 传输压力相应下降。

## 常见误区
- **Gbps 当 GB/s**: 写成 `bytes / (gbps·1e9)` 就少除了 8, 传输时间低估 8×。网卡/IB 永远按 bit 标。
- "分离能提高单请求速度": 不能, 单请求还多了一次传输。收益是**消除干扰** → 同样的 SLO 下每张卡能承载更多请求 (goodput)。
- "传输时间算进 TTFT": 首 token 是 P 节点 prefill 的产物, 可以直接回给用户; 传输影响的是第 2 个 token。
- "P、D 节点可以用不同权重精度": KV 是 P 节点的权重算出来的, D 节点权重不一致会让后续 token 悄悄跑偏 — 本 demo 两边同 seed 才能逐 token 相同。

## 自测题
1. **LLaMA-7B fp16、8k 上下文的 KV 过 200 Gbps 链路要多久?**
   2·32·8192·4096·2 = 4.29 GB; 200 Gbps = 25 GB/s → 约 172 ms。
2. **什么负载下 P/D 分离最划算, 什么时候不划算?**
   长 prompt + 严格 ITL SLO、且流量足以喂饱两组节点时最划算; 短 prompt (传输+调度开销 > 干扰损失) 或流量很低 (两组节点都闲着) 时不划算。
3. **为什么 demo 要让两个节点各自 `TinyLM(cfg)` 而不是共用一个 lm 对象?**
   共用对象时 "KV 传输" 可以悄悄退化成传引用, 测不出序列化/反序列化的 bug; 各自加载权重 + 只传 bytes 才是对真实部署的忠实模拟。

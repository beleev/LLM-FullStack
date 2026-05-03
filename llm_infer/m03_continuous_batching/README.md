# M03 — Continuous Batching: 调度器把 GPU 喂饱

## 问题
传统 "static batching" 把 N 条请求拼成一个 batch, **等所有请求都完成才出下一批**。
最长的请求拖死整批, GPU 利用率惨淡。

```
  static batching 时间线:
    seq A ──prefill─decode─decode─decode─[done]
    seq B ──prefill─decode─[done]                    ← B 早就好了
    seq C ──prefill─decode─decode─decode─decode─[done]
                                            ↑
                          整批必须等 C 完成才能切下一批 (idle B 浪费)
```

## 解法: continuous batching (in-flight batching)
**每一步 (step) 都重新组 batch**: 已完成的退出, 新到达的加入。
搭配 paged KV (m02), 序列长度不齐也无压力。

```
  continuous batching 时间线:
    step 0:  [A, B, C]  全部 prefill
    step 1:  [A, B, C]  decode
    step 2:  [A, B, C]  decode
    step 3:  [A, C, D]  ← B 完成退出, D 新到加入
    step 4:  [A, C, D, E]
    ...
```

## 调度策略
本模块实现 **prefill-priority** 策略 (vLLM/SGLang 默认):
1. **prefill 阶段** 优先: 把 waiting 队列中的新请求 prefill 满, 直到
   达到 `max_batch_tokens` 或 block 耗尽
2. **decode 阶段**: 处理 running 队列, 每条 +1 token
3. **preempt** (抢占): 若 decode 时新 token 需要新 block 但 pool 满,
   把"最年轻"的序列踢回 waiting (释放它的 block), 等下次 prefill 重算

## 接口
- `Sequence`: 序列状态 (WAITING/RUNNING/FINISHED)
- `Scheduler.add_request(prompt_ids)`: 用户提交请求
- `Scheduler.schedule()`: 给一组本步要执行的序列 + 阶段标记
- `Scheduler.postprocess(outputs)`: 把生成的 token 写回, 检查是否结束

## 运行
```bash
python -m llm_infer.m03_continuous_batching.demo
```

输出会演示:
- prefill / decode 切换
- 不同长度请求的并发执行
- pool 紧张时的抢占

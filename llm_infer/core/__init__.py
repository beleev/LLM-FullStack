"""
core — 各模块共享的极简 transformer + tokenizer。

为什么要有这个 core？
    每个推理优化模块都需要"一个能跑的模型"做实验, 但实验目标各不相同
    (KV cache 看显存; 量化看精度; spec decode 看加速比 ...)。
    把模型和 tokenizer 集中到 core, 让每个 mXX 只关注自己要演示的优化点。

模块清单
    - tiny_model.py      4 层 1 头 d=32 的极简 LM (numpy, 仅前向)
    - tokenizer.py       字符级 tokenizer
    - utils.py           softmax / RMSNorm / dense_attention (全库唯一的朴素 attention 基线) / 计时器
    - sequence.py        Sequence / SeqStatus / Stage: 一条请求的状态 (调度器与引擎共用)
"""
from llm_infer.core.tiny_model import TinyLM, ModelConfig, truncate_kv
from llm_infer.core.tokenizer import CharTokenizer
from llm_infer.core.utils import softmax, rms_norm, dense_attention, causal_mask, Timer
from llm_infer.core.sequence import Sequence, SeqStatus, Stage

__all__ = ["TinyLM", "ModelConfig", "truncate_kv", "CharTokenizer", "softmax", "rms_norm",
           "dense_attention", "causal_mask", "Timer", "Sequence", "SeqStatus", "Stage"]

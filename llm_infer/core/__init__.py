"""
core — 各模块共享的极简 transformer + tokenizer。

为什么要有这个 core？
    每个推理优化模块都需要"一个能跑的模型"做实验, 但实验目标各不相同
    (KV cache 看显存; 量化看精度; spec decode 看加速比 ...)。
    把模型和 tokenizer 集中到 core, 让每个 mXX 只关注自己要演示的优化点。

模块清单
    - tiny_model.py      4 层 1 头 d=32 的极简 LM (numpy, 仅前向)
    - tokenizer.py       字符级 tokenizer
    - utils.py           softmax / RMSNorm / 计时器等小工具
"""
from llm_infer.core.tiny_model import TinyLM, ModelConfig
from llm_infer.core.tokenizer import CharTokenizer
from llm_infer.core.utils import softmax, rms_norm, Timer

__all__ = ["TinyLM", "ModelConfig", "CharTokenizer", "softmax", "rms_norm", "Timer"]

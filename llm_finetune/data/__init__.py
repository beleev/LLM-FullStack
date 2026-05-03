"""
合成数据生成器子包 (微调专用)。

与 ``llm_models.training.data`` 的 ``DecoderOnlyDataGenerator`` (预训练用)
的核心差异:
    - InstructionDataGenerator: 模拟 (prompt, response) 配对, prompt 区 label = -100
    - PreferenceDataGenerator:  模拟 (prompt, chosen, rejected) 三元组
"""

from llm_finetune.data.instruction_data import InstructionDataGenerator
from llm_finetune.data.preference_data import PreferenceDataGenerator

__all__ = [
    "InstructionDataGenerator",
    "PreferenceDataGenerator",
]

"""合成数据: 一个可学习的任务 (tasks.SeqTask) + 三种取数方式 (指令 / 偏好对 / 只有 prompt)。"""

from llm_finetune.data.tasks import SeqTask, make_labels, completion_mask, PAD, SEP, EOS
from llm_finetune.data.instruction_data import InstructionDataGenerator
from llm_finetune.data.preference_data import PreferenceDataGenerator
from llm_finetune.data.prompt_data import PromptDataGenerator

__all__ = [
    "SeqTask", "make_labels", "completion_mask", "PAD", "SEP", "EOS",
    "InstructionDataGenerator", "PreferenceDataGenerator", "PromptDataGenerator",
]

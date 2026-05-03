"""
sparse — 稀疏 / 非注意力序列分支

  moe: MixtralMoE (经典稀疏 MoE FFN; DeepSeekMoE 在 models/moe/deepseekV3.py)
  ssm: SelectiveSSM (Mamba 核心, 非注意力序列混合)
"""

from llm_models.layers.sparse.moe import MixtralMoE
from llm_models.layers.sparse.ssm import SelectiveSSM

__all__ = ["MixtralMoE", "SelectiveSSM"]

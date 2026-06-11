"""
工具函数模块

汇总并对外暴露通用工具，目前主要是注意力机制需要的各类掩码（mask）：
- get_pad_mask: padding 掩码，屏蔽 pad token
- get_subsequent_mask / build_causal_mask: 因果掩码，自回归模型防偷看未来
- build_sliding_window_mask: 带状因果掩码（SWA，Mistral / Gemma / GPT-OSS）
- combine_causal_and_padding_mask / combine_masks: 多种掩码的组合
"""

from llm_models.utils.masks import (
    get_pad_mask,
    get_subsequent_mask,
    build_causal_mask,
    build_sliding_window_mask,
    combine_causal_and_padding_mask,
    combine_masks,
)

# __all__ 显式声明 from llm_models.utils import * 时导出的符号
__all__ = [
    "get_pad_mask",
    "get_subsequent_mask",
    "build_causal_mask",
    "build_sliding_window_mask",
    "combine_causal_and_padding_mask",
    "combine_masks",
]

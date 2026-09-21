"""llm_train 各 demo 共用的原语: 通信、玩具模型、数值格式、小工具。"""

from .collectives import (
    all_gather, all_reduce_mean, all_reduce_sum, all_to_all, comm,
    reduce_scatter_sum, ring_all_reduce_sum, ring_shift,
)
from .numerics import FLOAT_FORMATS, fake_quant_float, quant_blockwise
from .toy_model import LinearModel, MomentumSGD, ToyDataStream, adam_update
from .utils import (
    average_grad_trees, banner, bytes_of, clip_by_global_norm, flatten_tree, global_norm,
    has_overflow, kv, load_checkpoint, make_rng, max_abs_diff, relu, save_checkpoint,
    softmax, unflatten_like,
)

__all__ = [
    "FLOAT_FORMATS", "LinearModel", "MomentumSGD", "ToyDataStream", "adam_update",
    "all_gather", "all_reduce_mean", "all_reduce_sum", "all_to_all", "average_grad_trees",
    "banner", "bytes_of", "clip_by_global_norm", "comm", "fake_quant_float", "flatten_tree",
    "global_norm", "has_overflow", "kv", "load_checkpoint", "make_rng", "max_abs_diff",
    "quant_blockwise", "reduce_scatter_sum", "relu", "ring_all_reduce_sum", "ring_shift",
    "save_checkpoint", "softmax", "unflatten_like",
]

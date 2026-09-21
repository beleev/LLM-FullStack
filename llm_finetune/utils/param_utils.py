"""参数冻结 / 统计: 每种方法都要回答 "现在到底哪些参数在训"。"""

from typing import Dict, Optional

import torch.nn as nn


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """{"total", "trainable", "frozen"}; parameters() 对共享权重 (tied embedding) 只计一次。"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def freeze_module(module: nn.Module) -> None:
    """原地 requires_grad=False。用于 LoRA 的基座、DPO 的 ref、蒸馏的 teacher。"""
    for p in module.parameters():
        p.requires_grad = False


def print_trainable_parameters(module: nn.Module, name: Optional[str] = None) -> Dict[str, int]:
    stats = count_parameters(module)
    pct = 100.0 * stats["trainable"] / max(1, stats["total"])
    print(f"[{name or module.__class__.__name__}] trainable: {stats['trainable']:,} / total: {stats['total']:,} ({pct:.2f}%)")
    return stats

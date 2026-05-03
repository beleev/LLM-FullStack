"""
参数管理工具
================

微调过程中常见的"挑选/冻结/统计可训练参数"操作集中在此, 避免散落到各个
微调方法中。

为什么这一层值得单独存在?
    - 全参 SFT、LoRA、DPO 都要回答同一个问题:
      "哪些参数现在是可训练的, 哪些被我冻结了?"
    - 教学场景需要打印对比 (例如 LoRA 让可训练参数从 100% 降到 0.5%),
      把这个统计封装成函数, 三个示例脚本就能复用同一行调用。
    - 与 ``llm_models.utils`` 中的 attention mask 工具职责清晰分离。
"""

from typing import Dict, Optional

import torch.nn as nn


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """
    统计模块中的总参数量与可训练参数量。

    返回 dict 而非 tuple, 方便扩展 (例如未来添加 "frozen" 字段) 时
    不破坏调用方代码 (开闭原则)。

    Args:
        module: 任意 nn.Module。

    Returns:
        dict 含三个键:
            - "total":     全部参数 (含冻结)
            - "trainable": requires_grad=True 的参数
            - "frozen":    total - trainable
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def freeze_module(module: nn.Module) -> None:
    """
    冻结一个模块下的所有参数 (in-place 设置 requires_grad=False)。

    用途:
        - LoRA: 冻结基座, 仅训练插入的低秩矩阵
        - DPO:  把 reference model 整体冻结, 防止它跟着 policy 一起更新
        - 部分 SFT: 例如只想 finetune 最后几层时, 先冻整个模型再解冻顶层

    设计选择: 直接修改 in-place
        参数张量本身不可变 (PyTorch 没有 immutable nn.Parameter), 而 requires_grad
        是张量的元数据。我们没有别的选择。函数名以动词命名提醒副作用。

    Args:
        module: 需要冻结的子模块或整个模型。
    """
    for p in module.parameters():
        p.requires_grad = False


def print_trainable_parameters(
    module: nn.Module,
    name: Optional[str] = None,
) -> Dict[str, int]:
    """
    打印模块的可训练参数占比, 同时返回统计 dict。

    教学场景中, 把 "全参 vs LoRA" 的参数量差异直观地输出到 stdout, 学习者
    一眼就能看到 PEFT 的核心卖点 (例如 0.5% 的可训练参数即可工作)。

    输出形如:
        [LLaMA-Mini] trainable: 32,768  / total: 5,123,456  (0.64%)

    Args:
        module: 模型或子模块。
        name:   显示用的标签, 默认为模块的类名。

    Returns:
        ``count_parameters`` 的统计 dict, 便于调用方进一步断言/序列化。
    """
    stats = count_parameters(module)
    label = name if name is not None else module.__class__.__name__
    total = stats["total"]
    trainable = stats["trainable"]
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"[{label}] trainable: {trainable:,}  / total: {total:,}  ({pct:.2f}%)"
    )
    return stats

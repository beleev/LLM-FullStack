"""
LoRA — Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
=====================================================================

核心思想:
    预训练得到的权重 W ∈ R^{d_out × d_in} 已经"足够好", 微调只需在它**附近**
    寻找一个增量 ΔW。论文经验观察: 这个 ΔW 的 **本征秩很低**, 用一对低秩矩阵
    就足以表达。

数学形式:
    原层:        y = W x
    LoRA 层:     y = W x + (α/r) · B A x
                 其中 A ∈ R^{r × d_in},  B ∈ R^{d_out × r},  r << min(d_in, d_out)

    训练时仅更新 A, B (W 冻结);
    推理时可把 W' = W + (α/r) BA 合并回去, 推理零额外开销。

参数量对比:
    全参微调:     d_out × d_in
    LoRA:        r × (d_in + d_out)
    举例 (d_in = d_out = 4096, r = 8):  16M → 65K (约 0.4%)

关键超参:
    r       秩, 通常 4/8/16; 任务越难/数据越大 r 越大
    alpha   缩放, 论文建议 α = 2r 或 α = r; 经 (α/r) 归一化后,
            改 r 不需要重新调学习率
    target_modules
            选哪些 nn.Linear 加 LoRA。LLaMA 社区惯例:
            - 最小化:  q_proj, v_proj           (论文起点, 性能/参数最优)
            - 推荐:    q_proj, k_proj, v_proj, o_proj  (本实现的默认)
            - 全注入:  attn 全部 + ffn 全部     (类 QLoRA, 偏向极致效果)

初始化要点 (来自论文 Sec. 4.1):
    A 用 Kaiming 正态, B 用全零。
    乘积 BA 在训练开始时严格为 0, 因此 forward 输出 = 原模型输出,
    保证微调起点 = 预训练终点, 不会一上来就破坏已有能力。
"""

import math
import re
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.utils.param_utils import freeze_module


# 默认目标模块: LLaMA 风格的 attention 投影名 (见 layers/core/attention.py)。
# 推理时这四个矩阵决定了 attention 的所有线性变换, 是 LoRA 性价比最高的位点。
DEFAULT_TARGET_MODULES: List[str] = ["w_q", "w_k", "w_v", "w_o"]


class LoRALinear(nn.Module):
    """
    低秩适配的 nn.Linear 包装器。

    内部结构:
        - base:    被包装的原始线性层 (W, b), 训练中保持冻结
        - lora_A:  R^{r × d_in},  Kaiming 正态初始化
        - lora_B:  R^{d_out × r}, 零初始化
        - 推理:    y = base(x) + (alpha/r) * dropout(x) @ A^T @ B^T

    为什么不直接继承 nn.Linear?
        包装 (composition) 而非继承的好处:
          1. 原 base.weight 作为 buffer-like 引用可被 ``merge_lora_weights``
             安全地修改, 不需要担心多重继承的方法解析顺序
          2. ``nn.Linear`` 子类要兼顾 ``bias=None``、init 等多种构造路径,
             包装后只需要操心适配器即可

    Args:
        base_layer:   现有的 nn.Linear (来自被微调的模型)
        r:            低秩维度
        alpha:        缩放分子, 实际缩放 = alpha / r
        dropout:      作用于输入的 dropout, 0 表示禁用
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r 必须为正, 当前 r={r}")

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # dropout 设为 nn.Identity() 避免分支判断, 训练时一定走相同代码路径
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        )

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A: 输入侧矩阵, Kaiming 正态; 与 nn.Linear 默认初始化一致, 保证激活方差稳定
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: 输出侧矩阵, 零初始化; 让 BA = 0 → 训练起点 = 原模型, 关键的"无害启动"
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 冻结基座权重: requires_grad=False, 仅 lora_A / lora_B 计入梯度
        # 注: 这里只冻结 base 这一层; 全局冻结由 mark_only_lora_as_trainable 负责
        for p in self.base.parameters():
            p.requires_grad = False

        # merged 标记用于推理路径优化: True 表示已把 BA 合并回 base.weight
        # 合并后 forward 与普通 nn.Linear 等价, 跳过适配器分支
        self.merged: bool = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 已合并: 走纯 base 分支, 与原 nn.Linear 同速
        if self.merged:
            return self.base(x)

        # base 与 adapter 分支并行: y = Wx + (alpha/r) * (B A) x
        # 不直接构造 BA 矩阵, 而是分两次 matmul, 避免实例化 d_out × d_in 临时大矩阵
        base_out = self.base(x)
        adapter_in = self.lora_dropout(x)
        # (x @ A^T) -> [..., r];  再 @ B^T -> [..., d_out]
        adapter_out = F.linear(F.linear(adapter_in, self.lora_A), self.lora_B)
        return base_out + self.scaling * adapter_out

    @torch.no_grad()
    def merge(self) -> None:
        """
        把 (alpha/r) * B A 合并到 base.weight, 之后 forward 走快速路径。
        合并是 in-place 的; 需要再训练时调用 ``unmerge`` 还原。
        """
        if self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data.add_(delta)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """从 base.weight 中减去当前的 (alpha/r) * B A, 恢复可训练状态。"""
        if not self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data.sub_(delta)
        self.merged = False


# ---------------------------------------------------------------------------
# 模型级别的 LoRA 注入 / 冻结 / 合并 / 保存工具
# ---------------------------------------------------------------------------

def _name_matches(full_name: str, targets: Iterable[str]) -> bool:
    """
    判断模块的属性名是否落在 LoRA 目标集合中。

    full_name 形如 "layers.0.attn.w_q"。我们只比较最后一个属性段,
    因为 target_modules 给的是简短名 ("w_q" 等), 与具体层路径无关。
    """
    last = full_name.rsplit(".", 1)[-1]
    return last in set(targets)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    递归遍历模型, 把指定名字的 ``nn.Linear`` 替换为 ``LoRALinear``。

    步骤:
        1. 用 ``named_modules`` 找到所有候选父模块
        2. 对每个父模块, 检查它的 nn.Linear 子模块名是否命中 target_modules
        3. setattr 替换 — Python 的 setattr 会更新 _modules dict, PyTorch 会自动
           接管参数注册

    Args:
        model:          要注入 LoRA 的 nn.Module (例如 LLaMA)
        r, alpha, dropout: LoRA 超参, 见 ``LoRALinear``
        target_modules: 简短属性名列表, 默认替换 LLaMA 注意力的四个投影
                        ("w_q", "w_k", "w_v", "w_o")

    Returns:
        修改后的 model (仍为原对象, in-place); 返回值便于链式调用。

    Note:
        本函数只**注入**适配器, 不改变其它参数的 requires_grad。
        基座冻结需配合 ``mark_only_lora_as_trainable`` 使用。
    """
    targets = target_modules if target_modules is not None else DEFAULT_TARGET_MODULES

    # 先收集再修改: 避免在 named_modules 迭代过程中改 module dict 引发异常
    to_replace: List[tuple] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _name_matches(full_name, targets):
                to_replace.append((parent, child_name, child))

    if not to_replace:
        # 早期失败: 给出能立刻定位问题的报错, 而不是 "训练不收敛" 这种迟到症状
        raise ValueError(
            f"apply_lora: 未在模型中找到任何匹配 {targets} 的 nn.Linear。"
            f" 请检查 target_modules 是否与模型实际属性名一致。"
        )

    for parent, child_name, child in to_replace:
        new_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
        # 把替换后的层放回原位置; PyTorch 自动重新注册参数
        setattr(parent, child_name, new_layer)

    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    冻结模型中除 LoRA 参数外的所有参数。

    实现逻辑:
        - 先全部 freeze
        - 再把命名包含 "lora_A" / "lora_B" 的参数 requires_grad 置回 True

    为什么不只冻结 ``LoRALinear.base``?
        Token embedding / LayerNorm / lm_head 等不在 LoRALinear 内部的参数,
        在原始论文中也保持冻结。本函数实现"标准 LoRA"语义。
        如果想留出某些参数 (例如 lm_head) 可训, 调用本函数后再手动解冻即可。
    """
    freeze_module(model)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    把模型中所有 ``LoRALinear`` 的低秩增量合并回基座权重 (推理用)。

    合并后, 适配器 forward 走快速路径 (与 nn.Linear 同速); 但参数仍存在,
    重新调用 ``unmerge`` 可还原训练态。
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    抽出仅含 LoRA 参数的 state_dict, 便于落盘 (只保存 ~MB 级别的适配器,
    而非整个 GB 级基座)。

    上线流程:
        1. 训练完成 → ``torch.save(get_lora_state_dict(model), "adapter.pt")``
        2. 部署时: 加载基座 → ``apply_lora`` → ``model.load_state_dict(adapter, strict=False)``
        3. 可选: ``merge_lora_weights`` 合并以加速推理。
    """
    pattern = re.compile(r"\.lora_[AB]$|\.lora_[AB]\.")
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k or pattern.search(k) is not None
    }

"""
QLoRA — 4-bit NF4 量化基座 + LoRA 适配器 (Dettmers et al., 2023)
================================================================

历史背景:
    LoRA 把"可训练参数"从 100% 砍到 <1%, 但**基座权重本身**仍以 16 位驻留显存
    —— 65B 模型光权重就要 130GB, 单卡微调依然无门。
    QLoRA 的洞见: 基座反正是冻结的, 只读不写, 那就把它压到 4 bit 存放;
    LoRA 增量留在高精度。65B 微调从 780GB (全参) 压到 <48GB, 单卡可跑。

三个关键设计 (本文件实现前两个):
    1. **NF4 (4-bit NormalFloat)**: 神经网络权重近似 N(0, σ) 分布,
       与其用均匀 INT4 网格, 不如把 16 个量化格点放在正态分布的分位数上
       —— 每个格点"接住"等量的概率质量, 信息论意义上对正态数据最优。
    2. **block-wise absmax scaling**: 每 64 个权重一组, 各自缩放到 [-1, 1]
       再查 NF4 码本 (与 FP8 的 block scaling 同一思想, 见 llm_train/m13)。
    3. double quantization / paged optimizer: 把 scale 本身再量化、优化器状态
       分页到 CPU —— 工程优化, 教学版从略。

存储真相 (本实现诚实地按 4 bit 打包):
    两个 4-bit 索引拼进一个 uint8 → 每参数 0.5 字节 + 每 64 个参数一个
    float32 scale (摊到每参数 0.0625 字节), 合计 ~0.56 字节/参数,
    对比 FP32 的 4 字节/参数 ≈ **7.1x 压缩**。

前向:
    y = dequant(W_nf4) x + (alpha/r) · B A x
    教学版每次前向都反量化 (可读优先); 真实 kernel 在 GPU 上融合反量化与 GEMM。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.methods.lora import DEFAULT_TARGET_MODULES, _name_matches

# NF4 码本: 标准正态分布的 16 个等概率质量分位点, 归一化到 [-1, 1]
# (bitsandbytes 论文附录公布的常数, 0 被显式保留为一个格点)
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_quantize(
    w: torch.Tensor, block_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把权重量化成 NF4: 返回 (packed_uint8 [N/2], scales [n_blocks])。

    步骤: 按 block 切分 → absmax 缩放到 [-1,1] → 最近邻查码本 → 两索引拼一字节。
    """
    flat = w.detach().reshape(-1).float()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)

    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)  # absmax
    normed = blocks / scales                                          # ∈ [-1, 1]

    # 最近邻量化: 与 16 个码点逐一比距离 (教学写法; 真实实现用二分)
    idx = (normed.unsqueeze(-1) - NF4_CODEBOOK.to(w.device)).abs().argmin(dim=-1)
    idx = idx.view(-1).to(torch.uint8)                                # [N_padded]

    packed = (idx[0::2] << 4) | idx[1::2]                             # 两个 4bit 进一字节
    return packed, scales.view(-1)


def nf4_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: torch.Size,
    block_size: int = 64,
) -> torch.Tensor:
    """反量化: 拆包 → 查码本 → 乘回 block scale → reshape。"""
    hi = (packed >> 4).long()
    lo = (packed & 0x0F).long()
    idx = torch.stack([hi, lo], dim=1).view(-1)                       # 还原交错顺序

    codebook = NF4_CODEBOOK.to(packed.device)
    flat = codebook[idx].view(-1, block_size) * scales.view(-1, 1)
    numel = int(torch.tensor(shape).prod())
    return flat.view(-1)[:numel].view(shape)


class QLoRALinear(nn.Module):
    """
    NF4 量化的冻结基座 + 高精度 LoRA 旁路。

    基座以 buffer 形式存储 (packed uint8 + scales), **天然不可训练**;
    只有 lora_A / lora_B 是 nn.Parameter。与 methods/lora.py 的 LoRALinear
    接口对齐 (名字含 'lora_', mark_only_lora_as_trainable 同样适用)。
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.block_size = block_size
        self.weight_shape = base_layer.weight.shape

        packed, scales = nf4_quantize(base_layer.weight, block_size)
        self.register_buffer("packed_weight", packed)     # uint8, 0.5 B/参数
        self.register_buffer("scales", scales)            # fp32, 1/block 个
        # bias (若有) 很小, 保持高精度且冻结
        if base_layer.bias is not None:
            self.register_buffer("bias", base_layer.bias.detach().clone())
        else:
            self.bias = None

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))  # BA=0 无害启动

    def dequantized_weight(self) -> torch.Tensor:
        return nf4_dequantize(
            self.packed_weight, self.scales, self.weight_shape, self.block_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight()                     # 教学版: 每次前向反量化
        out = F.linear(x, w, self.bias)
        out = out + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out

    def memory_bytes(self) -> dict:
        """基座存储开销对账单: NF4 vs 原 FP32。"""
        nf4 = self.packed_weight.numel() + self.scales.numel() * 4
        fp32 = int(torch.tensor(self.weight_shape).prod()) * 4
        return {"nf4_bytes": nf4, "fp32_bytes": fp32}


def apply_qlora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    block_size: int = 64,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """把命中 target_modules 的 nn.Linear 替换为 QLoRALinear (复用 lora.py 的匹配规则)。"""
    targets = target_modules if target_modules is not None else DEFAULT_TARGET_MODULES

    to_replace: List[tuple] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _name_matches(full_name, targets):
                to_replace.append((parent, child_name, child))

    if not to_replace:
        raise ValueError(f"没有命中任何 nn.Linear, target_modules={targets}")

    for parent, child_name, child in to_replace:
        setattr(parent, child_name, QLoRALinear(child, r=r, alpha=alpha, block_size=block_size))
    return model

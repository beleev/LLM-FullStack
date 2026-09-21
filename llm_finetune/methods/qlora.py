"""
QLoRA — 4-bit NF4 冻结基座 + 高精度 LoRA (Dettmers et al., 2023)

是什么: 基座反正只读不写 → 压成 4 bit 存; 前向时反量化回高精度算, 梯度只流进 LoRA 支路。
解决什么: LoRA 省掉了梯度和优化器状态, 但 16-bit 基座本身还在显存里 (65B ≈ 130 GB)。
核心公式:  y = dequant(W_nf4) x + (α/r) B A x
           NF4 = 把 16 个格点放在 N(0,1) 的等概率分位数上 (权重近似正态 → 每格接住等量的权重);
           每 block_size 个权重共用一个 absmax scale。 存储 = 0.5 B/参数 + 4 B/block ≈ 0.56 B/参数 (vs fp32 的 4 B)。
量化谁: 所有 block 内的线性层 (注意力 4 个 + SwiGLU 3 个, 占参数的绝大部分)。
        embedding / lm_head 保持高精度: 二者共享权重; embedding 是查表不是 matmul, 每行只被少数 token 更新、
        分布远非正态; lm_head 的误差直接变成 logits 误差。bitsandbytes 默认同样跳过它们。
读代码时盯住: `NF4Linear.weight` —— 它是 property, 每次访问都现场反量化 (真实 kernel 把反量化融进 GEMM)。
未实现: double quantization (把 scale 再量化) 与 paged optimizer。
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_finetune.methods.lora import ALL_LINEARS, LoRALinear, replace_linears

# bitsandbytes 公布的 NF4 码本: 标准正态 16 个分位点, 归一化到 [-1, 1], 0 被显式保留
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_quantize(w: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """w → (packed uint8 [⌈N_pad/2⌉], scales [n_blocks])。N_pad = numel 向上对齐到 block_size。"""
    flat = w.detach().reshape(-1).float()
    flat = F.pad(flat, (0, (-flat.numel()) % block_size))             # 末块补 0
    blocks = flat.view(-1, block_size)                                # [n_blocks, bs]
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)  # absmax → 归一化到 [-1, 1]
    # 最近邻查码本: [n_blocks, bs, 1] − [16] → argmin (真实实现用二分)
    idx = ((blocks / scales).unsqueeze(-1) - NF4_CODEBOOK.to(w.device)).abs().argmin(dim=-1)
    idx = idx.view(-1).to(torch.uint8)                                # [N_pad]
    if idx.numel() % 2:                                               # block_size 为奇数时 N_pad 可能是奇数:
        idx = F.pad(idx, (0, 1))                                      # 补一个索引凑满最后一个字节
    return (idx[0::2] << 4) | idx[1::2], scales.view(-1)              # 高 4 位 | 低 4 位


def nf4_dequantize(packed: torch.Tensor, scales: torch.Tensor, shape: torch.Size,
                   block_size: int = 64) -> torch.Tensor:
    idx = torch.stack([packed >> 4, packed & 0x0F], dim=1).view(-1).long()   # 还原交错顺序
    idx = idx[: scales.numel() * block_size]                          # 丢掉凑字节的那个索引
    flat = NF4_CODEBOOK.to(packed.device)[idx].view(-1, block_size) * scales.view(-1, 1)
    return flat.view(-1)[: shape.numel()].view(shape)                 # 丢掉末块补的 0


class NF4Linear(nn.Module):
    """冻结的 4-bit 线性层: 权重只以 buffer (uint8 + scale) 存在, 天然没有梯度。"""

    def __init__(self, base: nn.Linear, block_size: int = 64) -> None:
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.block_size, self.shape = block_size, base.weight.shape
        packed, scales = nf4_quantize(base.weight, block_size)
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", None if base.bias is None else base.bias.detach().clone())

    @property
    def weight(self) -> torch.Tensor:
        return nf4_dequantize(self.packed_weight, self.scales, self.shape, self.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def apply_qlora(model: nn.Module, r: int = 8, alpha: float = 16, block_size: int = 64,
                target_modules: Optional[Sequence[str]] = None,
                layer_cls: type = LoRALinear) -> nn.Module:
    """命中的 nn.Linear → layer_cls(NF4Linear(原层))。默认命中全部 7 种线性层; layer_cls=DoRALinear 即 QDoRA。"""
    replace_linears(model, target_modules or ALL_LINEARS,
                    lambda m: layer_cls(NF4Linear(m, block_size), r=r, alpha=alpha))
    return model


def weight_bytes(model: nn.Module) -> int:
    """整个模型权重的真实字节数 (参数 + NF4 buffer); parameters() 已对共享权重去重。"""
    n = sum(p.numel() * p.element_size() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, NF4Linear):
            n += m.packed_weight.numel() + m.scales.numel() * m.scales.element_size()
    return n

"""
Mamba — 不用 attention 的语言模型 (Gu & Dao, 2023)

是什么: 每层 = Pre-RMSNorm + MambaLayer + 残差; 没有 QKV、没有因果 mask、没有位置编码、没有 FFN。
解决了什么: Transformer 训练 O(T²), 推理时 KV cache 随 T 线性增长;
           Mamba 训练 O(T), 推理每步 O(1) —— 全部历史压在定长状态 (h, conv 窗口) 里。
MambaLayer:  x ─ in_proj ─┬─ main: 因果 depthwise Conv1d → SiLU → SelectiveSSM ─┐
                          └─ gate: SiLU ───────────────────────────────────────⊙─ out_proj
    Conv 提供局部 token 混合 (SSM 前的"短程记忆"), gate 即 GLU 式门控, 顶替了 FFN。
关键数字: d_inner = 2·D, d_state N = 16, d_conv = 4; 每层解码状态 = D_in·N + D_in·(d_conv-1) 个数, 与 T 无关。
读代码时盯住: cache —— {"conv": 最近 d_conv-1 个输入, "h": SSM 状态}; 对照 Transformer 的 KV cache 会越长越大。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.sparse.ssm import SelectiveSSM
from llm_models.utils.generation import GenerationMixin, KVCache
from llm_models.utils.init import init_weights


class MambaLayer(nn.Module):
    """conv + selective SSM + gate。d_inner 默认 2·d_model。"""

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        d_inner = d_inner or 2 * d_model
        self.d_inner, self.d_conv = d_inner, d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)     # main + gate 一次投出
        # depthwise (groups=d_inner); padding=0, 因果性靠 forward 里手动左填充
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, bias=True)
        self.ssm = SelectiveSSM(d_model=d_inner, d_state=d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]"""
        x_main, x_gate = self.in_proj(x).chunk(2, dim=-1)              # 各 [B, T, d_inner]

        x_main = x_main.transpose(1, 2)                                # [B, d_inner, T]
        if cache and "conv" in cache:
            x_main = torch.cat([cache["conv"], x_main], dim=2)         # 左边接上一步留下的窗口
        else:
            x_main = F.pad(x_main, (self.d_conv - 1, 0))               # 左填 0 ⇒ 只看过去
        if cache is not None:
            cache["conv"] = x_main[:, :, x_main.size(2) - (self.d_conv - 1):]  # [B, d_inner, d_conv-1]
        # depthwise conv = 每通道对最近 d_conv 个输入加权求和。等价于 self.conv1d(x_main),
        # 但 CPU 上 grouped conv 是逐组循环 (实测占前向 90% 时间), 所以直接用 unfold 写出来
        win = x_main.unfold(2, self.d_conv, 1)                         # [B, d_inner, T, d_conv]
        x_main = (win * self.conv1d.weight[:, 0, None, :]).sum(-1) + self.conv1d.bias[:, None]
        x_main = F.silu(x_main.transpose(1, 2))                        # [B, T, d_inner]

        y = self.ssm(x_main, cache=cache) * F.silu(x_gate)             # [B, T, d_inner]
        return self.out_proj(y)                                        # [B, T, D]


class MambaBlock(nn.Module):
    """x + MambaLayer(RMSNorm(x))"""

    def __init__(self, d_model: int, d_inner: Optional[int] = None, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.layer = MambaLayer(d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        return x + self.layer(self.norm(x), cache=cache)


class Mamba(GenerationMixin, nn.Module):
    """idx -> Embedding -> N × MambaBlock -> RMSNorm -> lm_head (与 embedding 共享权重)"""

    max_len = 1 << 30   # 无位置编码, 无上下文上限 (GenerationMixin 用它裁剪前缀)

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        d_inner: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv) for _ in range(num_layers)]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight              # weight tying

        init_weights(self)                       # Linear/Embedding -> N(0, 0.02²) ⇒ 初始 CE ≈ ln V
        for blk in self.layers:                  # init_weights 会清零 dt_proj.bias, Δ 的专用初始化要补回来
            blk.layer.ssm.reset_dt()

    def forward(self, idx: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        """idx: [B, T] -> logits [B, T, V]。cache 里存的是递推状态而不是 K/V, 大小与已读长度无关。"""
        x = self.token_embedding(idx)            # [B, T, D]; 不乘 sqrt(D): 没有位置编码要与之平衡
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache.layers[i] if cache else None)
        if cache is not None:
            cache.pos += idx.size(1)
        return self.lm_head(self.ln_f(x))        # [B, T, V]

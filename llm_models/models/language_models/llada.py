"""
LLaDA — 掩码扩散语言模型 (Nie et al. 2025, "Large Language Diffusion Models")

是什么: 一个 **没有因果 mask 的 LLaMA**, 训练时随机遮住一部分 token 让它还原, 生成时从全 [MASK] 逐步去噪。
解决了什么: BERT 固定遮 15% → 只是表示学习器, 不能从全 [MASK] 生成;
           LLaDA 让遮蔽比例 t ~ U(0,1) 并给 loss 乘 1/t → 这个 loss 是 −log p(x) 的上界 (ELBO), 成了真正的生成模型。
           相对自回归: 每步并行预测所有位置、可任意位置填空、没有 "只会从左往右" 的反转诅咒; 代价是没有 KV cache。
关键公式:
    前向 (加噪):  每个 token 独立以概率 t 变成 [MASK]
    loss = E_t E_mask [ (1/t) · Σ_{i 被遮} −log p(x_i | x_noisy) ] / L        ≥ −log p(x) / L
    采样: 第 s 步 (共 N 步) 后仍遮住的个数 = round(n · (1 − s/N)); 留下置信度高的, 把置信度低的重新遮住
读代码时盯住: `t` (训练时的遮蔽比例) 和 `n_masked` (采样时每步还剩多少 [MASK])。
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.training.loss import LossComputer
from llm_models.utils.init import init_weights


def forward_process(
    x: torch.Tensor,
    mask_id: int,
    eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    扩散前向过程: 每条序列抽一个 t, 每个 token 独立以概率 t 换成 [MASK]。

    Returns: noisy [B, T], masked [B, T] bool, t [B]
    """
    B, T = x.shape
    # 分层采样: 一个 batch 内的 t 均匀铺满 (eps, 1), 比 B 个独立 U(0,1) 方差小得多 (1/t 权重很吃方差)
    u = torch.rand(1, generator=generator, device=x.device)
    t = (u + torch.arange(B, device=x.device) / B) % 1 * (1 - eps) + eps  # [B]
    masked = torch.rand(B, T, generator=generator, device=x.device) < t[:, None]  # [B, T]
    return x.masked_fill(masked, mask_id), masked, t


class LLaDALoss(LossComputer):
    """
    LLaDA 的 ELBO: 只在被遮位置算 CE, 乘 1/t, 再除以 **总 token 数** (不是被遮 token 数)。

    为什么是 1/t: t 小 → 被遮的 token 少, 不加权的话小 t 的样本几乎不贡献 loss;
    1/t 恰好使 E[被遮个数 / t] = L, 于是均匀猜测时 loss 的期望正好是 ln V, 与自回归 CE 同一量纲。

    不能直接塞进 Trainer: 加噪必须发生在 forward **之前**且每步重新随机, 所以训练脚本自己写循环。
    """

    def compute(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        masked: torch.Tensor = None,
        t: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # cross_entropy 要求类别维在 dim=1: [B, T, V] -> [B, V, T]
        ce = F.cross_entropy(model_output.transpose(1, 2), labels, reduction="none")  # [B, T]
        loss = (ce * masked / t[:, None]).sum() / labels.numel()
        return {"total_loss": loss}


class LLaDA(nn.Module):
    """
    LLaDA (教学版): LLaMA 的零件 (GQA + SwiGLU + RMSNorm + RoPE), 唯一的结构差异是 **不传因果 mask**。

    词表约定: 最后一个 id (vocab_size − 1) 是 [MASK]。

    Args 同 LLaMA; max_len 只用于 RoPE 预计算。
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 32,
        max_len: int = 4096,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size - 1
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(d_model // n_heads, max_len)
        if d_ff is None:
            d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64  # 同 LLaMA
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=GroupedQueryAttention(d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying
        init_weights(self)

    def forward(self, idx: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """idx [B, T] (可含 mask_id), attention_mask [B, T] 1=有效 0=pad → logits [B, T, V]"""
        if idx.size(1) > self.max_len:
            raise ValueError(f"序列长度 {idx.size(1)} 超过 max_len={self.max_len}")
        x = self.token_embedding(idx) * math.sqrt(self.d_model)  # [B, T, D]
        # 与 LLaMA 唯一的区别: 只有 padding mask, 没有下三角 → 每个位置能看到左右两边
        mask = None if attention_mask is None else attention_mask.bool().unsqueeze(1)  # [B, 1, T]
        for layer in self.layers:
            x = layer(x, mask=mask, rope=self.rope)
        return self.lm_head(self.ln_f(x))  # [B, T, V]

    @torch.inference_mode()
    def sample(
        self,
        x: torch.Tensor,
        steps: int,
        remasking: str = "low_confidence",
        temperature: float = 0.0,
        return_history: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        反向去噪: 把 x 中所有 mask_id 的位置填上 (位置任意 → 续写和填空是同一个函数)。

        每步: 预测所有 [MASK] → 全部暂时填上 → 把置信度最低的重新遮住, 使剩余 [MASK] 数服从线性日程。
        已经定下来的 token 置信度记为 +inf, 永不重遮 (论文 Algorithm 5)。

        Args:
            remasking:   "low_confidence" (置信度 = 所选 token 的概率) | "random"
            temperature: 0 → argmax; >0 → 按 softmax(logits / temperature) 采样
        Returns:
            x [B, T] (不含 mask_id); return_history=True 时附带每步结束后的 x 快照列表 (每项 [B, T])
        """
        assert remasking in ("low_confidence", "random")
        self.eval()
        x = x.clone()
        n_gen = (x == self.mask_id).sum(1)  # [B] 每行要生成多少个
        history = []
        for s in range(1, steps + 1):
            is_masked = x == self.mask_id  # [B, T]
            logits = self(x)  # [B, T, V]  每步整段重算: 双向注意力下没有 KV cache 可用
            logits[..., self.mask_id] = float("-inf")  # 永远不生成 [MASK] 本身
            probs = F.softmax(logits, dim=-1)
            if temperature > 0:
                p = F.softmax(logits / temperature, dim=-1)
                x0 = torch.multinomial(p.flatten(0, 1), 1).view_as(x)  # [B, T]
            else:
                x0 = probs.argmax(-1)  # [B, T]
            if remasking == "low_confidence":
                conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, T]
            else:
                conf = torch.rand(x.shape, device=x.device)
            conf = conf.masked_fill(~is_masked, float("inf"))  # 已定的 token 排在最后, 不会被重遮

            x = torch.where(is_masked, x0, x)  # 先全部填上
            n_masked = torch.round(n_gen * (1 - s / steps)).long()  # [B] 线性日程: 本步结束后应剩的 [MASK] 数
            rank = conf.argsort(1).argsort(1)  # [B, T] 置信度升序名次, 0 = 最没把握
            x = x.masked_fill(rank < n_masked[:, None], self.mask_id)  # 重新遮住最没把握的 n_masked 个
            history.append(x.clone())
        return (x, history) if return_history else x

    def generate(self, prompt: torch.Tensor, gen_len: int, steps: int, **kwargs):
        """prompt [B, P] → [B, P + gen_len]: 在 prompt 后接 gen_len 个 [MASK] 再去噪。"""
        blanks = prompt.new_full((prompt.size(0), gen_len), self.mask_id)
        return self.sample(torch.cat([prompt, blanks], dim=1), steps, **kwargs)

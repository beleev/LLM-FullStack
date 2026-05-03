"""
注意力机制模块 — Attention 演进史的浓缩版

按时间线 / 论文顺序排列:
- ScaledDotProductAttention: "Attention Is All You Need" (Vaswani et al., 2017) 的数学核心
- SingleHeadSelfAttention:   单头自注意力，教学用，便于观察权重分布
- MultiHeadAttention:        教学版多头 (per-head 显式循环)，便于逐头可视化
- GroupedQueryAttention:     GQA (Ainslie et al., 2023) — LLaMA-2 70B / Qwen2 等用来缓解
                             KV cache 显存瓶颈：Q 头多、KV 头少，每组 Q 共享同一对 KV
- MultiHeadLatentAttention:  MLA (DeepSeek-V2/V3, 2024) — 把 KV 投到低秩 latent c_kv，
                             KV cache 减少 ~93%；同时引入解耦 RoPE 解决 latent 不能旋转的问题
- MultiHeadLatentSparseAttention: DSA (DeepSeek V3.2, 2025) — MLA + Lightning Indexer，
                             把 attention 从 O(L^2) 降到 O(L*k)，支持超长上下文

设计权衡总览:
    MHA   (KV cache 大、表达力满)
      → MQA (KV head=1，cache 最小但效果掉)
      → GQA (折中：cache 小且效果接近 MHA，工业首选)
      → MLA (低秩压缩 KV，cache 进一步降，引入更复杂的解耦 RoPE)
      → DSA (在 MLA 上再叠加稀疏选择，主攻长上下文成本)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_attn_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """把任意 2D/3D/4D 掩码扩到 [B, 1, T, S] 或 [1, 1, T, S] 以便广播。"""
    if mask is None:
        return None
    if mask.dim() == 2:  # [T, S] -> [1, 1, T, S]
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:  # [B, T, S] -> [B, 1, T, S]
        return mask.unsqueeze(1)
    return mask  # [B, 1, T, S] or [B, H, T, S]


def _call_rope(
    rope: nn.Module,
    x: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """兼容新旧 RoPE 接口：新 RoPE 支持 position_ids，旧的则不传。"""
    if position_ids is None:
        return rope(x)
    try:
        return rope(x, position_ids=position_ids)
    except TypeError:
        return rope(x)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 — Transformer 的最小数学核心

    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    为什么除以 sqrt(d_k):
        当 d_k 较大时，Q·K 内积的方差会随 d_k 线性增长，softmax 会被推到饱和区
        (概率几乎全压在一个 token 上)，反传梯度趋近于 0。除以 sqrt(d_k) 把内积方差
        重新拉回 O(1)，让 softmax 处在梯度健康的区间。
    """

    def forward(self, Q, K, V, mask=None):
        # Q: [..., T, d_k]   K: [..., S, d_k]   V: [..., S, d_v]
        d_k = Q.size(-1)

        # QK^T -> [..., T, S]，再做缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask 为 0/False 表示屏蔽：填 -inf 后 softmax(-inf)=0，相当于该位置看不到
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # 沿 key 维归一化
        output = torch.matmul(attn_weights, V)    # [..., T, d_v]
        return output, attn_weights


class SingleHeadSelfAttention(nn.Module):
    """
    单头自注意力 (教学用)

    生产代码通常直接写多头；单头版本保留用于教学：
        - 便于把注意力权重可视化成单张热力图
        - 便于断点观察 Q/K/V 的形状与数值范围
    支持可选的 RoPE 注入；真实多头见下方 MultiHeadAttention / GQA。
    """

    def __init__(self, d_input: int, d_out: int):
        super().__init__()
        self.w_q = nn.Linear(d_input, d_out)
        self.w_k = nn.Linear(d_input, d_out)
        self.w_v = nn.Linear(d_input, d_out)
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        # k/v 为 None 时退化为自注意力 (Q=K=V 同源)；否则做交叉注意力
        k = k if k is not None else q
        v = v if v is not None else q

        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # RoPE 仅作用于 Q/K (位置参与匹配)，V 不旋转 (V 携带的是内容值)
        if rope is not None:
            q_proj = _call_rope(rope, q_proj, position_ids)
            k_proj = _call_rope(rope, k_proj, position_ids)

        output, _ = self.attention(q_proj, k_proj, v_proj, mask)
        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力 (教学版) — Vaswani et al. 2017 原始 MHA

    多头的动机:
        单头 attention 只能学到一种 (Q,K) 相关性；多头让不同 head 在不同子空间
        关注不同模式 (语法 / 语义 / 位置邻近 …)，再拼接融合。

    本实现用 nn.ModuleList per-head 循环，运算等价于标准 MHA 但更慢，
    优势是可以逐头单独观察、调试。生产代码请用 GroupedQueryAttention
    (num_kv_heads = num_heads 即等价于 MHA)。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList(
            [SingleHeadSelfAttention(d_input=d_model, d_out=self.d_head) for _ in range(num_heads)]
        )
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        # 每个 head 独立投影并算 attention，输出 [B, T, d_head]
        head_outputs = [
            head(q, k, v, mask, rope=rope, position_ids=position_ids) for head in self.heads
        ]
        # 拼接所有 head -> [B, T, num_heads * d_head] = [B, T, d_model]，再做输出投影 W_O
        concat_output = torch.cat(head_outputs, dim=-1)
        return self.w_o(concat_output)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) — Ainslie et al., 2023

    动机:
        推理阶段 KV cache 显存随 num_heads 线性增长，是大模型显存的最大头之一
        (LLaMA-2 70B 单 token 的 KV cache 高达数 MB)。
        GQA 让多组 Q head 共享同一对 K/V head，cache 体积 ÷ num_groups，
        几乎不掉点 (论文显示与 MHA 差距 < 1%)。

    极限情况:
        - num_kv_heads == num_heads     → 普通 MHA (cache 最大)
        - num_kv_heads == 1             → MQA (cache 最小，但效果略差)
        - 1 < num_kv_heads < num_heads  → GQA (LLaMA-2 70B / Qwen2 / Mistral 标配)

    Args:
        d_model: 模型维度
        num_heads: Q 的 head 数
        num_kv_heads: K/V 的 head 数；默认等于 num_heads (即 MHA)
        bias: 是否使用偏置 (现代 LLM 普遍 bias=False，节省参数且训练更稳)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_q = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.w_o = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        k = q if k is None else k
        v = q if v is None else v

        B, T, _ = q.shape
        S = k.size(1)

        # 投影并切头：[B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh]
        # transpose(1,2) 是为了让 head 维放到 batch 维之后，便于批量 matmul
        Q = self.w_q(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(k).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(v).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 在切头之后、scores 之前应用：每个 head 独立旋转
        if rope is not None:
            Q = _call_rope(rope, Q, position_ids)
            K = _call_rope(rope, K, position_ids)

        # 把 KV head 复制 num_groups 份，使其形状对齐 Q head，能直接做 matmul
        # 教学实现用 repeat_interleave；高性能 kernel (FlashAttention) 会跳过这一步
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)

        # [B, H, T, Dh] · [B, H, Dh, S] -> [B, H, T, S]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        norm_mask = _normalize_attn_mask(mask)
        if norm_mask is not None:
            scores = scores.masked_fill(norm_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, H, T, Dh]

        # 还原回 [B, T, H*Dh] 再做输出投影。contiguous() 是因为 transpose 不连续
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.w_o(out)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) — DeepSeek-V2/V3 论文核心

    背景:
        GQA 通过减少 KV head 数压缩 cache，但仍需缓存 num_kv_heads * head_dim 维度。
        DeepSeek-V2 提出: 把 KV 投到一个 **更小的低秩 latent c_kv** (例如 512 维)，
        实际 KV 现场升维即可。论文报告 KV cache 减少约 93% 而效果不掉。

    三大关键设计:
        1. KV 低秩压缩 (LoRA-like):
               c_kv = W_DKV(x)，shape [B, T, kv_lora_rank]
           K/V 都由同一个 c_kv 升维而来。推理只需缓存 c_kv (+ 共享 K-rope)。

        2. 解耦 RoPE (decoupled RoPE):
           为什么 latent c_kv 不能直接 RoPE？因为 RoPE 是 head_dim 上的旋转，
           作用在低秩 c_kv 上会破坏其低秩结构 (吸收 trick 失效)。
           方案: 把 Q/K 的每头维度拆成两段:
               - nope 段 (no positional): 走 latent，不旋转
               - rope 段 (rotary):        独立小投影，加旋转，承载位置信息
           推理时 nope 段的 K 可被 W_UK 吸收进 Q 侧，只对 c_kv 算 matmul。

        3. 共享 K-rope:
           rope 段在所有 head 之间共享同一个 [B, T, rope_dim] 向量，再 broadcast，
           进一步压缩 cache (rope 段不按 head 切分)。

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        kv_lora_rank: KV 潜变量维度 c_kv (DeepSeek-V3 用 512)
        qk_nope_head_dim: Q/K 的 nope (不加 rope) 段每头维度
        qk_rope_head_dim: Q/K 的 rope (加旋转) 段每头维度
        v_head_dim: V 每头维度，默认等于 qk_nope_head_dim
        q_lora_rank: Q 的低秩维度 (可选，DeepSeek-V3 也压缩了 Q 来省训练显存)
        bias: 是否使用偏置

    备注:
        本实现走的是 "先升维再算 attention" 的训练等价路径，便于阅读；
        生产推理会做 "权重吸收" (absorb W_UK into W_Q)，仅缓存 c_kv 即可，
        KV cache 大小 = (kv_lora_rank + qk_rope_head_dim) per token。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_lora_rank: int = 128,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        bias: bool = False,
        # 兼容旧调用：latent_dim 将被映射为 kv_lora_rank
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        if latent_dim is not None:
            kv_lora_rank = latent_dim

        v_head_dim = v_head_dim if v_head_dim is not None else qk_nope_head_dim

        self.d_model = d_model
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # --- Q 投影 (可选低秩) ---
        if q_lora_rank is not None:
            self.q_down = nn.Linear(d_model, q_lora_rank, bias=bias)
            self.q_up = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.q_down = None
            self.q_up = nn.Linear(d_model, num_heads * self.qk_head_dim, bias=bias)

        # --- KV 低秩 + 解耦 K rope (一次投影到 [kv_lora + rope_dim]) ---
        self.kv_down = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.k_up = nn.Linear(kv_lora_rank, num_heads * qk_nope_head_dim, bias=bias)
        self.v_up = nn.Linear(kv_lora_rank, num_heads * v_head_dim, bias=bias)

        # --- 输出投影 ---
        self.w_o = nn.Linear(num_heads * v_head_dim, d_model, bias=bias)

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        # MLA 仅用于 self-attention，k/v 若传入需与 q 同；保留参数仅为兼容统一签名
        x = q
        B, T, _ = x.shape
        H = self.num_heads

        # --- 1) Q 投影 (可选低秩 down→up)，再切头并切分 nope/rope 两段 ---
        q_proj = self.q_up(self.q_down(x)) if self.q_down is not None else self.q_up(x)
        q_proj = q_proj.view(B, T, H, self.qk_head_dim).transpose(1, 2)  # [B, H, T, qk_head]
        q_nope, q_rope = torch.split(
            q_proj, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # --- 2) KV 压缩：一次投到 [c_kv | k_rope]，c_kv 是低秩 latent，k_rope 共享 ---
        kv_mix = self.kv_down(x)  # [B, T, kv_lora + rope_dim]
        c_kv, k_rope = torch.split(
            kv_mix, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # 解耦 K rope 在所有 head 之间共享: [B, T, rope] -> [B, 1, T, rope]
        k_rope = k_rope.unsqueeze(1)

        # 从同一个 c_kv 升维出 K-nope 和 V (两组 up-projection)
        k_nope = self.k_up(c_kv).view(B, T, H, self.qk_nope_head_dim).transpose(1, 2)
        v_heads = self.v_up(c_kv).view(B, T, H, self.v_head_dim).transpose(1, 2)

        # --- 3) 仅对 rope 段做 RoPE 旋转 (nope 段保持不变以维持低秩结构) ---
        if rope is not None:
            q_rope = _call_rope(rope, q_rope, position_ids)
            k_rope = _call_rope(rope, k_rope, position_ids)

        # 把共享 K-rope 广播到所有 head: [B, 1, T, rope] -> [B, H, T, rope]
        k_rope = k_rope.expand(-1, H, -1, -1)

        # --- 4) 拼接 [nope | rope] 后做标准缩放点积注意力 ---
        q_combined = torch.cat([q_nope, q_rope], dim=-1)  # [B, H, T, qk_head]
        k_combined = torch.cat([k_nope, k_rope], dim=-1)

        scores = torch.matmul(q_combined, k_combined.transpose(-2, -1)) * self.scale

        norm_mask = _normalize_attn_mask(mask)
        if norm_mask is not None:
            scores = scores.masked_fill(norm_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_heads)  # [B, H, T, v_head]

        out = out.transpose(1, 2).contiguous().view(B, T, H * self.v_head_dim)
        return self.w_o(out)


class LightningIndexer(nn.Module):
    """
    Lightning Indexer — DSA (DeepSeek V3.2) 的廉价选择器

    背景:
        长上下文场景下，标准 attention 是 O(L^2)，128k 上下文几乎不可行。
        DSA 的思路是 "先粗选后细算"：
            1. 用一个轻量 Indexer 快速估计 query-key 相关性
            2. 仅保留每行 top-k 个候选位置
            3. 再让昂贵的 MLA 只在这 k 个位置上算 softmax+加权

    设计要点:
        - Indexer head 数小、维度小 (<< 主 attention)，开销可忽略
        - 用 ReLU 而非 softmax: 避免归一化导致的稀疏性丢失，单调性即可用于排序
        - 多 head 分数相加再排序，提供集成稳健性

    公式 (简化):
        score(q_t, k_s) = Σ_h ReLU( (q_h · k_h) / sqrt(d_head) )
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除，"
                    "或显式指定 head_dim"
                )
            head_dim = d_model // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.w_q = nn.Linear(d_model, num_heads * head_dim, bias=bias)
        self.w_k = nn.Linear(d_model, num_heads * head_dim, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            index_scores: [B, T, S]
        """
        k = k if k is not None else q
        q_heads = self._split_heads(self.w_q(q))
        k_heads = self._split_heads(self.w_k(k))

        # einsum 等价于 matmul，但语义更清晰：算所有 (t, s) 对的内积
        scores = torch.einsum("bhtd,bhsd->bhts", q_heads, k_heads) * self.scale
        scores = F.relu(scores)        # 负相关直接置 0，保留正相关
        scores = scores.sum(dim=1)     # 聚合所有 indexer head -> [B, T, S]

        norm_mask = _normalize_attn_mask(mask)
        if norm_mask is not None:
            # 将 [B,1,T,S] 压回 [B,T,S]
            mask_flat = norm_mask.squeeze(1).bool()
            scores = scores.masked_fill(~mask_flat, float("-inf"))

        return scores


class MultiHeadLatentSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) + MLA — DeepSeek V3.2 (2025)

    在 MLA 之上叠加 Lightning Indexer 选 top-k 的稀疏化方案，
    把 attention 复杂度从 O(L^2) 降到 O(L*k)，主攻超长上下文成本。

    流程:
        1) Lightning Indexer 算粗略相关性，**在因果/padding mask 之后** 选 top-k
        2) 把 top-k 编成稀疏 mask 喂给 MLA，MLA 仅在这些位置算 softmax+V 加权

    教学实现: 仍构造 [B, T, S] 的稠密 sparse_mask 再传给 MLA；
              生产版本会跳过被 mask 掉的 KV 读取，配合自定义 CUDA kernel。

    重要正确性提醒:
        必须先用 base_mask 把不可见位置的 indexer 分数压成 -inf，再 topk；
        否则 top-k 可能选中未来 token，造成训练时数据泄漏。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_lora_rank: int = 128,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        indexer_heads: int = 4,
        indexer_head_dim: Optional[int] = None,
        sparse_top_k: int = 128,
        bias: bool = False,
        # 兼容旧参数
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        if sparse_top_k <= 0:
            raise ValueError(f"sparse_top_k 必须为正数，当前 {sparse_top_k}")

        self.sparse_top_k = sparse_top_k

        self.mla = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            bias=bias,
            latent_dim=latent_dim,
        )

        self.indexer = LightningIndexer(
            d_model=d_model,
            num_heads=indexer_heads,
            head_dim=indexer_head_dim,
            bias=bias,
        )

    def _sparse_mask_from_topk(
        self,
        index_scores: torch.Tensor,
        base_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        在 base_mask 限定的可见区域内取 top-k；选不到 k 个时自动退化为全可见。

        Args:
            index_scores: [B, T, S]
            base_mask: [B, T, S] (True=可见)

        Returns:
            sparse_mask: [B, T, S] bool
        """
        B, T, S = index_scores.shape
        k = min(self.sparse_top_k, S)

        if k >= S:
            # 稀疏实际等价于稠密
            if base_mask is None:
                return torch.ones_like(index_scores, dtype=torch.bool)
            return base_mask.bool()

        # 重要修复: 先屏蔽被 base_mask 禁止的位置，再取 top-k
        scores = index_scores
        if base_mask is not None:
            scores = scores.masked_fill(~base_mask.bool(), float("-inf"))

        topk_indices = torch.topk(scores, k=k, dim=-1).indices
        sparse = torch.zeros(B, T, S, dtype=torch.bool, device=index_scores.device)
        sparse.scatter_(-1, topk_indices, True)

        if base_mask is not None:
            sparse = sparse & base_mask.bool()
        return sparse

    def forward(self, q, k=None, v=None, mask=None, rope=None, position_ids=None):
        # 1) Lightning Indexer 给所有 (t, s) 打分
        index_scores = self.indexer(q, k=q, mask=mask)

        # 把外部 mask 规范成 [B, T, S]，传给 _sparse_mask_from_topk 做先 mask 再 topk
        base_mask_3d = None
        if mask is not None:
            m = _normalize_attn_mask(mask).squeeze(1)
            base_mask_3d = m if m.dim() == 3 else m.unsqueeze(0).expand(q.size(0), -1, -1)

        sparse_mask = self._sparse_mask_from_topk(index_scores, base_mask_3d)

        # 2) MLA 在稀疏 mask 限定的 top-k 候选上做精细 attention
        return self.mla(q, mask=sparse_mask, rope=rope, position_ids=position_ids)

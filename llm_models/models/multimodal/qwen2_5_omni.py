"""
Qwen2.5-Omni 架构演示模块

论文出处:
    "Qwen2.5-Omni Technical Report" (Xu et al., 2025, 阿里通义千问团队)
    第一个开源支持 "文本+图像+音频+视频" 全模态输入与 "文本+流式语音" 输出的端到端模型

核心创新 (相比 GPT-4o / Gemini 等闭源对手):
    1) Thinker-Talker 双脑结构:
       - Thinker: 大型 LLM 主干，吸收所有模态做"思考"，输出文本与隐状态
       - Talker: 小型自回归解码器，只读 Thinker 的 hidden 来流式生成语音 codec token
       为什么拆成两个? 因为语音生成是流式的 (要边想边说), 而文本理解是批量的;
       解耦后 Talker 可以用极小模型实现低延迟流式输出，Thinker 专注质量
    2) TMRoPE (Time-aligned M-RoPE):
       在 M-RoPE 基础上把视频帧与音频片段按真实时间戳对齐到同一个时间轴,
       让模型理解"画面里嘴巴动 == 同一时刻的语音"
    3) 音频用 mel-spectrogram + Whisper 风格 encoder:
       而非直接送波形 — 频域特征更紧凑、训练更稳，且能直接利用 ASR 预训练权重

本文件采用早融合 + cross-attention 混合策略:
    所有模态 token 拼到 Thinker 前缀 (早融合); Talker 通过 cross-attention
    访问 Thinker 隐状态 (晚融合), 兼顾理解能力与流式生成

通用编码器/投影器/重采样器抽到 multimodal.py; Thinker 直接复用 Qwen2VLDecoder,
本文件只保留 Talker 和顶层胶水。
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNCrossBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import (
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)
from llm_models.layers.multimodal import (
    ModalityProjector,
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResampler,
)
from llm_models.models.multimodal.qwen2_vl import Qwen2VLDecoder
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


# Thinker 与 Qwen2-VL 解码器同构 (decoder-only LLM + M-RoPE)
# 仅用别名导出，避免重复实现。这也说明 Omni 的"理解侧"复用了 VL 的全部成熟设计
OmniThinkerDecoder = Qwen2VLDecoder


class OmniTalkerDecoder(nn.Module):
    """
    Talker — 流式语音解码器 (Qwen2.5-Omni 的"嘴巴")

    结构 (每层): self-attn -> cross-attn -> FFN，与经典 Transformer decoder 相同
        x -> RMSNorm -> Masked Self-Attn (GQA) -> Add
          -> RMSNorm -> Cross-Attn (context=Thinker hidden) -> Add
          -> RMSNorm -> SwiGLU FFN -> Add

    设计要点:
        - 词表是离散语音 codec token (如 Encodec / SoundStream 输出的 1024 个码本索引)
          不是文字; 由专门的 codec decoder 还原波形
        - 通过 cross-attention 而非 prefix 注入 Thinker 隐状态:
          (a) Talker 每生成一个语音 token 就要 cross 一次, 不污染自身上下文
          (b) Thinker 隐状态可流式追加, Talker 边想边说
          (c) Talker 可以做得很小 (低延迟), 与大 Thinker 解耦

    Args:
        vocab_size: 语音 codec 词表大小 (典型 1024)
        d_model: 隐维度
        n_heads: Q head 数
        num_kv_heads: K/V head 数 (GQA, None 则等于 n_heads)
        num_layers: 层数 (Talker 通常 12 层左右, 远少于 Thinker)
        max_len: 最大音频 token 序列长度
        dropout: Dropout 概率
        use_rope: 是否使用 RoPE (否则使用 Sinusoidal)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        max_len: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        d_ff: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_rope = use_rope

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self._embed_scale = math.sqrt(d_model)

        if use_rope:
            d_head = d_model // n_heads
            self.pos_encoder = RotaryPositionalEncoding(d_head, max_len)
        else:
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)

        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)

        self.layers = nn.ModuleList(
            [
                PreLNCrossBlock(
                    d_model=d_model,
                    self_attn=GroupedQueryAttention(
                        d_model=d_model,
                        num_heads=n_heads,
                        num_kv_heads=num_kv_heads,
                    ),
                    cross_attn=GroupedQueryAttention(
                        d_model=d_model,
                        num_heads=n_heads,
                        num_kv_heads=num_kv_heads,
                    ),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = RMSNorm(d_model)
        # Talker 通常不与 token_embedding 共享权重 — 因为语音 codec 词表与文本词表
        # 完全不同; 共享反而会拖累训练
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        causal = build_causal_mask(max_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids) * self._embed_scale

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    @staticmethod
    def _normalize_context_mask(
        context_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """将 [B, S] padding mask 扩成 [B, 1, S] 便于广播到 [B, T, S]。"""
        if context_mask is None:
            return None
        if context_mask.dim() == 2:
            return context_mask.unsqueeze(1)
        return context_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Talker 必须依赖 Thinker 的语义条件才能生成有意义的语音
        # 否则就退化为无条件语音 LM, 失去 "想清楚再说" 的设计意图
        if context is None:
            raise ValueError("Talker 需要 context (Thinker 隐状态) 作为条件输入")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("input_ids 与 inputs_embeds 必须且只能传入一个")

        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        _, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")

        rope_handler: Optional[nn.Module] = None
        if self.use_rope:
            rope_handler = self.pos_encoder
        else:
            x = self.pos_encoder(x)

        # 双 mask:
        #   self_mask  — 自身因果 + padding (生成端约束)
        #   cross_mask — 标记 context 中哪些位置可以 attend (来自 Thinker 的 padding)
        causal = self._causal_mask(seq_len)
        self_mask = combine_causal_and_padding_mask(causal, attention_mask)
        cross_mask = self._normalize_context_mask(context_mask)

        for layer in self.layers:
            x = layer(
                x,
                context=context,
                self_mask=self_mask,
                context_mask=cross_mask,
                rope=rope_handler,
            )

        x = self.ln_f(x)
        return self.lm_head(x)


class Qwen2_5_OmniModel(nn.Module):
    """
    Qwen2.5-Omni 顶层模型 — 全模态输入 + 文本/语音双输出

    融合策略:
        - 输入侧: 早融合 (所有模态 token 拼到 Thinker 前缀, 共享因果注意力)
        - 输出侧: 双头解耦
            * text_logits  来自 Thinker — 给文本输出
            * audio_logits 来自 Talker  — 通过 cross-attention 读 Thinker 隐状态
              生成流式语音 codec token

    数据流:
        vision   -> Encoder -> (Resampler) -> Projector ─┐
        video    -> Encoder -> (Resampler) -> Projector ─┤
        audio    -> Encoder -> (Resampler) -> Projector ─┼── Concat ──┐
        text_ids -> TokenEmbed                            ┘            │
                                                                       ▼
                                            Thinker (decoder-only LLM, return_hidden=True)
                                                ├── text_logits
                                                └── hidden_states ──► Talker (cross-attn) ──► audio_logits
    """

    # 4 种模态用于 segment embedding 区分 — 视频与图像分开是因为它们时间属性不同
    MODALITY_VISION = 0
    MODALITY_VIDEO = 1
    MODALITY_AUDIO = 2
    MODALITY_TEXT = 3

    def __init__(
        self,
        vocab_size: int,
        audio_vocab_size: int = 1024,
        text_d_model: int = 1024,
        text_n_heads: int = 16,
        text_num_kv_heads: Optional[int] = None,
        text_num_layers: int = 24,
        max_len: int = 2048,
        # Vision
        vision_image_size: int = 224,
        vision_patch_size: int = 14,
        vision_d_model: int = 1024,
        vision_n_heads: int = 16,
        vision_num_layers: int = 24,
        vision_num_latents: int = 64,
        vision_num_latent_layers: int = 2,
        # Audio (声谱图)
        audio_spec_size: Tuple[int, int] = (256, 128),
        audio_patch_size: Union[Tuple[int, int], int] = (16, 16),
        audio_in_channels: int = 1,
        audio_d_model: int = 512,
        audio_n_heads: int = 8,
        audio_num_layers: int = 12,
        audio_num_latents: int = 64,
        audio_num_latent_layers: int = 2,
        # Video
        video_size: Tuple[int, int, int] = (8, 224, 224),
        video_tubelet_size: int = 2,
        video_patch_size: int = 14,
        video_in_channels: int = 3,
        video_d_model: int = 1024,
        video_n_heads: int = 16,
        video_num_layers: int = 12,
        video_num_latents: int = 64,
        video_num_latent_layers: int = 2,
        # Projector
        projector_hidden_dim: Optional[int] = None,
        # Talker
        talker_d_model: Optional[int] = None,
        talker_n_heads: int = 16,
        talker_num_kv_heads: Optional[int] = None,
        talker_num_layers: int = 12,
        talker_max_len: int = 1024,
        # Misc
        dropout: float = 0.1,
        use_rope: bool = True,
        use_mrope: bool = False,
        use_modality_embedding: bool = True,
    ):
        super().__init__()

        # --- Vision: 静态图像 (RGB), 与 Qwen2-VL 视觉端同构 ---
        self.vision_encoder = PatchTransformerEncoder(
            patch_embed=PatchEmbed2D(
                input_size=vision_image_size,
                patch_size=vision_patch_size,
                in_channels=3,
                embed_dim=vision_d_model,
            ),
            d_model=vision_d_model,
            n_heads=vision_n_heads,
            num_layers=vision_num_layers,
            dropout=dropout,
        )
        self.vision_resampler = self._maybe_resampler(
            vision_num_latents, vision_d_model, vision_n_heads,
            vision_num_latent_layers, dropout,
        )
        self.vision_projector = ModalityProjector(
            vision_d_model, text_d_model, projector_hidden_dim
        )

        # --- Audio: 输入是 mel-spectrogram (1 通道二维张量), 不是原始波形 ---
        # 频域特征更紧凑 (16kHz 1 秒波形 16000 点 -> 100 帧 mel), 模型训练更稳
        # 也方便复用 Whisper 等预训练 encoder 权重
        self.audio_encoder = PatchTransformerEncoder(
            patch_embed=PatchEmbed2D(
                input_size=audio_spec_size,
                patch_size=audio_patch_size,
                in_channels=audio_in_channels,
                embed_dim=audio_d_model,
            ),
            d_model=audio_d_model,
            n_heads=audio_n_heads,
            num_layers=audio_num_layers,
            dropout=dropout,
        )
        self.audio_resampler = self._maybe_resampler(
            audio_num_latents, audio_d_model, audio_n_heads,
            audio_num_latent_layers, dropout,
        )
        self.audio_projector = ModalityProjector(
            audio_d_model, text_d_model, projector_hidden_dim
        )

        # --- Video: 用 PatchEmbed3D (tubelet) 同时处理时间与空间, 节省 token 数 ---
        self.video_encoder = PatchTransformerEncoder(
            patch_embed=PatchEmbed3D(
                video_size=video_size,
                tubelet_size=video_tubelet_size,
                patch_size=video_patch_size,
                in_channels=video_in_channels,
                embed_dim=video_d_model,
            ),
            d_model=video_d_model,
            n_heads=video_n_heads,
            num_layers=video_num_layers,
            dropout=dropout,
        )
        self.video_resampler = self._maybe_resampler(
            video_num_latents, video_d_model, video_n_heads,
            video_num_latent_layers, dropout,
        )
        self.video_projector = ModalityProjector(
            video_d_model, text_d_model, projector_hidden_dim
        )

        # --- Thinker: 全模态理解主干, 直接复用 VL 解码器 (decoder-only LLM + M-RoPE) ---
        self.thinker = Qwen2VLDecoder(
            vocab_size=vocab_size,
            d_model=text_d_model,
            n_heads=text_n_heads,
            num_kv_heads=text_num_kv_heads,
            num_layers=text_num_layers,
            max_len=max_len,
            dropout=dropout,
            use_rope=use_rope,
            use_mrope=use_mrope,
        )

        # --- Talker: 流式语音解码器, 通常做得比 Thinker 小以降低延迟 ---
        if talker_d_model is None:
            talker_d_model = text_d_model

        self.talker = OmniTalkerDecoder(
            vocab_size=audio_vocab_size,
            d_model=talker_d_model,
            n_heads=talker_n_heads,
            num_kv_heads=talker_num_kv_heads,
            num_layers=talker_num_layers,
            max_len=talker_max_len,
            dropout=dropout,
            use_rope=use_rope,
        )

        # 维度桥接: 当 Talker 比 Thinker 窄时, 把 Thinker 隐状态线性投到 Talker 维度
        # 仅当维度不一致才创建, 同维度时省去这次投影
        self.thinker_to_talker: Optional[nn.Linear] = None
        if talker_d_model != text_d_model:
            self.thinker_to_talker = nn.Linear(text_d_model, talker_d_model, bias=False)

        # --- Modality Embedding (可选) ---
        self.use_modality_embedding = use_modality_embedding
        if use_modality_embedding:
            self.modality_embedding = nn.Embedding(4, text_d_model)
        else:
            self.modality_embedding = None

    @staticmethod
    def _maybe_resampler(
        num_latents: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> Optional[PerceiverResampler]:
        """工厂函数: num_latents <= 0 表示该模态不启用 Resampler (token 直接送 LLM)。"""
        if not num_latents or num_latents <= 0:
            return None
        return PerceiverResampler(
            num_latents=num_latents,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def _encode_modality(
        x: torch.Tensor,
        encoder: nn.Module,
        resampler: Optional[nn.Module],
        projector: nn.Module,
    ) -> torch.Tensor:
        """统一三阶段编码流水线 (encoder -> resampler? -> projector), 各模态共用。"""
        tokens = encoder(x)
        if resampler is not None:
            tokens = resampler(tokens)
        return projector(tokens)

    def _build_modality_embeddings(
        self,
        batch_size: int,
        segments: List[Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        按 segments 顺序为每段填充对应模态 ID, 然后查表得到 modality embedding。
        segments 形如 [(MODALITY_VISION, 64), (MODALITY_TEXT, 128), ...],
        最终输出 [B, T_total, D] 与拼接后的 combined_embeds 同形, 直接相加即可。
        """
        if self.modality_embedding is None:
            raise ValueError("modality_embedding 未启用")
        ids = [
            torch.full(
                (batch_size, seg_len), modality_id,
                dtype=torch.long, device=device,
            )
            for modality_id, seg_len in segments
        ]
        return self.modality_embedding(torch.cat(ids, dim=1))

    def _build_attention_mask(
        self,
        batch_size: int,
        segments: List[Tuple[int, int]],
        text_attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        非文本模态固定长度无 padding, 直接全 1; 文本段沿用调用者给的 mask。
        最后按 segments 顺序拼成 [B, T_total] 的有效位掩码。
        """
        if text_attention_mask is None:
            return None
        masks = []
        for modality_id, seg_len in segments:
            if modality_id == self.MODALITY_TEXT:
                masks.append(text_attention_mask)
            else:
                # 视觉/视频/音频经过 Resampler 后是定长 token, 全部有效
                masks.append(
                    torch.ones(
                        batch_size, seg_len, device=device,
                        dtype=text_attention_mask.dtype,
                    )
                )
        return torch.cat(masks, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio_spectrograms: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_input_ids: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            input_ids:           文本 [B, T_text]
            images:              图像 [B, 3, H, W] (可选)
            audio_spectrograms:  mel 声谱图 [B, 1, F, T_a] (可选)
            videos:              视频 [B, 3, T_v, H, W] (可选)
            text_attention_mask: 文本 padding mask [B, T_text]
            audio_input_ids:     传入则同时跑 Talker 生成 audio_logits, 否则只输出文本
            audio_attention_mask: Talker 自注意力的 padding mask
        """
        batch_size = input_ids.size(0)
        text_embeds = self.thinker.embed_tokens(input_ids)

        # 按固定顺序拼接: vision -> video -> audio -> text
        # segment_info 记录每段长度, 便于后续构造 modality embedding 与 mask
        modality_embeds_list: List[torch.Tensor] = []
        segment_info: List[Tuple[int, int]] = []

        if images is not None:
            vision_embeds = self._encode_modality(
                images, self.vision_encoder, self.vision_resampler, self.vision_projector
            )
            modality_embeds_list.append(vision_embeds)
            segment_info.append((self.MODALITY_VISION, vision_embeds.size(1)))

        if videos is not None:
            video_embeds = self._encode_modality(
                videos, self.video_encoder, self.video_resampler, self.video_projector
            )
            modality_embeds_list.append(video_embeds)
            segment_info.append((self.MODALITY_VIDEO, video_embeds.size(1)))

        if audio_spectrograms is not None:
            audio_embeds = self._encode_modality(
                audio_spectrograms,
                self.audio_encoder, self.audio_resampler, self.audio_projector,
            )
            modality_embeds_list.append(audio_embeds)
            segment_info.append((self.MODALITY_AUDIO, audio_embeds.size(1)))

        # 文本始终放最后 — 因果注意力下文本可看到全部模态前缀
        modality_embeds_list.append(text_embeds)
        segment_info.append((self.MODALITY_TEXT, text_embeds.size(1)))

        combined_embeds = torch.cat(modality_embeds_list, dim=1)

        combined_attention_mask = self._build_attention_mask(
            batch_size, segment_info, text_attention_mask, combined_embeds.device,
        )

        if self.use_modality_embedding:
            modality_embeds = self._build_modality_embeddings(
                batch_size, segment_info, combined_embeds.device,
            )
            combined_embeds = combined_embeds + modality_embeds

        # Thinker 同时返回 logits 与 hidden_states:
        #   logits      给文本输出 (与普通 LLM 一致)
        #   hidden      作为 Talker 的 cross-attention 条件源
        text_logits, thinker_hidden = self.thinker(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            return_hidden=True,
        )

        # Talker 是按需开启的: 训练或推理纯文本任务时可以省去, 节省显存与算力
        audio_logits = None
        if audio_input_ids is not None:
            context = thinker_hidden
            # 维度桥接 (Talker 比 Thinker 窄时)
            if self.thinker_to_talker is not None:
                context = self.thinker_to_talker(context)
            # context_mask 复用 combined_attention_mask: 让 Talker 忽略 Thinker 中的 padding
            audio_logits = self.talker(
                input_ids=audio_input_ids,
                context=context,
                attention_mask=audio_attention_mask,
                context_mask=combined_attention_mask,
            )

        if return_dict:
            return {
                "text_logits": text_logits,
                "audio_logits": audio_logits,
                "thinker_hidden_states": thinker_hidden,
            }
        return text_logits, audio_logits, thinker_hidden

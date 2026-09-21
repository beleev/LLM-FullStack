"""
Qwen2.5-Omni — 全模态输入 + 文本/语音双输出的 Thinker-Talker 架构 (Xu et al., 阿里, 2025)

是什么: Thinker = 吃 [图像; 视频; 音频; 文本] 前缀的 LLM (就是 Qwen2-VL 的 decoder), 输出文本;
       Talker = 小型自回归 decoder, 通过 cross-attention 读 Thinker 的隐状态, 输出离散语音 codec token。
解决了什么: 级联方案 (LLM 出文本 → 独立 TTS) 丢失语气/情绪, 且要等整句文本; Talker 直接读 Thinker 的隐状态
           (比文本信息多), 并可边想边说 (流式)。拆成两个模型也让文本 loss 和语音 loss 互不干扰。
关键公式: x = concat([vision; video; audio; text]) → Thinker → (text_logits, hidden)
         audio_logits = Talker(codec_tokens, context=hidden);   loss = CE_text + λ·CE_audio  (λ=0.5)
原版还有 TMRoPE (音视频按真实时间戳对齐位置); 本实现未做, 位置就是拼接后的下标。
读代码时盯住: thinker_hidden —— 两个"大脑"之间唯一的连接。
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNCrossBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding, SinPositionalEncoding
from llm_models.layers.multimodal import (
    ModalityProjector,
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResampler,
)
from llm_models.models.multimodal.qwen2_vl import Qwen2VLDecoder
from llm_models.utils.init import init_weights
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


# Thinker 与 Qwen2-VL 的 decoder 同构, 直接复用
OmniThinkerDecoder = Qwen2VLDecoder


class OmniTalkerDecoder(nn.Module):
    """
    Talker: 每层 = 因果 self-attn (GQA) + cross-attn (context = Thinker 隐状态) + SwiGLU, 全部 Pre-RMSNorm。

    词表是语音 codec 码本索引 (如 1024 个), 由外部 codec decoder 还原波形。
    用 cross-attn 而非 prefix: Thinker 隐状态可以流式追加, 不占 Talker 自己的上下文; Talker 可以很小 (低延迟)。
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
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)      # 不与 embedding 共享

        self.register_buffer("causal_mask", build_causal_mask(max_len, torch.device("cpu")), persistent=False)
        init_weights(self)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids) * self._embed_scale     # [B, T, D]

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        return self.causal_mask[:, :seq_len, :seq_len]                 # [1, T, T]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """input_ids [B, T_a] (codec token), context [B, S, D] (Thinker 隐状态), context_mask [B, S] -> [B, T_a, V_audio]"""
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

        self_mask = combine_causal_and_padding_mask(self._causal_mask(seq_len), attention_mask)   # [B|1, T, T]
        # cross-attn 只屏蔽 Thinker 侧的 padding: [B, S] -> [B, 1, S]
        cross_mask = context_mask.unsqueeze(1) if context_mask is not None and context_mask.dim() == 2 else context_mask

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
    image ─ ViT ──────────┐
    video ─ ViT(tubelet) ─┼ (Resampler) ─ Projector ─┐
    audio ─ ViT(mel) ─────┘                          ├ concat ─ Thinker ─┬─ text_logits
    text  ─ token embedding ─────────────────────────┘                   └─ hidden ─(cross-attn)─ Talker ─ audio_logits

    拼接顺序固定为 vision → video → audio → text; 文本在最后, 因果 mask 下看得到全部模态前缀。
    不传 audio_input_ids 就不跑 Talker (纯文本输出)。
    """

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
        if use_mrope:
            raise ValueError("本教学实现的 Omni.forward 不构造 [3, B, T] position_ids, use_mrope 未接通 (M-RoPE 演示见 qwen2_vl)")

        # --- Vision ---
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

        # --- Audio: mel 声谱图当作 1 通道图像切 patch (1 s 波形 16000 点 → 100 帧 mel, 紧凑得多) ---
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

        # --- Video: tubelet (时间×空间) 切块 ---
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

        # --- Thinker ---
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

        # --- Talker (通常比 Thinker 小得多) ---
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

        # Talker 与 Thinker 维度不同时, 先把 hidden 线性投到 Talker 维度
        self.thinker_to_talker: Optional[nn.Linear] = None
        if talker_d_model != text_d_model:
            self.thinker_to_talker = nn.Linear(text_d_model, talker_d_model, bias=False)

        self.use_modality_embedding = use_modality_embedding
        self.modality_embedding = nn.Embedding(4, text_d_model) if use_modality_embedding else None

        init_weights(self)

    @staticmethod
    def _maybe_resampler(
        num_latents: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> Optional[PerceiverResampler]:
        """num_latents <= 0 ⇒ 该模态不用 Resampler。"""
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
        """encoder -> (resampler) -> projector: x -> [B, N, D_text]"""
        tokens = encoder(x)
        if resampler is not None:
            tokens = resampler(tokens)
        return projector(tokens)

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
        input_ids [B, T]; images [B, 3, H, W]; audio_spectrograms [B, 1, F, T_a]; videos [B, 3, T_v, H, W] (后三者可选)
        audio_input_ids [B, T_audio]: 给了才跑 Talker
        -> text_logits [B, N_total, V], audio_logits [B, T_audio, V_audio] 或 None, thinker_hidden [B, N_total, D]
        """
        B = input_ids.size(0)
        parts: List[Tuple[int, torch.Tensor]] = []                     # (模态 id, [B, N, D]) 按固定顺序
        for modality, x, enc, res, proj in (
            (self.MODALITY_VISION, images, self.vision_encoder, self.vision_resampler, self.vision_projector),
            (self.MODALITY_VIDEO, videos, self.video_encoder, self.video_resampler, self.video_projector),
            (self.MODALITY_AUDIO, audio_spectrograms, self.audio_encoder, self.audio_resampler, self.audio_projector),
        ):
            if x is not None:
                parts.append((modality, self._encode_modality(x, enc, res, proj)))
        parts.append((self.MODALITY_TEXT, self.thinker.embed_tokens(input_ids)))

        combined_embeds = torch.cat([e for _, e in parts], dim=1)      # [B, N_total, D]
        device = combined_embeds.device

        if self.modality_embedding is not None:
            modality_ids = torch.cat([torch.full((e.size(1),), m, device=device) for m, e in parts])   # [N_total]
            combined_embeds = combined_embeds + self.modality_embedding(modality_ids)

        # 非文本段没有 padding: mask 全 1
        combined_attention_mask = None
        if text_attention_mask is not None:
            combined_attention_mask = torch.cat(
                [
                    text_attention_mask if m == self.MODALITY_TEXT else text_attention_mask.new_ones(B, e.size(1))
                    for m, e in parts
                ],
                dim=1,
            )                                                          # [B, N_total]

        text_logits, thinker_hidden = self.thinker(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            return_hidden=True,
        )

        audio_logits = None
        if audio_input_ids is not None:
            context = thinker_hidden                                   # [B, N_total, D]
            if self.thinker_to_talker is not None:
                context = self.thinker_to_talker(context)
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

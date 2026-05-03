"""
合成数据生成模块
================

为不同模型架构提供 "假" 训练数据生成器，用于教学/集成测试。
真实训练中应替换为读取 tokenized 语料的 DataLoader，但教学代码用合成数据
有几个好处：
    1. 零依赖，不需下载/预处理大规模语料；
    2. 可控的形状和词表，便于快速验证 forward / backward / loss 通路；
    3. 可保证模型 forward 不抛 shape 错误，方便对照学习。

每个生成器实现 `generate_batch()` 返回一个 dict，dict 中既包含
模型 `forward(**batch)` 所需的输入张量，也包含 "labels" (以及音频标签等)
供 LossComputer 使用。Trainer 会从 dict 中弹出 labels，剩余字段直接展开
传给 model。这种 "数据生成 + 损失" 解耦的策略让 Trainer 对模型类型保持中立。

生成器对照模型：
    - DecoderOnlyDataGenerator   : GPT-3, DeepSeekV3 / V3.2
    - EncoderDecoderDataGenerator: 经典 Transformer (Vaswani et al. 2017)
    - VisionLanguageDataGenerator: Qwen2-VL
    - OmniDataGenerator          : Qwen2.5-Omni (文本 + 图像 + 音频 + 视频)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch

from llm_models.utils.masks import get_pad_mask, get_subsequent_mask, combine_masks


class SyntheticDataGenerator(ABC):
    """
    合成数据生成器基类 (策略模式中的 "策略" 接口)。

    Trainer 持有一个 SyntheticDataGenerator 实例，每步训练调用一次
    `generate_batch()`，无需关心具体是哪种模态/架构。
    """

    @abstractmethod
    def generate_batch(self) -> Dict[str, torch.Tensor]:
        """
        生成一个训练 batch。

        Returns:
            dict, 必须满足:
              - 包含模型 `forward()` 所需的全部关键字参数；
              - 至少包含一个 "labels" 字段供 loss 计算；
              - 多模态模型可附加 "audio_labels" 等额外标签。
        """
        raise NotImplementedError


class DecoderOnlyDataGenerator(SyntheticDataGenerator):
    """
    Decoder-Only (因果 LM) 模型的合成数据生成器。

    适用模型: GPT-3, DeepSeekV3, DeepSeekV3_2

    Teacher Forcing 与 shift-by-one 标签:
        语言模型本质是 P(x_t | x_<t)，即给定前 t-1 个 token 预测第 t 个。
        训练时使用 "teacher forcing"：每个位置都喂入真实历史 (而非模型自己的预测)，
        并将 labels 设为输入右移一位。
        实现技巧：先生成长度 seq_len + 1 的序列 X，再切片：
            input  = X[:, :-1]   # 长度 seq_len
            labels = X[:,  1:]   # 长度 seq_len，每个位置的目标恰好是输入下一位

    为什么 token 从 1 开始 randint(1, vocab)？
        约定 0 通常作为 pad 索引；从 1 开始可避免合成数据中误生成 pad token，
        让所有位置都参与 loss 计算。

    Args:
        vocab_size: 词表大小。
        batch_size: 批次大小。
        seq_len:    输入序列长度。
        device:     张量所在设备。
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        # 多生成 1 个 token，便于切片得到对齐的 input / shifted labels。
        tokens = torch.randint(
            1, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device
        )
        return {
            "idx": tokens[:, :-1],     # 模型输入
            "labels": tokens[:, 1:],   # 标签 = 输入右移一位 (next-token prediction)
        }


class EncoderDecoderDataGenerator(SyntheticDataGenerator):
    """
    Encoder-Decoder (seq2seq) 模型的合成数据生成器。

    适用模型: 原版 Transformer (Vaswani et al. 2017)

    数据构造:
        - src        : 源语言 token, 由 encoder 自注意力处理
        - tgt_input  : 目标语言 teacher forcing 输入
        - labels     : 目标语言右移一位
        - src_mask   : 源 padding 掩码 (encoder 自注意力中屏蔽 pad)
        - tgt_mask   : 目标 padding 掩码 与 因果掩码 的并集
                       (decoder 自注意力既要屏蔽 pad，也要屏蔽未来 token)

    为什么需要 causal mask？
        训练时整个目标序列被并行喂入 decoder，但每个位置不应看到未来 token，
        否则就泄漏了答案。下三角 mask 在 softmax 前置 -inf，阻断未来信息。

    为什么 pad 不能参与计算？
        pad 是为对齐 batch 而填充的占位符，没有语义。若不掩掉，
        attention 会把 pad 当成有效 key，污染表示；loss 也会浪费在 pad 上。

    Args:
        src_vocab_size: 源词表大小。
        tgt_vocab_size: 目标词表大小。
        batch_size:     批次大小。
        src_len:        源序列长度。
        tgt_len:        目标序列长度。
        pad_idx:        padding token 的索引 (默认 0)。
        device:         设备。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        batch_size: int,
        src_len: int,
        tgt_len: int,
        pad_idx: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.batch_size = batch_size
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_idx = pad_idx
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        # 源序列从 1 开始，避开 pad_idx=0，保证全部位置有效。
        src = torch.randint(
            1, self.src_vocab_size, (self.batch_size, self.src_len), device=self.device
        )

        # 目标序列：先生成 tgt_len + 1 长度，再切片得到 input 与 shifted labels
        tgt_full = torch.randint(
            1, self.tgt_vocab_size, (self.batch_size, self.tgt_len + 1), device=self.device
        )
        tgt_input = tgt_full[:, :-1]  # decoder 输入 [B, tgt_len]
        labels = tgt_full[:, 1:]      # decoder 目标 [B, tgt_len]

        # 构造掩码
        # src_mask: 仅 padding mask, 形状 [B, 1, src_len]，可广播到 [B, H, T_q, src_len]
        src_mask = get_pad_mask(src, self.pad_idx)
        # tgt_mask: padding ∩ causal, 让 decoder 既忽略 pad 又看不到未来
        tgt_pad_mask = get_pad_mask(tgt_input, self.pad_idx)             # [B, 1, tgt_len]
        tgt_subsequent_mask = get_subsequent_mask(tgt_input)             # [1, tgt_len, tgt_len]
        tgt_mask = combine_masks(tgt_pad_mask, tgt_subsequent_mask)

        return {
            "src": src,
            "tgt": tgt_input,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "labels": labels,
        }


class VisionLanguageDataGenerator(SyntheticDataGenerator):
    """
    视觉语言模型 (VLM) 的合成数据生成器。

    适用模型: Qwen2-VL

    数据构造:
        - input_ids: 文本 token [B, seq_len]
        - images:    随机像素图 [B, 3, H, W]
        - labels:    在视觉 token 位置填 -100 (不计 loss)，文本位置为 next-token

    为什么 vision tokens 位置要填 -100？
        Qwen2-VL 的 forward 流程：图像 → vision encoder → resampler →
        固定数量 (num_vision_tokens) 的视觉 embedding，与文本 embedding 拼接成
        [vision_tokens, text_tokens] 一起进入 LLM。
        最终 logits 形状为 [B, num_vision_tokens + seq_len, V]。
        但视觉 token 不需要 "预测下一 token"——它们是给文本生成提供条件的，
        因此对应位置标签设为 -100，让 cross_entropy 通过 ignore_index=-100 跳过。
        若不忽略，模型会被强迫为视觉 embedding "预测" 一个无意义的随机文本 token，
        浪费容量并干扰真正的语言建模目标。

    Args:
        vocab_size:        文本词表大小。
        batch_size:        批次大小。
        seq_len:           文本序列长度。
        image_size:        图像边长 (H = W)。
        num_vision_tokens: resampler 后输出的视觉 token 数 (通常远小于 patch 数)。
        device:            设备。
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        image_size: int = 224,
        num_vision_tokens: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.num_vision_tokens = num_vision_tokens
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        # 文本 token: 多生成 1 个用于 shift
        text_tokens = torch.randint(
            1, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device
        )
        input_ids = text_tokens[:, :-1]      # 模型输入 [B, seq_len]
        text_labels = text_tokens[:, 1:]     # next-token 标签 [B, seq_len]

        # 图像: 模拟标准化后的像素 (高斯分布即可，反正训练几步看不出差别)
        images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device
        )

        # 视觉 token 位置: 全 -100, cross_entropy 会跳过这些位置
        vision_ignore = torch.full(
            (self.batch_size, self.num_vision_tokens), -100,
            dtype=torch.long, device=self.device,
        )
        # 拼接顺序必须和模型 forward 中 [vision, text] 的拼接顺序一致
        labels = torch.cat([vision_ignore, text_labels], dim=1)

        return {
            "input_ids": input_ids,
            "images": images,
            "labels": labels,
        }


class OmniDataGenerator(SyntheticDataGenerator):
    """
    全模态模型 (文本 + 视觉 + 音频 + 视频) 的合成数据生成器。

    适用模型: Qwen2.5-Omni (Thinker + Talker 双分支结构)

    Thinker / Talker 概念:
        - Thinker: 主 LLM，处理多模态输入并产出文本 logits；
        - Talker:  额外的小型自回归头，基于 Thinker 隐状态生成离散音频 token，
                   实现端到端 "说话"。
        因此本生成器同时给出 文本 labels 与 音频 labels。

    标签处理同 VLM:
        所有模态 (vision / audio_spec / video) 经各自 encoder + resampler 投影成
        固定数量的 embedding，拼接到文本前作为前缀；这些前缀位置的标签全部 -100，
        只有真正的文本 token 参与 LM loss。

    Args:
        vocab_size:         文本词表大小。
        audio_vocab_size:   音频离散 token 词表大小 (Talker 输出)。
        batch_size:         批次大小。
        seq_len:            文本序列长度。
        audio_seq_len:      音频 token 序列长度 (Talker 自回归长度)。
        image_size:         图像边长。
        audio_spec_size:    声谱图尺寸 (T, F)，T 为帧数, F 为梅尔/频率维度。
        video_size:         视频尺寸 (T, H, W)。
        num_vision_tokens:  resampler 后视觉 token 数。
        num_audio_tokens:   resampler 后音频 token 数。
        num_video_tokens:   resampler 后视频 token 数。
        device:             设备。
    """

    def __init__(
        self,
        vocab_size: int,
        audio_vocab_size: int,
        batch_size: int,
        seq_len: int,
        audio_seq_len: int = 16,
        image_size: int = 224,
        audio_spec_size: Tuple[int, int] = (256, 128),
        video_size: Tuple[int, int, int] = (8, 224, 224),
        num_vision_tokens: int = 64,
        num_audio_tokens: int = 64,
        num_video_tokens: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.audio_seq_len = audio_seq_len
        self.image_size = image_size
        self.audio_spec_size = audio_spec_size
        self.video_size = video_size
        self.num_vision_tokens = num_vision_tokens
        self.num_audio_tokens = num_audio_tokens
        self.num_video_tokens = num_video_tokens
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        # ---- 文本分支 ----
        text_tokens = torch.randint(
            1, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device
        )
        input_ids = text_tokens[:, :-1]
        text_labels = text_tokens[:, 1:]

        # ---- 多模态原始信号 (随机噪声充当占位)----
        images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device
        )
        audio_spectrograms = torch.randn(
            self.batch_size, 1, self.audio_spec_size[0], self.audio_spec_size[1],
            device=self.device,
        )
        t, h, w = self.video_size
        videos = torch.randn(
            self.batch_size, 3, t, h, w, device=self.device
        )

        # ---- 音频离散 token (Talker 输入 + 自回归标签)----
        audio_tokens = torch.randint(
            1, self.audio_vocab_size,
            (self.batch_size, self.audio_seq_len + 1), device=self.device,
        )
        audio_input_ids = audio_tokens[:, :-1]
        audio_labels = audio_tokens[:, 1:]

        # ---- 文本标签：模态前缀位置全部 -100 ----
        # 拼接顺序必须与模型 forward 内 [vision, video, audio, text] 的拼接顺序一致；
        # 否则 logits 与 labels 错位，loss 完全无意义。
        num_modality_tokens = (
            self.num_vision_tokens + self.num_video_tokens + self.num_audio_tokens
        )
        modality_ignore = torch.full(
            (self.batch_size, num_modality_tokens), -100,
            dtype=torch.long, device=self.device,
        )
        labels = torch.cat([modality_ignore, text_labels], dim=1)

        return {
            "input_ids": input_ids,
            "images": images,
            "audio_spectrograms": audio_spectrograms,
            "videos": videos,
            "audio_input_ids": audio_input_ids,
            "labels": labels,
            "audio_labels": audio_labels,
        }


# =============================================================================
# 新增: BERT / CLIP / Whisper / VAE / Diffusion / VAR 数据生成器
# =============================================================================


class MaskedLMDataGenerator(SyntheticDataGenerator):
    """
    BERT MLM 合成数据生成器。

    15% 概率做 mask 处理 (80% → [MASK] id, 10% → 随机 token, 10% 保持原样),
    仅 mask 位置的 label 是原 token, 其余位置填 -100。

    Args:
        vocab_size: 词表大小
        mask_token_id: [MASK] 的 id (约定为 vocab_size - 1, 避免与真实 token 冲突)
        mlm_prob:   总 mask 概率 (BERT 论文 0.15)
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        mlm_prob: float = 0.15,
        mask_token_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mlm_prob = mlm_prob
        # 默认约定: 最后一个 token id 当 [MASK], 真实 tokenizer 会自带
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size - 1
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        input_ids = torch.randint(
            1, self.vocab_size - 1, (self.batch_size, self.seq_len), device=self.device,
        )
        labels = torch.full_like(input_ids, -100)

        # 选中 mask 的位置 (按概率采样)
        mask = torch.rand_like(input_ids, dtype=torch.float) < self.mlm_prob
        labels[mask] = input_ids[mask]

        # 在 mask 位置里再分三份: 80% [MASK], 10% 随机, 10% 保持
        rand = torch.rand_like(input_ids, dtype=torch.float)
        mask_replace = mask & (rand < 0.8)
        mask_random = mask & (rand >= 0.8) & (rand < 0.9)
        # 剩下的 mask 区间 (rand >= 0.9) 保持原样, 不改 input_ids

        input_ids = torch.where(
            mask_replace,
            torch.full_like(input_ids, self.mask_token_id),
            input_ids,
        )
        random_tokens = torch.randint(
            1, self.vocab_size - 1, input_ids.shape, device=self.device,
        )
        input_ids = torch.where(mask_random, random_tokens, input_ids)

        return {"input_ids": input_ids, "labels": labels}


class CLIPDataGenerator(SyntheticDataGenerator):
    """
    CLIP 合成数据生成器.

    生成 B 对 (image, text), 对比 loss 的正样本由对角线隐式给出, 不需要额外 labels。
    "labels" 字段保留是为了符合 Trainer 的接口 (用 dummy 零张量)。

    Args:
        vocab_size:   文本词表
        batch_size:   每步样本对数
        text_len:     文本序列长度
        image_size:   图像边长
        eos_token_id: EOS token id (用于 pooler), 默认 vocab_size - 1
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        text_len: int,
        image_size: int = 224,
        eos_token_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.text_len = text_len
        self.image_size = image_size
        self.eos_token_id = eos_token_id if eos_token_id is not None else vocab_size - 1
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        text = torch.randint(
            1, self.vocab_size - 1, (self.batch_size, self.text_len), device=self.device,
        )
        # 在每行末尾填 EOS, 保证 CLIPTextEncoder 的 pooler 能找到位置
        text[:, -1] = self.eos_token_id

        images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device,
        )

        return {
            "images": images,
            "input_ids": text,
            "eos_token_id": self.eos_token_id,
            # labels 占位, ContrastiveLoss 不读它
            "labels": torch.zeros(self.batch_size, dtype=torch.long, device=self.device),
        }


class WhisperDataGenerator(SyntheticDataGenerator):
    """
    Whisper 合成数据生成器。

    - mel spectrogram: 随机张量, [B, n_mels, T_mel]
    - decoder 输入: 随机 token 序列, labels 为右移一位 (teacher forcing)

    Args:
        vocab_size:   文本词表
        n_mels:       mel 滤波器数 (Whisper 用 80)
        t_mel:        mel 帧数 (Whisper 30s@100Hz = 3000; 教学用更小)
        batch_size / tgt_len / device 同其他生成器
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        tgt_len: int = 32,
        n_mels: int = 80,
        t_mel: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.tgt_len = tgt_len
        self.n_mels = n_mels
        self.t_mel = t_mel
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        mel = torch.randn(self.batch_size, self.n_mels, self.t_mel, device=self.device)

        tokens = torch.randint(
            1, self.vocab_size, (self.batch_size, self.tgt_len + 1), device=self.device,
        )
        return {
            "mel": mel,
            "decoder_input_ids": tokens[:, :-1],
            "labels": tokens[:, 1:],
        }


class ImageDataGenerator(SyntheticDataGenerator):
    """
    通用图像合成数据生成器 (给 VAE / Tokenizer / VAR 用)。

    生成 [B, 3, H, W] 的随机像素图 (高斯噪声); labels = 同一张图 (VAE 自监督目标)。

    Args:
        batch_size:      batch 大小
        image_size:      边长
        image_channels:  通道 (默认 3)
    """

    def __init__(
        self,
        batch_size: int,
        image_size: int,
        image_channels: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        x = torch.randn(
            self.batch_size, self.image_channels, self.image_size, self.image_size,
            device=self.device,
        )
        return {"x": x, "labels": x}


class DiffusionDataGenerator(SyntheticDataGenerator):
    """
    扩散训练的合成数据生成器 (Image DiT / MM-DiT)

    每步:
        1) 采 x_0 (真实样本; 这里合成用随机 latent 代替)
        2) 采 t, 用 scheduler.add_noise 得 (x_t, target)
        3) 同时提供 y (类别) 或 text_embeds (MM-DiT)

    约定:
        batch 里的 "x" 为含噪 latent, "t" 为时间步, "labels" 为 target (noise / velocity);
        可选 "y", "text_embeds", "text_pooled" 等按需字段。
    """

    def __init__(
        self,
        scheduler,
        batch_size: int,
        latent_channels: int,
        latent_size: int,
        num_classes: int = 0,
        text_seq_len: Optional[int] = None,
        text_dim: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.text_seq_len = text_seq_len
        self.text_dim = text_dim
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        # 1) 模拟一批 "x_0" (真实训练应来自 VAE.encode)
        x0 = torch.randn(
            self.batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=self.device,
        )
        # 2) 采 t, 加噪
        t = self.scheduler.sample_timesteps(self.batch_size, self.device)
        noised = self.scheduler.add_noise(x0, t)

        batch = {
            "x": noised.noisy,
            "t": noised.t_norm,
            "labels": noised.target,
        }
        # 可选类别条件
        if self.num_classes > 0:
            batch["y"] = torch.randint(
                0, self.num_classes, (self.batch_size,), device=self.device,
            )
        # 可选文本条件 (MM-DiT)
        if self.text_seq_len is not None and self.text_dim is not None:
            batch["text_embeds"] = torch.randn(
                self.batch_size, self.text_seq_len, self.text_dim, device=self.device,
            )
            batch["text_pooled"] = torch.randn(
                self.batch_size, self.text_dim, device=self.device,
            )
        return batch


class VideoDiffusionDataGenerator(SyntheticDataGenerator):
    """
    Video DiT 训练合成数据, 与 DiffusionDataGenerator 同构但输入是 5D 视频 latent。
    """

    def __init__(
        self,
        scheduler,
        batch_size: int,
        latent_channels: int,
        latent_size: Tuple[int, int, int],   # (T', H', W')
        num_classes: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        T, H, W = self.latent_size
        x0 = torch.randn(
            self.batch_size, self.latent_channels, T, H, W, device=self.device,
        )
        t = self.scheduler.sample_timesteps(self.batch_size, self.device)
        noised = self.scheduler.add_noise(x0, t)

        batch = {"x": noised.noisy, "t": noised.t_norm, "labels": noised.target}
        if self.num_classes > 0:
            batch["y"] = torch.randint(
                0, self.num_classes, (self.batch_size,), device=self.device,
            )
        return batch


class VARImageDataGenerator(SyntheticDataGenerator):
    """
    VAR 训练数据: 只需原始图像, tokenizer 在模型内部离散化。
    """

    def __init__(
        self,
        batch_size: int,
        image_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device,
        )
        # VARLoss 从 model_output 里读 labels, 这里只是 Trainer 接口占位
        return {"images": images, "labels": images}

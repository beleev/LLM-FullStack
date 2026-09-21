"""infer / train 共用的 Tiny 配置 (CPU 秒级)。"""

V_TEXT, V_AUDIO = 500, 200
IMAGE, SPEC, VIDEO = 28, (32, 16), (2, 28, 28)                         # 图 28², mel (F=32, T=16), 视频 2 帧 28²
N_LAT = 4                                                              # 每个模态经 Resampler 压成 4 个 token

TINY = dict(
    vocab_size=V_TEXT, audio_vocab_size=V_AUDIO,
    text_d_model=64, text_n_heads=4, text_num_kv_heads=2, text_num_layers=2, max_len=64,
    vision_image_size=IMAGE, vision_patch_size=14, vision_d_model=64, vision_n_heads=4,
    vision_num_layers=1, vision_num_latents=N_LAT, vision_num_latent_layers=1,
    audio_spec_size=SPEC, audio_patch_size=(8, 8), audio_in_channels=1, audio_d_model=32, audio_n_heads=4,
    audio_num_layers=1, audio_num_latents=N_LAT, audio_num_latent_layers=1,
    video_size=VIDEO, video_tubelet_size=2, video_patch_size=14, video_in_channels=3, video_d_model=64,
    video_n_heads=4, video_num_layers=1, video_num_latents=N_LAT, video_num_latent_layers=1,
    talker_d_model=32, talker_n_heads=4, talker_num_kv_heads=2, talker_num_layers=1, talker_max_len=64,
    dropout=0.0,
)

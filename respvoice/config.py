from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    D: int = 768              # embedding dimension
    codebook_size: int = 8192 # VQ codebook size
    vq_beta: float = 0.25     # VQ commitment loss weight
    n_mels: int = 64          # mel bins
    sr: int = 16000           # sample rate
    target_sec: float = 8.0   # unified clip length
    win_ms: float = 64.0      # STFT window (ms)
    hop_ms: float = 32.0      # STFT hop (ms)
    patch_h: int = 8          # mel patch height
    patch_w: int = 8          # time patch width
    encoder_layers: int = 6   # transformer layers in encoder
    encoder_heads: int = 8    # attention heads
    predictor_layers: int = 2 # JEPA predictor depth
    mask_ratio: float = 0.60  # JEPA masking ratio
    n_sigreg_slices: int = 256 # SIGReg random directions
    backbone: str = "custom"  # "custom" | "opera_ct" | "ast"
    # VQ anti-collapse settings
    vq_use_ema: bool = True          # EMA codebook update
    vq_ema_decay: float = 0.99       # EMA decay rate
    vq_restart_threshold: int = 1    # re-seed codes used < N times per batch
    vq_restart_every: int = 1        # run restart check every N forward passes
    vq_l2_normalize: bool = True     # unit-sphere VQ (critical for D>=256)


@dataclass
class TrainConfig:
    # Stage 1 — LeJEPA self-supervised pretraining
    stage1_epochs: int = 100
    stage1_lr: float = 5e-4
    lam_sig: float = 0.02     # single LeJEPA trade-off hyperparameter

    # Stage 2 — VQ tokenizer (encoder frozen)
    stage2_epochs: int = 50
    stage2_lr: float = 1e-4
    lam_recon: float = 0.1    # optional reconstruction loss weight

    # Stage 3 — downstream fine-tuning
    stage3_epochs: int = 50
    stage3_lr: float = 1e-3
    lam_reg: float = 0.1      # regression auxiliary loss weight

    batch_size: int = 32
    weight_decay: float = 5e-2
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0


@dataclass
class RespVoiceConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_dir: str = "./data/audio"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

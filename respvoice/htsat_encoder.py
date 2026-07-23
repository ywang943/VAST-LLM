"""
Multi-Scale HTS-AT Encoder with Cross-Scale Attention Fusion (CSAF).

Architecture:
  Input: mel (B, 1, n_mels=64, T)
      ↓ [HTS-AT preprocessing: bn0 + reshape_wav2img]
      ↓ [patch_embed]
      ↓
  Stage 1 (Swin, dim=96,  64x64 res) → pool to 8x8 → E_micro  (B,64,192*)
  Stage 2 (Swin, dim=192, 32x32 res) → pool to 8x8 → E_meso   (B,64,384*)
  Stage 3 (Swin, dim=384, 16x16 res) → pool to 8x8 → E_mid    (B,64,768*)
  Stage 4 (Swin, dim=768,  8x8  res) →               E_macro  (B,64,768)
      ↓
  [Cross-Scale Attention Fusion — CSAF]
  Each temporal position (64 tokens) attends across 4 scales.
  Scale-type embeddings encode which temporal scale each representation comes from.
  Learned gating aggregates scale contributions.
      ↓
  E_acoustic (B, 64, 768)  →  LeJEPA predictor

*After PatchMerging inside each BasicLayer:
  After Stage 1 (layer[0]): 64×64=4096 → 32×32=1024 tokens, dim=192
  After Stage 2 (layer[1]): 32×32=1024 → 16×16=256  tokens, dim=384
  After Stage 3 (layer[2]): 16×16=256  →  8×8=64    tokens, dim=768
  After Stage 4 (layer[3]):  8×8=64    →  8×8=64    tokens, dim=768  (no downsamp)
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csa_fusion import CrossScaleAttentionFusion, PoolToResolution

OPERA_ROOT = Path(__file__).parent.parent / "opera_src"


class MultiScaleHTSATEncoder(nn.Module):
    """
    HTS-AT backbone with Cross-Scale Attention Fusion.

    Extracts multi-scale features from all 4 Swin stages and fuses them
    via Cross-Scale Attention, capturing short-time (jitter/shimmer ~1s)
    through long-time (F0 contour, spasmodic events ~4s) acoustic features.

    Args:
        ckpt_path:        path to OPERA-CT checkpoint
        D:                output dimension (768 = HTS-AT final dim)
        freeze_backbone:  whether to freeze HTS-AT weights during LeJEPA
    """

    # HTS-AT stage output dimensions (after PatchMerging)
    STAGE_DIMS = (192, 384, 768, 768)
    TARGET_L = 64    # 8×8 spatial grid = final resolution of HTS-AT
    D_OUT = 768

    def __init__(
        self,
        ckpt_path: Optional[str] = "checkpoints/opera_cache/encoder-operaCT.ckpt",
        D: int = 768,
        freeze_backbone: bool = False,
        use_csaf: bool = True,
    ):
        super().__init__()
        self.D = D
        self.use_csaf = use_csaf
        self._load_htsat(ckpt_path)

        if freeze_backbone:
            for p in self.htsat.parameters():
                p.requires_grad = False

        # Poolers: pool Stage 1 (1024 tokens) and Stage 2 (256 tokens) to 64
        self.pool1 = PoolToResolution(8, 8)   # 1024 → 64
        self.pool2 = PoolToResolution(8, 8)   # 256  → 64
        # Stage 3 and 4 already output 64 tokens — no pooling needed

        # Cross-Scale Attention Fusion
        self.csaf = CrossScaleAttentionFusion(
            D=D,
            n_scales=4,
            n_heads=8,
            depth=2,
            scale_dims=self.STAGE_DIMS,
        )

    # ------------------------------------------------------------------
    def _load_htsat(self, ckpt_path: Optional[str]):
        sys.path.insert(0, str(OPERA_ROOT))
        try:
            from src.model.htsat import config as htsat_config
            htsat_config.mel_bins = 64
            htsat_config.sample_rate = 16000
            htsat_config.window_size = 1024
            htsat_config.hop_size = 320
            from src.model.htsat.htsat import HTSATWrapper
        except Exception as e:
            raise RuntimeError(f"Cannot import HTSATWrapper: {e}")

        self.htsat_wrapper = HTSATWrapper()
        self.htsat = self.htsat_wrapper.htsat

        if ckpt_path is None:
            print("[MultiScaleHTSATEncoder] HTS-AT initialized from scratch")
            return

        p = Path(ckpt_path)
        if p.exists():
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"]
            enc_sd = {
                k.replace("encoder.encoder.htsat.", "", 1): v
                for k, v in sd.items()
                if k.startswith("encoder.encoder.htsat.")
            }
            missing, unexpected = self.htsat.load_state_dict(enc_sd, strict=False)
            loaded = len(enc_sd) - len(missing)
            print(f"[MultiScaleHTSATEncoder] Loaded {loaded}/{len(enc_sd)} weights "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            warnings.warn(f"[MultiScaleHTSATEncoder] checkpoint not found: {p}")

    # ------------------------------------------------------------------
    def _preprocess(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, 1, n_mels=64, T) → (B, 1, T, n_mels) → BN → reshape_wav2img → (B, 1, 256, 256)
        """
        # mel: (B, 1, F, T) → transpose to (B, 1, T, F)
        x = mel.transpose(2, 3)              # (B, 1, T, F)

        # BN0 expects (B, F, T, 1) layout
        x = x.transpose(1, 3)               # (B, F, T, 1)
        x = self.htsat.bn0(x)
        x = x.transpose(1, 3)               # (B, 1, T, F)

        # Pad/tile time axis and reshape to 2D spectrogram image (256×256)
        return self._reshape_for_htsat(x)   # (B, 1, 256, 256)

    def _reshape_for_htsat(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        target_T = int(self.htsat.spec_size * self.htsat.freq_ratio)  # 1024

        if T < target_T:
            repeats = (target_T // T) + 1
            x = x.repeat(1, 1, repeats, 1)
        x = x[:, :, :target_T, :]   # (B, 1, 1024, 64)

        return self.htsat.reshape_wav2img(x)   # (B, 1, 256, 256)

    # ------------------------------------------------------------------
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, n_mels=64, T)

        Returns:
            (B, 64, 768) — multi-scale fused patch-level representations
        """
        # Preprocessing
        x = self._preprocess(mel)                  # (B, 1, 256, 256)

        # Patch embedding
        x = self.htsat.patch_embed(x)              # (B, 4096, 96)
        if self.htsat.ape:
            x = x + self.htsat.absolute_pos_embed
        x = self.htsat.pos_drop(x)

        # ── Extract multi-scale features from each Swin stage ──────────
        # Each layer(x) returns (downsampled_x, attn):
        #   After layer[0]: (B, 1024, 192)  → captures ~1s context
        #   After layer[1]: (B,  256, 384)  → captures ~2s context
        #   After layer[2]: (B,   64, 768)  → captures ~4s context
        #   After layer[3]: (B,   64, 768)  → global context (no downsamp)

        x, _ = self.htsat.layers[0](x)    # (B, 1024, 192)  — micro scale
        e1 = self.pool1(x)                 # (B,   64, 192)

        x, _ = self.htsat.layers[1](x)    # (B,  256, 384)  — meso scale
        e2 = self.pool2(x)                 # (B,   64, 384)

        x, _ = self.htsat.layers[2](x)    # (B,   64, 768)  — mid scale
        e3 = x                             # (B,   64, 768)

        x, _ = self.htsat.layers[3](x)    # (B,   64, 768)  — macro scale
        x = self.htsat.norm(x)            # LayerNorm on final features
        e4 = x                             # (B,   64, 768)

        if not self.use_csaf:
            # Legacy mode: return Stage-4 features only (no CSAF fusion)
            # Used for backward compatibility with checkpoints trained without CSAF
            return e4

        # ── Cross-Scale Attention Fusion ────────────────────────────────
        # CSAF lets each temporal position attend across all 4 time scales,
        # enabling the model to correlate short-time jitter patterns with
        # long-time F0 trajectory context — matching the multi-scale design.
        fused = self.csaf([e1, e2, e3, e4])   # (B, 64, 768)

        return fused


def build_htsat_encoder(
    ckpt_path: Optional[str] = "checkpoints/opera_cache/encoder-operaCT.ckpt",
    freeze_backbone: bool = False,
    use_csaf: bool = True,
) -> MultiScaleHTSATEncoder:
    """
    Factory — returns the multi-scale HTS-AT + CSAF encoder.
    Set use_csaf=False for backward-compatible single-scale (Stage-4 only) mode.
    """
    return MultiScaleHTSATEncoder(
        ckpt_path=ckpt_path,
        freeze_backbone=freeze_backbone,
        use_csaf=use_csaf,
    )

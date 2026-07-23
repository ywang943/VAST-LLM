"""
Dual-Input Encoder: Mel spectrogram + HuBERT CNN raw waveform features.

Architecture:
  Mel (B,1,64,T)  → BN + reshape → (B,1,256,256)  ─┐
                                                      ├─ gated fusion → (B,1,256,256) → HTS-AT → CSAF
  Raw (B, samples) → HuBERT CNN → adapter → (B,1,256,256) ─┘

The two input streams are aligned to the same spatial shape and fused
via a learnable gate before entering the shared HTS-AT backbone.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from respvoice.htsat_encoder import MultiScaleHTSATEncoder

OPERA_ROOT = Path(__file__).parent.parent / "opera_src"


class WaveformAdapter(nn.Module):
    """Convert HuBERT CNN output (B, 512, T_cnn) → (B, 1, 256, 256) matching HTS-AT input."""

    def __init__(self, cnn_dim=512, target_T=1024, target_F=64):
        super().__init__()
        self.target_T = target_T
        self.target_F = target_F
        # Project 512-dim CNN features to 64-dim (matching Mel freq bins)
        self.freq_proj = nn.Linear(cnn_dim, target_F)
        self.norm = nn.LayerNorm(target_F)

    def forward(self, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        cnn_features: (B, 512, T_cnn) from HuBERT CNN
        Returns: (B, 1, 256, 256) matching HTS-AT expected input
        """
        B = cnn_features.shape[0]
        x = cnn_features.transpose(1, 2)  # (B, T_cnn, 512)
        x = self.freq_proj(x)             # (B, T_cnn, 64)
        x = self.norm(x)
        x = x.unsqueeze(1)                # (B, 1, T_cnn, 64)

        # Pad/tile time to target_T (1024)
        T = x.shape[2]
        if T < self.target_T:
            repeats = (self.target_T // T) + 1
            x = x.repeat(1, 1, repeats, 1)
        x = x[:, :, :self.target_T, :]    # (B, 1, 1024, 64)

        # reshape_wav2img: (B, 1, 1024, 64) → (B, 1, 256, 256)
        x = x.reshape(B, 1, 4, 256, 64)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, 256, 256)
        return x


class DualInputFusion(nn.Module):
    """Learnable gated fusion of Mel and waveform features."""

    def __init__(self):
        super().__init__()
        # Learnable gate: how much to weight waveform vs mel
        self.gate = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, mel_img: torch.Tensor, wav_img: torch.Tensor) -> torch.Tensor:
        """
        mel_img: (B, 1, 256, 256) from Mel preprocessing
        wav_img: (B, 1, 256, 256) from waveform adapter
        Returns: (B, 1, 256, 256) fused
        """
        stacked = torch.cat([mel_img, wav_img], dim=1)  # (B, 2, 256, 256)
        weights = self.gate(stacked)                      # (B, 2, 256, 256)
        fused = weights[:, 0:1] * mel_img + weights[:, 1:2] * wav_img
        return fused


class DualInputHTSATEncoder(MultiScaleHTSATEncoder):
    """
    Extends MultiScaleHTSATEncoder with a second raw-waveform input path.

    forward() accepts either:
      - mel only:         forward(mel)           → standard behavior
      - mel + waveform:   forward(mel, waveform) → dual-input fusion
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = "checkpoints/opera_cache/encoder-operaCT.ckpt",
        D: int = 768,
        freeze_backbone: bool = False,
        freeze_cnn: bool = True,
        use_csaf: bool = True,
    ):
        super().__init__(
            ckpt_path=ckpt_path, D=D,
            freeze_backbone=freeze_backbone, use_csaf=use_csaf,
        )

        # HuBERT CNN frontend
        from transformers import HubertModel
        hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_cnn = hubert.feature_extractor
        del hubert

        if freeze_cnn:
            for p in self.hubert_cnn.parameters():
                p.requires_grad = False

        self.wav_adapter = WaveformAdapter(cnn_dim=512)
        self.fusion = DualInputFusion()

    def forward(self, mel: torch.Tensor,
                waveform: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mel:      (B, 1, 64, T) Mel spectrogram
            waveform: (B, samples) raw audio at 16kHz, or None for mel-only

        Returns:
            (B, 64, 768) multi-scale fused representations
        """
        # Mel path
        mel_img = self._preprocess(mel)  # (B, 1, 256, 256)

        if waveform is not None:
            # Waveform path
            with torch.set_grad_enabled(
                any(p.requires_grad for p in self.hubert_cnn.parameters())
            ):
                cnn_out = self.hubert_cnn(waveform)  # (B, 512, T_cnn)
            wav_img = self.wav_adapter(cnn_out)       # (B, 1, 256, 256)

            # Fuse
            x = self.fusion(mel_img, wav_img)         # (B, 1, 256, 256)
        else:
            x = mel_img

        # Standard HTS-AT + CSAF pipeline
        x = self.htsat.patch_embed(x)
        if self.htsat.ape:
            x = x + self.htsat.absolute_pos_embed
        x = self.htsat.pos_drop(x)

        x, _ = self.htsat.layers[0](x)
        e1 = self.pool1(x)
        x, _ = self.htsat.layers[1](x)
        e2 = self.pool2(x)
        x, _ = self.htsat.layers[2](x)
        e3 = x
        x, _ = self.htsat.layers[3](x)
        e4 = self.htsat.norm(x)

        if self.use_csaf:
            return self.csaf([e1, e2, e3, e4])
        return e4


def build_dual_input_encoder(
    ckpt_path=None, D=768, freeze_backbone=False,
    freeze_cnn=True, use_csaf=True,
):
    return DualInputHTSATEncoder(
        ckpt_path=ckpt_path, D=D,
        freeze_backbone=freeze_backbone,
        freeze_cnn=freeze_cnn, use_csaf=use_csaf,
    )

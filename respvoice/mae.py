"""
MAE (Masked Autoencoder) pretraining for audio — comparison baseline.

Protocol:
  - Mask 60% of patches (same ratio as JEPA)
  - Lightweight pixel-space decoder reconstructs masked mel patches
  - Loss: MSE between reconstructed and original masked patches

Key difference from JEPA:
  JEPA:  predict latent REPRESENTATION of masked patches
  MAE:   reconstruct raw SIGNAL of masked patches

Both mask 60% of patches; JEPA operates in representation space, MAE in signal space.
The comparison tests whether predicting representations (JEPA) is better than
reconstructing the signal (MAE) for learning useful acoustic features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .jepa import random_masking


class MAEDecoder(nn.Module):
    """
    Lightweight pixel-space decoder for MAE pretraining.
    Takes masked encoder outputs + mask tokens, outputs reconstructed patches.
    """

    def __init__(self, D: int = 128, patch_h: int = 8, patch_w: int = 8,
                 depth: int = 2, num_heads: int = 4):
        super().__init__()
        patch_dim = patch_h * patch_w  # pixels per patch (before mel dim)
        # Project encoder dim → decoder dim, then reconstruct patch pixels
        self.proj = nn.Linear(D, D)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads,
            dim_feedforward=D * 4, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        # Project to patch pixel space (patch_h × patch_w per mel bin)
        # Actually reconstruct the mel values: D → patch_h * patch_w (for mel patches)
        self.head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, patch_h * patch_w),
        )

    def forward(self, z_ctx: torch.Tensor, ctx_idx: torch.Tensor,
                tgt_idx: torch.Tensor, L: int) -> torch.Tensor:
        """
        z_ctx: (B, N_ctx, D) context patch representations
        ctx_idx: (N_ctx,) indices of context patches
        tgt_idx: (N_mask,) indices of masked patches
        L: total sequence length

        Returns:
            pred: (B, N_mask, patch_h*patch_w) reconstructed patch values
        """
        B = z_ctx.size(0)
        # Build full sequence: context + mask tokens
        full = self.mask_token.expand(B, L, -1).clone()  # (B, L, D)
        full[:, ctx_idx, :] = self.proj(z_ctx)
        out = self.decoder(full)                          # (B, L, D)
        pred = self.head(out[:, tgt_idx, :])              # (B, N_mask, patch_h*patch_w)
        return pred


def mae_loss(
    encoder: nn.Module,
    decoder: MAEDecoder,
    mel: torch.Tensor,
    patch_h: int = 8,
    patch_w: int = 8,
    mask_ratio: float = 0.60,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MAE reconstruction loss.

    Args:
        encoder: acoustic encoder (produces patch representations)
        decoder: MAE decoder (reconstructs masked patches from signal)
        mel: (B, 1, n_mels, T) log-Mel spectrogram
        patch_h, patch_w: patch dimensions
        mask_ratio: fraction of patches to mask

    Returns:
        loss_mae: MSE loss on masked patches
        z_cont: (B, L, D) encoder representations (for SIGReg / diagnostics)
    """
    z_cont = encoder(mel)              # (B, L, D)
    B, L, D = z_cont.shape

    ctx_idx, tgt_idx = random_masking(L, mask_ratio, z_cont.device)

    z_ctx = z_cont[:, ctx_idx, :]     # (B, N_ctx, D)

    # Reconstruct masked patches from decoder
    pred = decoder(z_ctx, ctx_idx, tgt_idx, L)   # (B, N_mask, patch_h*patch_w)

    # Build ground-truth: extract masked patches from original mel
    # mel: (B, 1, n_mels, T) → patchify
    n_mels = mel.shape[2]
    T = mel.shape[3]
    nH = n_mels // patch_h   # number of patches along mel axis
    nW = T // patch_w        # number of patches along time axis

    # Patchify: (B, 1, nH*patch_h, nW*patch_w) → (B, nH*nW, patch_h*patch_w)
    mel_crop = mel[:, 0, :nH * patch_h, :nW * patch_w]   # (B, nH*ph, nW*pw)
    patches = mel_crop.reshape(B, nH, patch_h, nW, patch_w)
    patches = patches.permute(0, 1, 3, 2, 4).reshape(B, nH * nW, patch_h * patch_w)
    # patches: (B, L, patch_h*patch_w)

    target = patches[:, tgt_idx, :]   # (B, N_mask, patch_h*patch_w)

    loss_mae = F.mse_loss(pred, target.detach())
    return loss_mae, z_cont

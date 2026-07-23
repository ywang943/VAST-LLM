"""
COLA-style contrastive pretraining — comparison baseline.

Protocol (mirrored from OPERA-CT's contrastive approach):
  - Two augmented views of the same audio clip form a positive pair
  - All other clips in the batch form negatives
  - NT-Xent (InfoNCE) loss maximizes agreement between positive pairs

Augmentations used:
  - SpecAugment: time/frequency masking
  - Additive Gaussian noise
  - Random crop in time

Key difference from JEPA:
  JEPA:  predicts masked REGIONS from context (one view, spatial)
  COLA:  aligns two AUGMENTED VIEWS of the whole clip (two views, global)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ContrastiveHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps encoder output → normalized projection space.
    """

    def __init__(self, D: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(inplace=True),
            nn.Linear(D, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def pool_sequence(z: torch.Tensor) -> torch.Tensor:
    """(B, L, D) → (B, D) by mean pooling."""
    return z.mean(dim=1)


def audio_augment(mel: torch.Tensor, mode: str = "a") -> torch.Tensor:
    """
    Simple augmentation for contrastive learning on mel spectrograms.
    mel: (B, 1, n_mels, T)
    mode: 'a' or 'b' for two different augmentations
    """
    B, C, n_mels, T = mel.shape
    out = mel.clone()

    if mode == "a":
        # Time masking: mask 20% of time frames
        t_mask = max(1, int(T * 0.20))
        t_start = random.randint(0, T - t_mask)
        out[:, :, :, t_start:t_start + t_mask] = 0.0
        # Add small Gaussian noise
        out = out + torch.randn_like(out) * 0.05
    else:
        # Frequency masking: mask 20% of mel bins
        f_mask = max(1, int(n_mels * 0.20))
        f_start = random.randint(0, n_mels - f_mask)
        out[:, :, f_start:f_start + f_mask, :] = 0.0
        # Random time crop (shift)
        shift = random.randint(-T // 8, T // 8)
        out = torch.roll(out, shift, dims=-1)

    return out


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) contrastive loss for two views.
    z1, z2: (B, proj_dim) normalized projections
    """
    B = z1.size(0)
    # Concatenate: (2B, proj_dim)
    z = torch.cat([z1, z2], dim=0)
    # Cosine similarity matrix: (2B, 2B)
    sim = z @ z.T / temperature
    # Remove diagonal (self-similarity)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def cola_loss(
    encoder: nn.Module,
    proj_head: ContrastiveHead,
    mel: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    COLA contrastive loss.

    Args:
        encoder: acoustic encoder
        proj_head: projection head
        mel: (B, 1, n_mels, T)
        temperature: InfoNCE temperature

    Returns:
        loss_cola: contrastive loss scalar
        z_cont: (B, L, D) encoder representations (for diagnostics)
    """
    # Two augmented views
    mel_a = audio_augment(mel, mode="a")
    mel_b = audio_augment(mel, mode="b")

    # Encode
    z_a = encoder(mel_a)   # (B, L, D)
    z_b = encoder(mel_b)   # (B, L, D)

    # Pool to clip-level
    h_a = pool_sequence(z_a)   # (B, D)
    h_b = pool_sequence(z_b)   # (B, D)

    # Project
    p_a = proj_head(h_a)   # (B, proj_dim)
    p_b = proj_head(h_b)   # (B, proj_dim)

    loss = nt_xent_loss(p_a, p_b, temperature)
    return loss, z_a   # return z_a as the representation

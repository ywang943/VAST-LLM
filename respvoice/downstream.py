"""
Downstream task heads for RespVoice.

Two protocols (following OPERA benchmark):
  (a) Linear Probe: encoder + VQ frozen, only head trained
  (b) Fine-tune:    small lr on entire model

Supports:
  - Classification (disease type, severity category)
  - Regression (severity score)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Weighted average pooling via a learned query vector: (B,L,D) → (B,D)."""

    def __init__(self, D: int = 768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(D))
        self.scale = D ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        attn = (x @ self.query) * self.scale   # (B, L)
        attn = F.softmax(attn, dim=1)
        return (x * attn.unsqueeze(-1)).sum(1)  # (B, D)


class DownstreamHead(nn.Module):
    """
    Downstream task head: pooling + classification + optional regression.

    Args:
        D: encoder embedding dimension
        n_classes: number of target classes (set to 0 to disable classification)
        use_regression: add auxiliary regression head (severity score)
        dropout: dropout before head
    """

    def __init__(self, D: int = 768, n_classes: int = 6,
                 use_regression: bool = True, dropout: float = 0.1):
        super().__init__()
        self.pool = AttentionPool(D)
        self.dropout = nn.Dropout(dropout)
        self.n_classes = n_classes
        self.use_regression = use_regression

        if n_classes > 0:
            self.cls = nn.Linear(D, n_classes)
        if use_regression:
            self.reg = nn.Linear(D, 1)

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: (B, L, D) — can be z_q (quantized) or z_cont (continuous)

        Returns dict with "logits" and/or "score".
        """
        h = self.pool(z)        # (B, D)
        h = self.dropout(h)

        out = {}
        if self.n_classes > 0:
            out["logits"] = self.cls(h)       # (B, n_classes)
        if self.use_regression:
            out["score"] = self.reg(h).squeeze(-1)  # (B,)
        return out


def downstream_loss(
    outputs: dict,
    labels: torch.Tensor,
    scores: "Optional[torch.Tensor]" = None,
    lam_reg: float = 0.1,
) -> torch.Tensor:
    """
    Combined classification + regression loss.

    Args:
        outputs: dict from DownstreamHead.forward()
        labels: (B,) int class labels
        scores: (B,) float severity scores (optional)
        lam_reg: weight for regression term
    """
    loss = torch.tensor(0.0, device=labels.device)

    if "logits" in outputs:
        loss = loss + F.cross_entropy(outputs["logits"], labels)

    if "score" in outputs and scores is not None:
        loss = loss + lam_reg * F.mse_loss(outputs["score"], scores.float())

    return loss


class LinearProbe(nn.Module):
    """
    Linear evaluation protocol: frozen backbone, train only the linear head.
    Compatible with both z_cont and z_q inputs.
    """

    def __init__(self, D: int = 768, n_classes: int = 6):
        super().__init__()
        self.pool = AttentionPool(D)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.pool(z)
        h = self.norm(h)
        return self.head(h)    # (B, n_classes)

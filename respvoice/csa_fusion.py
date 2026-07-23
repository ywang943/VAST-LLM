"""
Temporal-Position-Aware Cross-Scale Attention Fusion (TPA-CSAF).

Motivation:
  Voice pathology involves three distinct time scales:
    - Short  (~50-500ms): jitter, shimmer — vocal fold irregularities
    - Medium (~500ms-2s): HNR, formant stability — vocal tract characteristics
    - Long   (~2-5s):    F0 contour, spasmodic events — neural/laryngeal control

  Key insight: the OPTIMAL scale fusion strategy differs across temporal positions.
    - Early onset positions (e.g., 0-15/64): often dominated by fine-grained Stage-1
      features capturing attack transients and jitter-like micro-irregularities.
    - Sustained middle positions (e.g., 20-50/64): Stage-2/3 features capture
      HNR stability and formant consistency over longer windows.
    - Trailing positions (e.g., 50-64/64): Stage-3/4 features capture F0
      trajectory completion and long-range temporal dependencies.

  Standard CSAF (position-agnostic): all 64 positions share the same cross-scale
  attention weights, treating onset and sustain identically.

  TPA-CSAF (temporal-position-aware): each position l ∈ [0, L) receives a
  learnable temporal position embedding added to its scale tokens BEFORE
  cross-scale attention. This lets the model learn position-specific fusion:
    "At position 5 (onset), weight Stage-1 more for jitter detection."
    "At position 50 (sustain), weight Stage-3/4 more for F0 analysis."

Architecture:
  Input: 4 scale features [E1, E2, E3, E4] each (B, L=64, D=768)

  For each temporal position l:
    1. Project each scale to D=768 (if needed)
    2. Add scale-type embedding  (which scale am I?)
    3. Add temporal-position embedding  (where in time am I?)  ← NEW
    4. Cross-scale attention over 4 scale tokens
    5. Soft gating → fused token

  Output: (B, L=64, D=768)

Comparison to vanilla CSAF:
  - +temporal_pos_embed: (max_len, D) learnable, ~49K params
  - Total params increase: negligible (~0.5% of 9.9M)
  - Expressivity: position l can learn scale-l-specific attention patterns
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleProjection(nn.Module):
    """Projects a single scale's feature to the common dimension D."""

    def __init__(self, in_dim: int, D: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
        ) if in_dim != D else nn.Sequential(nn.LayerNorm(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossScaleAttentionFusion(nn.Module):
    """
    Temporal-Position-Aware Cross-Scale Attention Fusion (TPA-CSAF).

    Adds explicit temporal position encoding to cross-scale attention,
    allowing the model to learn POSITION-SPECIFIC scale fusion strategies.

    Args:
        D:           common feature dimension (768 for HTS-AT)
        n_scales:    number of scales to fuse (4 Swin stages)
        n_heads:     attention heads for cross-scale attention
        depth:       number of cross-scale attention layers
        scale_dims:  input dims for each scale before projection
        max_len:     maximum sequence length for temporal embeddings (default 128)
    """

    def __init__(
        self,
        D: int = 768,
        n_scales: int = 4,
        n_heads: int = 8,
        depth: int = 2,
        scale_dims: tuple = (192, 384, 768, 768),
        max_len: int = 128,
    ):
        super().__init__()
        self.D = D
        self.n_scales = n_scales
        self.max_len = max_len

        # Project each scale to common dim D
        self.scale_projs = nn.ModuleList([
            ScaleProjection(dim, D) for dim in scale_dims
        ])

        # Scale-type embedding: encodes WHICH temporal scale a token comes from
        # Answers: "Am I a Stage-1 (fine-grained) or Stage-4 (global) token?"
        self.scale_type_embed = nn.Embedding(n_scales, D)

        # ── NEW: Temporal Position Embedding ──────────────────────────────
        # Encodes WHERE IN TIME this position is within the 8-second clip.
        # Added to ALL scale tokens at position l before cross-scale attention.
        # This enables the cross-scale attention to learn DIFFERENT fusion
        # weights depending on the temporal position:
        #   position  5/64 (~600ms):  onset, favor Stage-1 for jitter detection
        #   position 32/64 (~4s):    sustain, favor Stage-2/3 for HNR/F0
        #   position 60/64 (~7.5s):  trailing, favor Stage-4 for long patterns
        #
        # Initialized with sinusoidal encoding for faster convergence, then
        # made learnable so the model can specialize further.
        self.temporal_pos_embed = nn.Parameter(
            self._sinusoidal_init(max_len, D), requires_grad=True
        )
        # ─────────────────────────────────────────────────────────────────

        # Cross-scale Transformer: for each position, attend across 4 scales
        cross_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=n_heads,
            dim_feedforward=D * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.cross_scale_attn = nn.TransformerEncoder(cross_layer, num_layers=depth)

        # Learned gating: position-specific scale importance weights
        self.scale_gate = nn.Linear(D, 1)
        self.out_norm = nn.LayerNorm(D)

    @staticmethod
    def _sinusoidal_init(max_len: int, D: int) -> torch.Tensor:
        """Sinusoidal positional encoding for initialization."""
        pe = torch.zeros(max_len, D)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, D, 2).float() * (-math.log(10000.0) / D)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, D)

    def forward(self, scale_features: list) -> torch.Tensor:
        """
        Args:
            scale_features: list of n_scales tensors, each (B, L, dim_i)

        Returns:
            (B, L, D) temporally-aware fused multi-scale representation
        """
        assert len(scale_features) == self.n_scales

        B, L, _ = scale_features[0].shape
        device = scale_features[0].device

        # 1. Project each scale to common dim + add scale-type embedding
        scale_ids = torch.arange(self.n_scales, device=device)
        type_emb = self.scale_type_embed(scale_ids)  # (n_scales, D)

        projected = []
        for i, (feat, proj) in enumerate(zip(scale_features, self.scale_projs)):
            p = proj(feat)                                          # (B, L, D)
            p = p + type_emb[i].unsqueeze(0).unsqueeze(0)          # + scale_type
            projected.append(p)

        # 2. Stack: (B, L, n_scales, D)
        stacked = torch.stack(projected, dim=2)  # (B, L, n_scales, D)

        # 3. ── Add Temporal Position Embedding ──────────────────────────
        #    temporal_pos: (L, D) → (1, L, 1, D) → broadcast to (B, L, n_scales, D)
        #    This injects "when in the clip" into EVERY scale token at position l,
        #    so the cross-scale attention can differentiate:
        #      "I'm at onset position 5: I should attend more to fine-scale Stage-1"
        #      "I'm at sustain position 50: I should attend more to Stage-3/4"
        temp_pos = self.temporal_pos_embed[:L]                  # (L, D)
        temp_pos = temp_pos.unsqueeze(0).unsqueeze(2)           # (1, L, 1, D)
        stacked = stacked + temp_pos                            # (B, L, n_scales, D)
        # ────────────────────────────────────────────────────────────────

        # 4. Reshape and apply cross-scale attention
        BL = B * L
        x = stacked.reshape(BL, self.n_scales, self.D)          # (BL, n_scales, D)
        x = self.cross_scale_attn(x)                            # (BL, n_scales, D)

        # 5. Position-conditioned soft gating
        gates = F.softmax(self.scale_gate(x), dim=1)            # (BL, n_scales, 1)
        fused = (x * gates).sum(dim=1)                          # (BL, D)

        return self.out_norm(fused.reshape(B, L, self.D))


class PoolToResolution(nn.Module):
    """2D adaptive average pooling from token sequence to target_L tokens."""

    def __init__(self, target_h: int = 8, target_w: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((target_h, target_w))
        self.target_L = target_h * target_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        h = w = int(L ** 0.5)
        x2d = x.reshape(B, h, w, C).permute(0, 3, 1, 2)
        x2d = self.pool(x2d)
        return x2d.flatten(2).transpose(1, 2)               # (B, 64, C)

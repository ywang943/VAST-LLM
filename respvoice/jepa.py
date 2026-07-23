"""
JEPA (Joint Embedding Predictive Architecture) for audio.

LeJEPA variant: NO EMA teacher, NO stop-gradient on the online encoder.
Target is produced by the same encoder with stop-grad applied only to the
target slice, which is then regularized by SIGReg to prevent collapse.

Reference: audio.md Section 2.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """
    Lightweight predictor: given context patch representations and
    target patch positional queries, predicts target representations.

    Uses a small Transformer (2 layers by default as per audio.md).
    """

    def __init__(self, D: int = 768, depth: int = 2, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(D)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D,
            nhead=num_heads,
            dim_feedforward=int(D * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))

    def forward(self, z_context: torch.Tensor,
                target_pos_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_context: (B, N_ctx, D) context patch representations
            target_pos_emb: (B, N_mask, D) positional query for masked patches
        Returns:
            pred: (B, N_mask, D) predicted representations
        """
        query = self.query_norm(target_pos_emb)
        pred = self.decoder(query, z_context)   # cross-attend to context
        return self.head(pred)


def random_masking(L: int, mask_ratio: float, device: torch.device):
    """Returns (ctx_idx, tgt_idx) as sorted index tensors."""
    n_mask = max(1, int(L * mask_ratio))
    n_ctx = L - n_mask
    perm = torch.randperm(L, device=device)
    ctx_idx = perm[:n_ctx].sort().values
    tgt_idx = perm[n_ctx:].sort().values
    return ctx_idx, tgt_idx


def jepa_loss(
    encoder: nn.Module,
    predictor: JEPAPredictor,
    mel: torch.Tensor,
    pos_embed: torch.Tensor,
    mask_ratio: float = 0.60,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute JEPA prediction loss.

    LeJEPA does NOT use EMA teacher. Target = encoder(mel) with stop-grad
    on the target slice (the SIGReg loss on z_cont handles anti-collapse).

    Args:
        encoder: acoustic encoder f_θ
        predictor: JEPA predictor
        mel: (B, 1, n_mels, T)
        pos_embed: (B, L, D) positional embeddings for the full sequence
        mask_ratio: fraction of patches to mask

    Returns:
        loss_jepa: scalar smooth-L1 prediction loss
        z_cont: (B, L, D) full representations (for SIGReg)
        pred: (B, N_mask, D) predicted representations
    """
    z_all = encoder(mel)   # (B, L, D) — full sequence
    B, L, D = z_all.shape

    ctx_idx, tgt_idx = random_masking(L, mask_ratio, z_all.device)

    z_context = z_all[:, ctx_idx, :]              # (B, N_ctx, D)
    z_target = z_all[:, tgt_idx, :].detach()      # stop-grad on target slice only

    target_pos = pos_embed[:, tgt_idx, :]         # (B, N_mask, D) positional query

    pred = predictor(z_context, target_pos)       # (B, N_mask, D)

    loss = F.smooth_l1_loss(pred, z_target)
    return loss, z_all, pred

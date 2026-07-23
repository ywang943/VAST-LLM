"""
Vector Quantizer (VQ) — discrete tokenization of acoustic representations.

Key design choices:
- L2 normalization before quantization: puts z_cont and codebook on unit sphere,
  preventing high-norm collapse in high-dimensional (D≥256) encoders. This is
  the standard fix used in DALL-E, ViT-VQGAN, and SpeechTokenizer.
- EMA codebook update: stable cluster tracking without noisy batch gradients.
- Dead code restart: re-seeds unused codes from current batch, breaking the
  positive-feedback collapse cycle.
- Straight-through estimator: passes downstream gradients back through the
  discrete bottleneck to the encoder.

Paper ablation story:
  VQ alone (no norm)        → catastrophic collapse at D=768 (util<0.01)
  + EMA + restart (no norm) → restart thrashes (restarted=~K every batch)
  + L2 norm + EMA + restart → stable high utilization at any scale  ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    L2-normalized EMA vector quantizer with dead-code restart.

    Args:
        codebook_size: K — number of discrete codes
        D: codebook / embedding dimension
        beta: commitment loss weight
        use_ema: EMA codebook update (recommended)
        ema_decay: EMA decay rate
        restart_threshold: restart codes used fewer than N times per batch
        restart_every: run restart check every N forward passes
        l2_normalize: normalize z_cont and codebook to unit sphere before lookup.
                      Critical for high-dimensional encoders (D >= 256) to prevent
                      norm-dominated distance collapse.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        D: int = 768,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        restart_threshold: int = 1,
        restart_every: int = 1,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.D = D
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold
        self.restart_every = restart_every
        self.l2_normalize = l2_normalize
        self._step = 0

        self.codebook = nn.Embedding(codebook_size, D)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
        # Initialize codebook on unit sphere when using L2 normalization
        if l2_normalize:
            with torch.no_grad():
                self.codebook.weight.data = F.normalize(
                    self.codebook.weight.data, dim=-1
                )

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
            self.register_buffer("ema_embed_sum", self.codebook.weight.data.clone())
            self.codebook.weight.requires_grad = False

    # ------------------------------------------------------------------
    def forward(self, z_cont: torch.Tensor) -> dict:
        """
        Args:
            z_cont: (B, L, D) continuous representations (post SIGReg)

        Returns dict:
            z_q:         (B, L, D)   quantized (STE), in original z_cont space
            ids:         (B, L)      discrete token ids
            loss:        scalar      VQ loss
            util:        float       codebook utilization in [0, 1]
            perplexity:  float       effective codes used
            n_restarted: int         dead codes restarted this step
        """
        B, L, D = z_cont.shape
        flat = z_cont.detach().reshape(-1, D)  # (N, D)

        # --- L2 normalize for lookup (unit sphere) ---
        if self.l2_normalize:
            flat_n = F.normalize(flat, dim=-1)
            cb_n = F.normalize(self.codebook.weight, dim=-1)
        else:
            flat_n = flat
            cb_n = self.codebook.weight

        # --- Nearest-neighbour on (normalized) space ---
        # ||z_n - e_n||² = 2 - 2 * z_n · e_n  (since ||z_n||=||e_n||=1)
        # argmin = argmax of inner product
        sim = flat_n @ cb_n.T          # (N, K) cosine similarities
        ids_flat = sim.argmax(1)       # (N,)
        z_q_flat_n = cb_n[ids_flat]    # (N, D) normalized codebook entries

        # --- Codebook update ---
        n_restarted = 0
        if self.training:
            self._step += 1
            if self.use_ema:
                # EMA runs on *normalized* space to stay on sphere
                self._ema_update(flat_n, ids_flat)
            if self._step % self.restart_every == 0:
                n_restarted = self._restart_dead_codes(flat_n, ids_flat)

        # --- Loss ---
        # Project back to original scale for loss computation
        if self.l2_normalize:
            # Estimate the scale from z_cont norms, apply to normalized q
            scale = flat.norm(dim=-1, keepdim=True)        # (N, 1)
            z_q_flat = z_q_flat_n * scale                  # (N, D) rescaled
        else:
            z_q_flat = z_q_flat_n

        if self.use_ema:
            loss_vq = self.beta * F.mse_loss(
                z_q_flat.detach(), z_cont.reshape(-1, D)
            )
        else:
            loss_codebook = F.mse_loss(z_q_flat, z_cont.reshape(-1, D).detach())
            loss_commit   = F.mse_loss(z_q_flat.detach(), z_cont.reshape(-1, D))
            loss_vq = loss_codebook + self.beta * loss_commit

        # --- Straight-through estimator ---
        z_q_st = z_cont + (z_q_flat.reshape(B, L, D) - z_cont).detach()

        util, perplexity = self._metrics(ids_flat)

        return {
            "z_q": z_q_st,
            "ids": ids_flat.reshape(B, L),
            "loss": loss_vq,
            "util": util,
            "perplexity": perplexity,
            "n_restarted": n_restarted,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, flat_n: torch.Tensor, ids_flat: torch.Tensor):
        """EMA update in normalized space (keeps codebook on unit sphere)."""
        one_hot = F.one_hot(ids_flat, self.codebook_size).float()  # (N, K)

        new_cluster_size = one_hot.sum(0)                  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            new_cluster_size, alpha=1 - self.ema_decay
        )

        new_embed_sum = one_hot.T @ flat_n                 # (K, D) — normalized
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            new_embed_sum, alpha=1 - self.ema_decay
        )

        # Update codebook: normalize so entries stay on unit sphere
        # clamp avoids division by zero for dead codes
        n = self.ema_cluster_size.clamp(min=1e-5)
        updated = self.ema_embed_sum / n.unsqueeze(1)
        self.codebook.weight.data.copy_(F.normalize(updated, dim=-1))

    @torch.no_grad()
    def _restart_dead_codes(self, flat_n: torch.Tensor, ids_flat: torch.Tensor) -> int:
        """Re-seed dead codes from current normalized encoder outputs."""
        counts = torch.bincount(ids_flat, minlength=self.codebook_size)
        dead = (counts < self.restart_threshold).nonzero(as_tuple=False).squeeze(1)
        n_dead = dead.numel()
        if n_dead == 0:
            return 0

        n_avail = flat_n.size(0)
        perm = torch.randint(0, n_avail, (n_dead,), device=flat_n.device)
        seeds = F.normalize(flat_n[perm], dim=-1)  # unit sphere seeds

        self.codebook.weight.data[dead] = seeds.float()
        if self.use_ema:
            self.ema_cluster_size[dead] = 1.0
            self.ema_embed_sum[dead] = seeds.float()

        return int(n_dead)

    @torch.no_grad()
    def _metrics(self, ids_flat: torch.Tensor) -> tuple[float, float]:
        counts = torch.bincount(ids_flat, minlength=self.codebook_size).float()
        util = (counts > 0).sum().item() / self.codebook_size
        probs = counts / (counts.sum() + 1e-10)
        perplexity = (-(probs * (probs + 1e-10).log()).sum()).exp().item()
        return util, perplexity

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) → z_q: (B, L, D)"""
        return self.codebook(ids)


class OptionalDecoder(nn.Module):
    """Lightweight mel reconstruction decoder for Stage 2 auxiliary loss."""

    def __init__(self, D: int = 768, n_mels: int = 64, T_out: int = 250):
        super().__init__()
        self.T_out = T_out
        self.proj = nn.Linear(D, n_mels)
        self.norm = nn.LayerNorm(n_mels)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.proj(z_q)).permute(0, 2, 1)   # (B, n_mels, L)
        return F.interpolate(
            x.unsqueeze(1),
            size=(x.shape[1], self.T_out),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).unsqueeze(1)

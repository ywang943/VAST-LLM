"""
RespVoiceModel: unified acoustic foundation model.

Wires together:
  encoder (f_θ) → SIGReg → VQ → downstream head

Exposes per-stage forward methods so the trainer can call
the right combination depending on training stage.
"""

import math
import torch
import torch.nn as nn

from .config import ModelConfig
from .encoder import build_encoder
from .sigreg import SIGReg
from .jepa import JEPAPredictor, jepa_loss
from .vq import VectorQuantizer, OptionalDecoder
from .downstream import DownstreamHead


class RespVoiceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D

        # --- Encoder ---
        self.encoder = build_encoder(
            backbone=cfg.backbone,
            D=D,
            n_mels=cfg.n_mels,
            patch_h=cfg.patch_h,
            patch_w=cfg.patch_w,
            encoder_layers=cfg.encoder_layers,
            encoder_heads=cfg.encoder_heads,
        )

        # --- Shared positional embeddings (used by JEPA predictor) ---
        self.max_seq = 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq, D))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Stage 1: LeJEPA ---
        self.predictor = JEPAPredictor(
            D=D,
            depth=cfg.predictor_layers,
            num_heads=cfg.encoder_heads,
        )
        self.sigreg = SIGReg(n_slices=cfg.n_sigreg_slices)

        # --- Stage 2: VQ ---
        self.vq = VectorQuantizer(
            codebook_size=cfg.codebook_size,
            D=D,
            beta=cfg.vq_beta,
            use_ema=cfg.vq_use_ema,
            ema_decay=cfg.vq_ema_decay,
            restart_threshold=cfg.vq_restart_threshold,
            restart_every=cfg.vq_restart_every,
            l2_normalize=cfg.vq_l2_normalize,
        )
        self.decoder = OptionalDecoder(D=D, n_mels=cfg.n_mels)

        # --- Stage 3: Downstream head (placeholder; replaced per task) ---
        self.head = DownstreamHead(D=D, n_classes=2)  # binary default

    # ------------------------------------------------------------------
    # Stage 1: LeJEPA forward
    # ------------------------------------------------------------------

    def forward_stage1(self, mel: torch.Tensor, lam_sig: float = 0.02) -> dict:
        """
        Runs LeJEPA pretraining forward pass.

        L_total = L_jepa + lam_sig * L_sigreg

        Args:
            mel: (B, 1, n_mels, T)
            lam_sig: SIGReg trade-off weight (single hyperparameter)
        Returns:
            dict with losses and representations
        """
        B, _, n_mels, T = mel.shape
        L_approx = (n_mels // self.cfg.patch_h) * (T // self.cfg.patch_w)
        pos = self.pos_embed[:, :L_approx, :].expand(B, -1, -1)

        loss_jepa, z_cont, pred = jepa_loss(
            self.encoder, self.predictor, mel, pos,
            mask_ratio=self.cfg.mask_ratio,
        )
        loss_sigreg = self.sigreg(z_cont)
        loss_total = loss_jepa + lam_sig * loss_sigreg

        return {
            "loss": loss_total,
            "loss_jepa": loss_jepa,
            "loss_sigreg": loss_sigreg,
            "z_cont": z_cont,
        }

    # ------------------------------------------------------------------
    # Stage 2: VQ tokenizer forward (encoder frozen)
    # ------------------------------------------------------------------

    def forward_stage2(self, mel: torch.Tensor, lam_recon: float = 0.1) -> dict:
        """
        Trains VQ codebook on top of frozen encoder.

        L_total = L_vq + lam_recon * L_recon
        """
        with torch.no_grad():
            z_cont = self.encoder(mel)

        vq_out = self.vq(z_cont)
        loss_vq = vq_out["loss"]

        loss_recon = torch.tensor(0.0, device=mel.device)
        if lam_recon > 0:
            recon = self.decoder(vq_out["z_q"])
            target = mel[:, :, :, : recon.shape[-1]]
            loss_recon = ((recon - target) ** 2).mean()

        loss_total = loss_vq + lam_recon * loss_recon

        return {
            "loss": loss_total,
            "loss_vq": loss_vq,
            "loss_recon": loss_recon,
            "ids": vq_out["ids"],
            "util": vq_out["util"],
            "perplexity": vq_out["perplexity"],
            "n_restarted": vq_out.get("n_restarted", 0),
        }

    # ------------------------------------------------------------------
    # Stage 3: downstream forward (linear probe or fine-tune)
    # ------------------------------------------------------------------

    def forward_stage3(self, mel: torch.Tensor,
                       use_quantized: bool = True) -> dict:
        """
        Downstream task forward pass.

        Args:
            mel: (B, 1, n_mels, T)
            use_quantized: if True use z_q, else use z_cont

        Returns:
            dict with "logits", "score", "ids" (if quantized)
        """
        z_cont = self.encoder(mel)

        if use_quantized:
            vq_out = self.vq(z_cont)
            z = vq_out["z_q"]
            ids = vq_out["ids"]
        else:
            z = z_cont
            ids = None

        raw = self.head(z)
        # LinearProbe returns a raw tensor; DownstreamHead returns a dict
        if isinstance(raw, torch.Tensor):
            head_out = {"logits": raw}
        else:
            head_out = raw
        head_out["ids"] = ids
        return head_out

    # ------------------------------------------------------------------
    # Inference: encode → quantize → return token ids
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_to_tokens(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B,1,n_mels,T) → token ids (B, L)"""
        z_cont = self.encoder(mel)
        vq_out = self.vq(z_cont)
        return vq_out["ids"]

    # ------------------------------------------------------------------
    # Utility: freeze / unfreeze sections for staged training
    # ------------------------------------------------------------------

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_vq(self):
        for p in self.vq.parameters():
            p.requires_grad = False

    def set_downstream_head(self, head: nn.Module):
        self.head = head

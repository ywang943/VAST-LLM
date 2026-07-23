"""
Acoustic encoder: log-Mel (B,1,n_mels,T) → patch sequence z_cont (B,L,D).

Three strategies in priority order:
  1. opera_ct  — downloads OPERA-CT checkpoint from HuggingFace and loads encoder weights
  2. ast       — uses HuggingFace ASTModel (Audio Spectrogram Transformer)
  3. custom    — lightweight 2D-Conv patch embedding + Transformer (default, CPU-friendly)
"""

import math
import warnings
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Patch embedding (2D-Conv → flat patch sequence)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, n_mels: int = 64, patch_h: int = 8, patch_w: int = 8, D: int = 768):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.proj = nn.Conv2d(1, D, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        self.norm = nn.LayerNorm(D)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, n_mels, T)
        x = self.proj(mel)             # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', D)
        return self.norm(x)            # (B, L, D)


# ---------------------------------------------------------------------------
# Positional embedding (sinusoidal, works for any sequence length)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, D: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, D)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Custom lightweight encoder (default, no external deps beyond torch)
# ---------------------------------------------------------------------------

class CustomAcousticEncoder(nn.Module):
    def __init__(self, n_mels: int = 64, patch_h: int = 8, patch_w: int = 8,
                 D: int = 768, num_layers: int = 6, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(n_mels, patch_h, patch_w, D)
        self.pos_embed = SinusoidalPE(D)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=num_heads,
            dim_feedforward=int(D * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(D)
        self.D = D

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, n_mels, T)
        x = self.patch_embed(mel)   # (B, L, D)
        x = self.pos_embed(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.norm(x)         # (B, L, D)


# ---------------------------------------------------------------------------
# OPERA-CT wrapper: downloads checkpoint and maps encoder weights
# ---------------------------------------------------------------------------

class OperaCTEncoder(nn.Module):
    """
    Wraps OPERA-CT encoder from HuggingFace.
    Downloads the checkpoint and extracts the HTS-AT backbone weights.
    Falls back to CustomAcousticEncoder if download fails.
    """

    def __init__(self, D: int = 768, n_mels: int = 64, **kwargs):
        super().__init__()
        self.D = D
        self._load_opera_ct(D, n_mels)

    def _load_opera_ct(self, D: int, n_mels: int):
        try:
            from huggingface_hub import hf_hub_download
            import os

            print("[OperaCT] Downloading OPERA-CT checkpoint from HuggingFace...")
            ckpt_path = hf_hub_download(
                repo_id="evelyn0414/OPERA",
                filename="operaCT.pth",
                cache_dir="./checkpoints/opera_cache",
            )
            print(f"[OperaCT] Checkpoint saved to {ckpt_path}")

            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # OPERA-CT state dict has "encoder.*" keys
            enc_state = {
                k.replace("encoder.", "", 1): v
                for k, v in state.items()
                if k.startswith("encoder.")
            }

            # Build a custom encoder and try to load matched weights
            self.backbone = CustomAcousticEncoder(n_mels=n_mels, D=D)
            missing, unexpected = self.backbone.load_state_dict(enc_state, strict=False)
            loaded = len(enc_state) - len(missing)
            print(f"[OperaCT] Loaded {loaded}/{len(enc_state)} encoder weights "
                  f"({len(missing)} missing, {len(unexpected)} unexpected).")
            self._using_custom = True

        except Exception as e:
            warnings.warn(
                f"[OperaCT] Could not load OPERA-CT checkpoint ({e}). "
                "Falling back to randomly-initialized custom encoder."
            )
            self.backbone = CustomAcousticEncoder(n_mels=n_mels, D=D)
            self._using_custom = True

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.backbone(mel)


# ---------------------------------------------------------------------------
# AST wrapper (HuggingFace ASTModel)
# ---------------------------------------------------------------------------

class ASTEncoder(nn.Module):
    """Uses facebook/ast-finetuned-audioset-10-10-0.4593 as frozen backbone."""

    def __init__(self, D: int = 768, n_mels: int = 64, **kwargs):
        super().__init__()
        self.D = D
        self._load_ast(D)

    def _load_ast(self, D: int):
        try:
            from transformers import ASTModel
            print("[AST] Loading ASTModel from HuggingFace...")
            self.ast = ASTModel.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            ast_dim = self.ast.config.hidden_size  # typically 768
            self.proj = nn.Identity() if ast_dim == D else nn.Linear(ast_dim, D)
            self._ast_ok = True
            print(f"[AST] Loaded. Hidden size: {ast_dim}")
        except Exception as e:
            warnings.warn(f"[AST] Could not load ASTModel ({e}). Falling back to custom encoder.")
            self._ast_ok = False
            self.fallback = CustomAcousticEncoder(D=D)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if not self._ast_ok:
            return self.fallback(mel)
        # AST expects (B, T_frames, n_mels)
        # mel: (B, 1, n_mels, T) → squeeze and transpose
        x = mel.squeeze(1).transpose(1, 2)   # (B, T, n_mels)
        out = self.ast(input_values=x)
        hidden = out.last_hidden_state        # (B, L', ast_dim)
        return self.proj(hidden)              # (B, L', D)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_encoder(backbone: str = "custom", D: int = 768, n_mels: int = 64,
                  patch_h: int = 8, patch_w: int = 8,
                  encoder_layers: int = 6, encoder_heads: int = 8) -> nn.Module:
    """
    Factory for acoustic encoders.

    Args:
        backbone: "custom" | "opera_ct" | "ast"
        D: output embedding dimension
    """
    if backbone == "opera_ct":
        return OperaCTEncoder(D=D, n_mels=n_mels)
    elif backbone == "ast":
        return ASTEncoder(D=D, n_mels=n_mels)
    else:
        return CustomAcousticEncoder(
            n_mels=n_mels, patch_h=patch_h, patch_w=patch_w,
            D=D, num_layers=encoder_layers, num_heads=encoder_heads,
        )

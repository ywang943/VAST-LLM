"""
LeJEPA pretraining on top of OPERA-CT HTS-AT backbone.

This is the fair comparison:
  - Same backbone as OPERA-CT (HTS-AT, 31M params, initialized from OPERA-CT weights)
  - Our SSL method: LeJEPA (JEPA + SIGReg)
  - Same data: pretrain_zenodo_no_icbhi (34K windows)
  - Evaluation: ICBHI official split, linear probe, 5 seeds

If LeJEPA + HTS-AT > OPERA-CT (COLA + HTS-AT) on LP,
it proves LeJEPA is a better SSL METHOD for the same backbone.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.sigreg import SIGReg
from respvoice.jepa import JEPAPredictor, jepa_loss
from scripts.run_opera_icbhi_disease import official_split, train_stage3_auc
from scripts.run_full_local import (
    class_weights_from_labels, evaluate_binary, labels_from_subset
)
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.downstream import DownstreamHead

PRETRAIN_CACHE = "./data/mel_cache/pretrain_zenodo_no_icbhi"
LABEL_CACHE    = "./data/mel_cache/opera_icbhi_disease"
D = 768           # HTS-AT output dim
EPOCHS_PRETRAIN = 5
EPOCHS_S3 = 64
BATCH_SIZE = 16   # smaller due to HTS-AT size
SEEDS = [0, 1, 2, 3, 4]
LAM_SIG = 0.02


def pretrain_htsat_lejepa(pretrain_loader, device, out_dir):
    print(f"\n=== LeJEPA on HTS-AT backbone ===")
    print(f"  Backbone: OPERA-CT HTS-AT (31M params, 768-dim)")
    print(f"  SSL: LeJEPA (JEPA + SIGReg, lam_sig={LAM_SIG})")
    print(f"  Batch: {BATCH_SIZE}  Epochs: {EPOCHS_PRETRAIN}")

    encoder = build_htsat_encoder().to(device)
    predictor = JEPAPredictor(D=D, depth=2, num_heads=8).to(device)
    sigreg = SIGReg(n_slices=256).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = AdamW(params, lr=1e-4, weight_decay=5e-2)   # lower lr for pretrained backbone
    sched = CosineAnnealingLR(opt, T_max=EPOCHS_PRETRAIN * len(pretrain_loader))

    # Shared positional embedding (768-dim, 64 patches from HTS-AT)
    max_seq = 256
    pos_embed = nn.Parameter(torch.zeros(1, max_seq, D, device=device))
    nn.init.trunc_normal_(pos_embed, std=0.02)

    best_loss = float("inf")
    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        encoder.train(); predictor.train()
        stats = {"loss": 0, "jepa": 0, "sigreg": 0, "n": 0}

        for batch in tqdm(pretrain_loader, desc=f"Ep{epoch}/{EPOCHS_PRETRAIN}", leave=False):
            mel = batch["mel"].to(device)
            B = mel.size(0)
            pos = pos_embed[:, :64, :].expand(B, -1, -1)  # HTS-AT gives L=64

            opt.zero_grad()
            loss_jepa, z_cont, _ = jepa_loss(encoder, predictor, mel, pos, mask_ratio=0.60)
            loss_sig = sigreg(z_cont)
            loss = loss_jepa + LAM_SIG * loss_sig

            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()

            stats["loss"]   += loss.item() * B
            stats["jepa"]   += loss_jepa.item() * B
            stats["sigreg"] += loss_sig.item() * B
            stats["n"]      += B

        n = stats["n"]
        avg_loss = stats["loss"] / n
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"  [Ep{epoch}] loss={avg_loss:.4f}  "
              f"jepa={stats['jepa']/n:.4f}  sigreg={stats['sigreg']/n:.4f}")

    # Save encoder state
    ckpt_path = Path(out_dir) / "htsat_lejepa_encoder.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.state_dict(), "D": D}, str(ckpt_path))
    print(f"  Saved: {ckpt_path}")
    return str(ckpt_path)


def run_lp_seeds(encoder_ckpt_path, device, out_dir):
    """5-seed linear probe on ICBHI using HTS-AT+LeJEPA frozen encoder."""
    from respvoice.model import RespVoiceModel

    label_ds = CachedMelDataset(
        root=LABEL_CACHE,
        meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2)

    seed_results = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        cfg = RespVoiceConfig(
            model=ModelConfig(D=D, codebook_size=512, encoder_layers=2,
                              encoder_heads=8, predictor_layers=1),
            train=TrainConfig(stage3_epochs=EPOCHS_S3, batch_size=BATCH_SIZE,
                              stage3_lr=3e-4, num_workers=0),
            checkpoint_dir=f"{out_dir}/htsat_lp_seed{seed}",
            log_dir=f"{out_dir}/htsat_lp_log_seed{seed}",
        )

        # Build a model wrapper: use HTS-AT as encoder
        class HTSATRespVoice(RespVoiceModel):
            def __init__(self, cfg_m, enc_ckpt):
                super().__init__(cfg_m)
                # Replace default encoder with HTS-AT (keep on CPU initially)
                self.encoder = build_htsat_encoder()
                state = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
                self.encoder.load_state_dict(state["encoder_state"], strict=True)
                # DO NOT move to device here; caller does .to(device) after full init

        model = HTSATRespVoice(cfg.model, encoder_ckpt_path).to(device)

        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, generator=g)
        result = train_stage3_auc(
            cfg, model, train_loader, val_loader,
            n_classes=2, device=device,
            use_quantized=False, linear_probe=True,
            class_weights=weights,
        )

        ckpt = torch.load(result["best_path"], map_location="cpu", weights_only=False)
        m_eval = HTSATRespVoice(ckpt["config"].model, encoder_ckpt_path)
        m_eval.set_downstream_head(DownstreamHead(D, n_classes=2, use_regression=False))
        # Load state dict BEFORE moving to device to avoid mixed-device state
        compatible = {k: v for k, v in ckpt["model_state"].items()
                      if k in m_eval.state_dict() and
                      m_eval.state_dict()[k].shape == v.shape}
        m_eval.load_state_dict(compatible, strict=False)
        m_eval.to(device)  # Move AFTER loading to ensure consistent device
        test_res = evaluate_binary(m_eval, test_loader, device, use_quantized=False)
        test_auroc = float(test_res.get("auroc", 0.0))
        print(f"    seed {seed}: val={result['best_auc']:.4f}  test={test_auroc:.4f}")
        seed_results.append({"seed": seed, "val_auc": result["best_auc"],
                              "test_auroc": test_auroc})

    aurocs = [r["test_auroc"] for r in seed_results]
    return {
        "method": "HTS-AT backbone + LeJEPA (ours)",
        "backbone": "OPERA-CT HTS-AT (31M)",
        "auroc_mean": round(float(np.mean(aurocs)), 4),
        "auroc_std":  round(float(np.std(aurocs)), 4),
        "per_seed": [round(a, 4) for a in aurocs],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = "./checkpoints/htsat_lejepa"
    pretrain_ds = CachedMelDataset(
        root=PRETRAIN_CACHE,
        meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
        include_labels=False,
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True, num_workers=0)
    print(f"Pretrain windows: {len(pretrain_ds)}  batches/epoch: {len(pretrain_loader)}")

    # Stage 1: LeJEPA on HTS-AT
    enc_ckpt = pretrain_htsat_lejepa(pretrain_loader, device, out_dir)

    # Stage 3: Linear probe (5 seeds)
    print("\nRunning linear probe (5 seeds)...")
    result = run_lp_seeds(enc_ckpt, device, out_dir)

    print(f"\n{'='*60}")
    print(f"  HTS-AT + LeJEPA LP AUROC: {result['auroc_mean']:.3f} +- {result['auroc_std']:.3f}")
    print(f"  per-seed: {result['per_seed']}")
    print(f"  Compare:")
    print(f"    OPERA-CT (COLA + HTS-AT, bs=32 repro): 0.812 +- 0.011")
    print(f"    D128 LeJEPA (our lightweight backbone): 0.770 +- 0.034")
    print(f"{'='*60}")

    out_path = Path(out_dir) / "htsat_lejepa_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
SSL Method Comparison: MAE vs COLA vs LeJEPA (ours)

All conditions use the SAME backbone (D=128, 2-layer Transformer),
SAME pretraining data (pretrain_zenodo_no_icbhi, 34K windows),
SAME downstream eval (ICBHI official split, 5 seeds, LINEAR PROBE).

This isolates the SSL METHOD contribution, controlling for backbone and data.

Conditions:
  MAE   - reconstruct masked mel patches (OPERA-GT style)
  COLA  - contrastive learning on augmented views (OPERA-CT style)
  JEPA  - predict latent representations of masked patches (no SIGReg)
  LeJEPA- JEPA + SIGReg (ours)

Existing results (from previous runs, reused here):
  JEPA only  : LP 0.551 +- 0.128
  LeJEPA     : LP 0.770 +- 0.034
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.encoder import build_encoder
from respvoice.mae import MAEDecoder, mae_loss
from respvoice.cola import ContrastiveHead, cola_loss
from scripts.run_opera_icbhi_disease import official_split, train_stage3_auc
from scripts.run_full_local import class_weights_from_labels, evaluate_binary, labels_from_subset
from respvoice.model import RespVoiceModel
from respvoice.downstream import DownstreamHead

PRETRAIN_CACHE = "./data/mel_cache/pretrain_zenodo_no_icbhi"
LABEL_CACHE    = "./data/mel_cache/opera_icbhi_disease"
D, LAYERS, HEADS = 128, 2, 4
EPOCHS_PRETRAIN = 5
EPOCHS_S3 = 64
BATCH_SIZE = 64
SEEDS = [0, 1, 2, 3, 4]


def make_encoder():
    return build_encoder(backbone="custom", D=D, n_mels=64,
                         patch_h=8, patch_w=8,
                         encoder_layers=LAYERS, encoder_heads=HEADS)


def pretrain_mae(pretrain_loader, device, out_dir):
    """Pretraining with MAE (masked autoencoder)."""
    print("\n=== MAE Pretraining ===")
    encoder = make_encoder().to(device)
    decoder = MAEDecoder(D=D, patch_h=8, patch_w=8, depth=2, num_heads=HEADS).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=5e-2)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS_PRETRAIN * len(pretrain_loader))

    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        encoder.train(); decoder.train()
        losses = []
        for batch in tqdm(pretrain_loader, desc=f"MAE Ep{epoch}/{EPOCHS_PRETRAIN}", leave=False):
            mel = batch["mel"].to(device)
            opt.zero_grad()
            loss, _ = mae_loss(encoder, decoder, mel, mask_ratio=0.60)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        print(f"  [MAE Ep{epoch}] loss={np.mean(losses):.4f}")

    ckpt_path = str(Path(out_dir) / "mae_encoder.pt")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.state_dict()}, ckpt_path)
    print(f"  Saved: {ckpt_path}")
    return ckpt_path


def pretrain_cola(pretrain_loader, device, out_dir):
    """Pretraining with COLA (contrastive learning)."""
    print("\n=== COLA Contrastive Pretraining ===")
    encoder  = make_encoder().to(device)
    proj_head = ContrastiveHead(D=D, proj_dim=64).to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=5e-2)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS_PRETRAIN * len(pretrain_loader))

    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        encoder.train(); proj_head.train()
        losses = []
        for batch in tqdm(pretrain_loader, desc=f"COLA Ep{epoch}/{EPOCHS_PRETRAIN}", leave=False):
            mel = batch["mel"].to(device)
            opt.zero_grad()
            loss, _ = cola_loss(encoder, proj_head, mel)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        print(f"  [COLA Ep{epoch}] loss={np.mean(losses):.4f}")

    ckpt_path = str(Path(out_dir) / "cola_encoder.pt")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.state_dict()}, ckpt_path)
    print(f"  Saved: {ckpt_path}")
    return ckpt_path


def run_lp_seeds(encoder_ckpt, method_name, device, out_base):
    """Run 5-seed linear probe on ICBHI using a pretrained encoder."""
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
        import random; random.seed(seed)
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cfg = RespVoiceConfig(
            model=ModelConfig(D=D, codebook_size=512,
                              encoder_layers=LAYERS, encoder_heads=HEADS,
                              predictor_layers=1, n_sigreg_slices=32),
            train=TrainConfig(stage3_epochs=EPOCHS_S3, batch_size=BATCH_SIZE,
                              stage3_lr=3e-4, num_workers=0),
            checkpoint_dir=f"{out_base}/{method_name}_seed{seed}",
            log_dir=f"{out_base}/{method_name}_log_seed{seed}",
        )
        model = RespVoiceModel(cfg.model)

        # Load only encoder weights
        if encoder_ckpt:
            raw = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            enc_state = raw.get("encoder_state") or {
                k.replace("encoder.", "", 1): v
                for k, v in raw.get("model_state", {}).items()
                if k.startswith("encoder.")
            }
            cur = {k: v for k, v in model.encoder.state_dict().items()}
            compat = {k: v for k, v in enc_state.items()
                      if k in cur and cur[k].shape == v.shape}
            model.encoder.load_state_dict(compat, strict=False)

        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, generator=g)

        result = train_stage3_auc(
            cfg, model, train_loader, val_loader,
            n_classes=2, device=device,
            use_quantized=False, linear_probe=True,
            class_weights=weights,
        )

        # Evaluate on test
        ckpt = torch.load(result["best_path"], map_location="cpu", weights_only=False)
        m_eval = RespVoiceModel(ckpt["config"].model)
        m_eval.set_downstream_head(DownstreamHead(D, n_classes=2, use_regression=False))
        m_eval.load_state_dict(ckpt["model_state"], strict=False)
        m_eval.to(device)
        test_res = evaluate_binary(m_eval, test_loader, device, use_quantized=False)
        test_auroc = float(test_res.get("auroc", 0.0))
        print(f"    seed {seed}: val={result['best_auc']:.4f}  test={test_auroc:.4f}")
        seed_results.append({"seed": seed, "val_auc": result["best_auc"],
                              "test_auroc": test_auroc})

    aurocs = [r["test_auroc"] for r in seed_results]
    return {
        "method": method_name,
        "auroc_mean": round(float(np.mean(aurocs)), 4),
        "auroc_std":  round(float(np.std(aurocs)), 4),
        "per_seed": [round(a, 4) for a in aurocs],
        "seed_results": seed_results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["MAE", "COLA"],
                   choices=["MAE", "COLA"])
    p.add_argument("--out-dir", default="./checkpoints/ssl_comparison")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pretrain_ds = CachedMelDataset(
        root=PRETRAIN_CACHE,
        meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
        include_labels=False,
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True, num_workers=0)
    print(f"Pretrain windows: {len(pretrain_ds)}")

    results = {
        "JEPA_only": {
            "method": "JEPA only (no SIGReg)",
            "auroc_mean": 0.551, "auroc_std": 0.128,
            "note": "from previous ablation run",
        },
        "LeJEPA": {
            "method": "LeJEPA (JEPA + SIGReg, ours)",
            "auroc_mean": 0.770, "auroc_std": 0.034,
            "note": "from previous ablation run",
        },
        "OPERA_CT": {
            "method": "OPERA-CT (COLA, HTS-AT, AudioSet)",
            "auroc_mean": 0.812, "auroc_std": 0.011,
            "note": "reference (different backbone+data)",
        },
    }

    if "MAE" in args.methods:
        ckpt = pretrain_mae(pretrain_loader, device, args.out_dir)
        print(f"\nRunning MAE linear probe (5 seeds)...")
        r = run_lp_seeds(ckpt, "MAE", device, args.out_dir)
        results["MAE"] = r
        print(f"  MAE: {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}")

    if "COLA" in args.methods:
        ckpt = pretrain_cola(pretrain_loader, device, args.out_dir)
        print(f"\nRunning COLA linear probe (5 seeds)...")
        r = run_lp_seeds(ckpt, "COLA", device, args.out_dir)
        results["COLA"] = r
        print(f"  COLA: {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}")

    print("\n" + "=" * 60)
    print("  SSL METHOD COMPARISON (Linear Probe, D=128, same backbone)")
    print("=" * 60)
    order = ["JEPA_only", "MAE", "COLA", "LeJEPA", "OPERA_CT"]
    for k in order:
        if k in results:
            r = results[k]
            note = f"  [{r.get('note', '')}]" if r.get("note") else ""
            print(f"  {r['method']:<45}  {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}{note}")

    out = Path(args.out_dir) / "ssl_comparison_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out}")


if __name__ == "__main__":
    main()

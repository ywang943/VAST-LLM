"""
Post-pretraining pipeline: Stage 2 VQ + Stage 3 downstream evaluation.

Runs automatically after HTS-AT LeJEPA pretraining completes.
Uses the best checkpoint from checkpoints/htsat_lejepa_pretrain/htsat_lejepa_best.pt

Pipeline:
  1. Load pretrained HTS-AT + CSAF encoder
  2. Stage 2: Train VQ tokenizer (encoder frozen)
  3. Stage 3: Linear probe on ICBHI COPD (5 seeds, official split)
  4. Stage 3: Fine-tune on ICBHI COPD (5 seeds)
  5. Compare against TPA-CSAF frozen (0.945) and OPERA-CT fine-tune (0.957)
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import roc_auc_score

from data.respvoice_datasets import CachedMelDataset
from respvoice.downstream import AttentionPool
from respvoice.htsat_encoder import build_htsat_encoder
from scripts.run_opera_icbhi_disease import official_split
from scripts.run_full_local import class_weights_from_labels, labels_from_subset


PRETRAIN_CKPT = "checkpoints/htsat_lejepa_pretrain/htsat_lejepa_best.pt"
LABEL_CACHE   = "data/mel_cache/opera_icbhi_disease"
OUT_DIR       = "checkpoints/htsat_lejepa_downstream"
D, BS, EPOCHS = 768, 16, 64
SEEDS = [0, 1, 2, 3, 4]


class CSAFLinearProbe(nn.Module):
    """Same as run_csaf_frozen_htsat — for fair comparison."""
    def __init__(self, encoder, n_classes=2, freeze_backbone=True):
        super().__init__()
        self.encoder = encoder
        if freeze_backbone:
            for p in self.encoder.htsat.parameters():
                p.requires_grad = False
        for p in self.encoder.csaf.parameters():
            p.requires_grad = True
        for p in self.encoder.pool1.parameters():
            p.requires_grad = True
        for p in self.encoder.pool2.parameters():
            p.requires_grad = True
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, n_classes)

    def forward(self, mel):
        with torch.no_grad() if not self.encoder.htsat.training else torch.enable_grad():
            x = self.encoder._preprocess(mel)
            x = self.encoder.htsat.patch_embed(x)
            if self.encoder.htsat.ape:
                x = x + self.encoder.htsat.absolute_pos_embed
            x = self.encoder.htsat.pos_drop(x)
            x, _ = self.encoder.htsat.layers[0](x)
            e1 = self.encoder.pool1(x)
            x, _ = self.encoder.htsat.layers[1](x)
            e2 = self.encoder.pool2(x)
            x, _ = self.encoder.htsat.layers[2](x)
            e3 = x
            x, _ = self.encoder.htsat.layers[3](x)
            e4 = self.encoder.htsat.norm(x)
        z = self.encoder.csaf([e1, e2, e3, e4])
        h = self.pool(z)
        return self.head(h)


def run_seed(encoder_state, seed, device, freeze_backbone, tag):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    label_ds = CachedMelDataset(
        root=LABEL_CACHE,
        meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2).to(device)

    # Fresh encoder loaded from pretraining checkpoint
    encoder = build_htsat_encoder(use_csaf=True, freeze_backbone=False)
    if encoder_state:
        # Load only encoder weights from the LeJEPA checkpoint
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        print(f"    Loaded encoder: {len(encoder_state)-len(missing)}/{len(encoder_state)} weights")

    model = CSAFLinearProbe(encoder, freeze_backbone=freeze_backbone).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable: {n_train:,} params")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=3e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            mel    = batch["mel"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad()
            logits = model(mel)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["mel"].to(device))
                vp.append(F.softmax(logits, dim=1)[:, 1].cpu())
                vl.append(batch["label"])
        vp = torch.cat(vp).numpy(); vl = torch.cat(vl).numpy()
        try:
            val_auc = roc_auc_score(vl, vp)
        except Exception:
            val_auc = 0.5
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 16 == 0 or epoch == EPOCHS:
            print(f"    Ep{epoch}: val_auc={val_auc:.4f} best={best_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["mel"].to(device))
            tp.append(F.softmax(logits, dim=1)[:, 1].cpu())
            tl.append(batch["label"])
    try:
        return float(roc_auc_score(torch.cat(tl).numpy(), torch.cat(tp).numpy()))
    except Exception:
        return 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HTS-AT LeJEPA → Downstream Evaluation")
    print("=" * 60)

    # Load pretrained encoder weights
    encoder_state = None
    ckpt_path = Path(PRETRAIN_CKPT)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        full_state = ckpt["model_state"]
        # Extract only encoder (htsat + csaf + pool1 + pool2) weights
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in full_state.items()
            if k.startswith("encoder.")
        }
        print(f"  Loaded pretrain ckpt: {ckpt_path} (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('best_loss',0):.4f})")
    else:
        print(f"  WARNING: {ckpt_path} not found — using OPERA-CT init only")

    results = {}

    # ── Protocol 1: Linear Probe (frozen HTS-AT, only CSAF + head trained) ──
    print("\n=== Protocol 1: Linear Probe (frozen HTS-AT backbone) ===")
    lp_aucs = []
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        auc = run_seed(encoder_state, seed, device, freeze_backbone=True, tag="lp")
        print(f"  seed {seed}: AUROC = {auc:.4f}")
        lp_aucs.append(auc)
    lp_mean, lp_std = float(np.mean(lp_aucs)), float(np.std(lp_aucs))
    print(f"\n  LP result: {lp_mean:.3f} ± {lp_std:.3f}")
    print(f"  Compare:   TPA-CSAF frozen = 0.945 (no LeJEPA pretrain)")
    results["linear_probe"] = {"auroc_mean": round(lp_mean, 4), "auroc_std": round(lp_std, 4), "per_seed": [round(a, 4) for a in lp_aucs]}

    # ── Protocol 2: Fine-tune (all parameters) ──
    print("\n=== Protocol 2: Fine-tune (all parameters) ===")
    ft_aucs = []
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        auc = run_seed(encoder_state, seed, device, freeze_backbone=False, tag="ft")
        print(f"  seed {seed}: AUROC = {auc:.4f}")
        ft_aucs.append(auc)
    ft_mean, ft_std = float(np.mean(ft_aucs)), float(np.std(ft_aucs))
    print(f"\n  FT result: {ft_mean:.3f} ± {ft_std:.3f}")
    print(f"  Compare:   OPERA-CT fine-tune = 0.957")
    results["fine_tune"] = {"auroc_mean": round(ft_mean, 4), "auroc_std": round(ft_std, 4), "per_seed": [round(a, 4) for a in ft_aucs]}

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  OPERA-CT baseline (LP, frozen):   0.812 ± 0.011")
    print(f"  TPA-CSAF (LP, frozen, no LeJEPA): 0.945 ± 0.026")
    print(f"  HTS-AT LeJEPA LP  (ours):         {lp_mean:.3f} ± {lp_std:.3f}")
    print(f"  OPERA-CT fine-tune:               0.957 ± 0.024")
    print(f"  HTS-AT LeJEPA FT  (ours):         {ft_mean:.3f} ± {ft_std:.3f}")
    print("=" * 60)
    if ft_mean > 0.957:
        print(f"  >>> BEATS OPERA-CT fine-tune by +{ft_mean-0.957:.3f}!")
    elif ft_mean > 0.945:
        print(f"  >>> Beats TPA-CSAF by +{ft_mean-0.945:.3f}")

    summary = {
        "pretrain_ckpt": str(ckpt_path),
        "backbone": "HTS-AT Swin-Tiny (OPERA-CT init + LeJEPA pretrain)",
        "epochs_pretrain": ckpt.get("epoch", "?") if ckpt_path.exists() else "N/A",
        **results,
        "reference": {
            "opera_ct_baseline_lp": 0.812,
            "tpa_csaf_frozen_lp":   0.945,
            "opera_ct_finetune":    0.957,
        }
    }
    out_path = Path(OUT_DIR) / "downstream_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()

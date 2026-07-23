"""
SVD Voice Pathology — Full fine-tuning of HTS-AT + CSAF.

Unfreezes the entire encoder (backbone + CSAF) and trains end-to-end
on SVD with speaker-independent 10-fold CV.

Key design choices matching MVP protocol:
  - Multi-source: vowel /a/ (normal pitch) + phrase
  - Per-subject: both sources processed, fused at feature level
  - Speaker-independent stratified 10-fold CV
  - Early stopping on validation AUC
  - Differential learning rate: backbone 1e-5, CSAF/head 1e-4
  - Gradient clipping, weight decay, cosine schedule
  - Mixup augmentation
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from respvoice.htsat_encoder import build_htsat_encoder

SVD_CACHE = Path("data/mel_cache/svd_full")


# ── Dataset ──────────────────────────────────────────────────────────────────

class SVDPairDataset(Dataset):
    """Returns (vowel_mel, phrase_mel, label) per subject."""
    def __init__(self, subjects, indices=None):
        self.subjects = [subjects[i] for i in indices] if indices is not None else subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        vowel = torch.load(s["vowel"], map_location="cpu")   # (1, 64, T)
        phrase = torch.load(s["phrase"], map_location="cpu")
        return vowel, phrase, s["label"]


def collate_fn(batch):
    vowels = torch.stack([b[0] for b in batch])
    phrases = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return vowels, phrases, labels


# ── Model ────────────────────────────────────────────────────────────────────

class SVDFineTuneModel(nn.Module):
    """Full encoder + multi-source fusion + classifier."""

    def __init__(self, encoder, fusion="concat", feat_dim=768, hidden=256,
                 dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion

        if fusion == "concat":
            in_dim = feat_dim * 2
        elif fusion == "attention":
            in_dim = feat_dim
            self.source_attn = nn.Sequential(
                nn.Linear(feat_dim, 1),
            )
        else:
            in_dim = feat_dim

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def encode(self, mel):
        z = self.encoder(mel)      # (B, 64, 768)
        return z.mean(dim=1)       # (B, 768)

    def forward(self, vowel_mel, phrase_mel):
        zv = self.encode(vowel_mel)   # (B, 768)
        zp = self.encode(phrase_mel)  # (B, 768)

        if self.fusion == "concat":
            z = torch.cat([zv, zp], dim=1)          # (B, 1536)
        elif self.fusion == "attention":
            stacked = torch.stack([zv, zp], dim=1)   # (B, 2, 768)
            w = F.softmax(self.source_attn(stacked), dim=1)  # (B, 2, 1)
            z = (stacked * w).sum(dim=1)              # (B, 768)
        else:
            z = (zv + zp) / 2                         # (B, 768)

        return self.head(z)


# ── Mixup ────────────────────────────────────────────────────────────────────

def mixup_data(x1, x2, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x1.size(0)
    index = torch.randperm(batch_size, device=x1.device)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, weights):
    return lam * F.cross_entropy(pred, y_a, weight=weights) + \
           (1 - lam) * F.cross_entropy(pred, y_b, weight=weights)


# ── Training ─────────────────────────────────────────────────────────────────

def _rebuild_encoder(init_state, encoder_type, ckpt_path):
    """Rebuild a fresh encoder and load initial weights."""
    encoder = build_htsat_encoder(
        ckpt_path=None if encoder_type == "checkpoint" else None,
        use_csaf=True,
    )
    if encoder_type == "checkpoint":
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    else:
        encoder = build_htsat_encoder(use_csaf=True)
    encoder.load_state_dict(init_state, strict=False)
    return encoder


def train_one_fold(subjects, train_idx, val_idx, test_idx,
                   init_state, args, device, fold_idx):
    """Train and evaluate one fold with full fine-tuning."""
    random.seed(fold_idx)
    np.random.seed(fold_idx)
    torch.manual_seed(fold_idx)
    torch.cuda.manual_seed_all(fold_idx)

    train_ds = SVDPairDataset(subjects, train_idx)
    val_ds = SVDPairDataset(subjects, val_idx)
    test_ds = SVDPairDataset(subjects, test_idx)

    g = torch.Generator().manual_seed(fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2,
                              pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    # Fresh encoder per fold (rebuild from state_dict, no deepcopy)
    encoder = _rebuild_encoder(init_state, args.encoder, args.ckpt)
    for p in encoder.parameters():
        p.requires_grad = True
    model = SVDFineTuneModel(encoder, fusion=args.fusion).to(device)

    # Differential LR
    backbone_params = list(model.encoder.htsat.parameters())
    csaf_params = list(model.encoder.csaf.parameters()) if hasattr(model.encoder, 'csaf') else []
    head_params = list(model.head.parameters())
    if hasattr(model, 'source_attn'):
        head_params += list(model.source_attn.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": csaf_params, "lr": args.csaf_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class weights
    labels = [subjects[i]["label"] for i in train_idx]
    counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    weights = (counts.sum() / (counts.clamp_min(1) * 2)).to(device)

    best_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for vowel, phrase, label in train_loader:
            vowel = vowel.to(device, non_blocking=True)
            phrase = phrase.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if args.mixup > 0:
                vowel, phrase, y_a, y_b, lam = mixup_data(
                    vowel, phrase, label, args.mixup)
                logits = model(vowel, phrase)
                loss = mixup_criterion(logits, y_a, y_b, lam, weights)
            else:
                logits = model(vowel, phrase)
                loss = F.cross_entropy(logits, label, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for vowel, phrase, label in val_loader:
                vowel = vowel.to(device, non_blocking=True)
                phrase = phrase.to(device, non_blocking=True)
                logits = model(vowel, phrase)
                val_probs.append(F.softmax(logits, dim=1)[:, 1].cpu())
                val_labels.append(label)
        val_probs = torch.cat(val_probs).numpy()
        val_labels = torch.cat(val_labels).numpy()
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}, "
                  f"val_auc={val_auc:.4f}, best={best_auc:.4f}")

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for vowel, phrase, label in test_loader:
            vowel = vowel.to(device, non_blocking=True)
            phrase = phrase.to(device, non_blocking=True)
            logits = model(vowel, phrase)
            test_probs.append(F.softmax(logits, dim=1)[:, 1].cpu())
            test_labels.append(label)
    test_probs = torch.cat(test_probs).numpy()
    test_labels = torch.cat(test_labels).numpy()
    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        test_auc = 0.5

    del model, encoder
    torch.cuda.empty_cache()

    return test_auc, best_auc


# ── Data ─────────────────────────────────────────────────────────────────────

def collect_subjects(cache_root):
    raw = json.loads((cache_root / "metadata.json").read_text())
    by_subject = defaultdict(dict)
    labels = {}
    for item in raw["samples"]:
        subj = item["subject_id"]
        labels[subj] = int(item["label"])
        by_subject[subj][item["source"]] = cache_root / item["path"]
    subjects = []
    for subj, parts in by_subject.items():
        if "vowel" in parts and "phrase" in parts:
            subjects.append({
                "subj": subj, "label": labels[subj],
                "vowel": parts["vowel"], "phrase": parts["phrase"],
            })
    return subjects


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="opera_ct",
                        choices=["opera_ct", "checkpoint"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--fusion", default="concat",
                        choices=["concat", "attention", "mean"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--csaf-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cache", type=Path, default=SVD_CACHE)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}, Fusion: {args.fusion}")

    # Build base encoder and save its state_dict for per-fold reconstruction
    if args.encoder == "checkpoint":
        if not args.ckpt:
            raise ValueError("--ckpt required")
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v
                 for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        encoder = build_htsat_encoder(use_csaf=True)
        print("Using OPERA-CT weights")

    # Save initial state for per-fold reconstruction (avoids deepcopy)
    init_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    del encoder

    subjects = collect_subjects(args.cache)
    y = np.array([s["label"] for s in subjects])
    groups = np.array([s["subj"] for s in subjects])
    n_healthy = (y == 0).sum()
    n_path = (y == 1).sum()
    print(f"Subjects: {len(subjects)} (healthy={n_healthy}, pathological={n_path})")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
    fold_aucs = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(y, y, groups)):
        # Split train_val → train + val
        tv_y = y[train_val_idx]
        n_val = max(1, int(len(train_val_idx) * 0.15))
        perm = torch.randperm(len(train_val_idx),
                              generator=torch.Generator().manual_seed(fold_idx)).numpy()
        val_local = perm[:n_val]
        train_local = perm[n_val:]
        train_idx = train_val_idx[train_local]
        val_idx = train_val_idx[val_local]

        print(f"\nFold {fold_idx}: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")

        test_auc, best_val_auc = train_one_fold(
            subjects, train_idx, val_idx, test_idx,
            init_state, args, device, fold_idx,
        )
        fold_aucs.append(test_auc)
        print(f"  Fold {fold_idx}: test_AUC={test_auc:.4f} "
              f"(best_val={best_val_auc:.4f})")

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print(f"\n{'='*60}")
    print(f"  FULL FINE-TUNE: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  MVP HuBERT ref: 0.958")
    print(f"{'='*60}")

    output = {
        "encoder": args.encoder,
        "checkpoint": args.ckpt,
        "fusion": args.fusion,
        "protocol": "full fine-tuning, 10-fold speaker-independent CV",
        "backbone_lr": args.backbone_lr,
        "csaf_lr": args.csaf_lr,
        "head_lr": args.head_lr,
        "mixup": args.mixup,
        "n_subjects": len(subjects),
        "auc_mean": round(mean_auc, 4),
        "auc_std": round(std_auc, 4),
        "per_fold": [round(a, 4) for a in fold_aucs],
        "mvp_reference": 0.958,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

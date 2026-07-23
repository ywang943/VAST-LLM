"""
SVD Voice Pathology — Full evaluation with trainable classifier.

Replaces the logistic-regression probe with a proper neural classifier:
  - AttentionPool + 2-layer MLP head (trainable)
  - Optional CSAF fine-tuning per fold
  - Speaker-independent 10-fold stratified CV
  - Early stopping on validation AUC

Fusion strategies:
  - vowel_only / phrase_only   (single-source)
  - concat                     (feature concatenation)
  - attention_fusion            (learned attention over vowel+phrase)

Compare with MVP HuBERT IFF-TE = 95.8% AUC on SVD.
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
from torch.utils.data import DataLoader, TensorDataset

from respvoice.htsat_encoder import build_htsat_encoder

SVD_CACHE = Path("data/mel_cache/svd_full")


# ── Classifier heads ─────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """2-layer MLP with dropout."""
    def __init__(self, in_dim, hidden=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class AttentionFusionClassifier(nn.Module):
    """Learned attention over vowel and phrase features, then MLP."""
    def __init__(self, feat_dim, hidden=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Softmax(dim=1),
        )
        self.mlp = MLPClassifier(feat_dim, hidden, n_classes, dropout)

    def forward(self, vowel_feat, phrase_feat):
        stacked = torch.stack([vowel_feat, phrase_feat], dim=1)  # (B, 2, D)
        weights = self.attn(stacked)  # (B, 2, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, D)
        return self.mlp(fused)


# ── Data collection ──────────────────────────────────────────────────────────

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


@torch.no_grad()
def extract_features_batch(encoder, subjects, device, use_csaf=True):
    """Extract mean-pooled features for all subjects."""
    encoder.eval()
    X_vowel, X_phrase, y, groups = [], [], [], []
    for i, s in enumerate(subjects):
        mv = torch.load(s["vowel"], map_location="cpu").unsqueeze(0).to(device)
        mp = torch.load(s["phrase"], map_location="cpu").unsqueeze(0).to(device)
        if use_csaf:
            zv = encoder(mv).mean(dim=1).squeeze(0).cpu()
            zp = encoder(mp).mean(dim=1).squeeze(0).cpu()
        else:
            zv = encoder.htsat_forward_only(mv).mean(dim=1).squeeze(0).cpu()
            zp = encoder.htsat_forward_only(mp).mean(dim=1).squeeze(0).cpu()
        X_vowel.append(zv)
        X_phrase.append(zp)
        y.append(s["label"])
        groups.append(s["subj"])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(subjects)}")
    return (torch.stack(X_vowel), torch.stack(X_phrase),
            torch.tensor(y, dtype=torch.long), groups)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_fold(X_train, y_train, X_val, y_val, X_test, y_test,
               variant, feat_dim, epochs=100, lr=1e-3, patience=15,
               device="cpu", seed=42):
    """Train classifier for one fold, return test AUC."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if variant == "vowel_only":
        train_x = X_train[0]
        val_x = X_val[0]
        test_x = X_test[0]
        clf = MLPClassifier(feat_dim).to(device)
    elif variant == "phrase_only":
        train_x = X_train[1]
        val_x = X_val[1]
        test_x = X_test[1]
        clf = MLPClassifier(feat_dim).to(device)
    elif variant == "concat":
        train_x = torch.cat([X_train[0], X_train[1]], dim=1)
        val_x = torch.cat([X_val[0], X_val[1]], dim=1)
        test_x = torch.cat([X_test[0], X_test[1]], dim=1)
        clf = MLPClassifier(feat_dim * 2).to(device)
    elif variant == "mean":
        train_x = (X_train[0] + X_train[1]) / 2
        val_x = (X_val[0] + X_val[1]) / 2
        test_x = (X_test[0] + X_test[1]) / 2
        clf = MLPClassifier(feat_dim).to(device)
    elif variant == "attention_fusion":
        clf = AttentionFusionClassifier(feat_dim).to(device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Class weights
    counts = torch.bincount(y_train, minlength=2).float()
    weights = (counts.sum() / (counts.clamp_min(1) * 2)).to(device)

    optimizer = AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(epochs):
        clf.train()

        if variant == "attention_fusion":
            logits = clf(X_train[0].to(device), X_train[1].to(device))
        else:
            logits = clf(train_x.to(device))
        loss = F.cross_entropy(logits, y_train.to(device), weight=weights)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        clf.eval()
        with torch.no_grad():
            if variant == "attention_fusion":
                val_logits = clf(X_val[0].to(device), X_val[1].to(device))
            else:
                val_logits = clf(val_x.to(device))
            probs = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            try:
                val_auc = roc_auc_score(y_val.numpy(), probs)
            except ValueError:
                val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Test
    if best_state:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        if variant == "attention_fusion":
            test_logits = clf(X_test[0].to(device), X_test[1].to(device))
        else:
            test_logits = clf(test_x.to(device))
        probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        try:
            return float(roc_auc_score(y_test.numpy(), probs))
        except ValueError:
            return 0.5


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="opera_ct",
                        choices=["opera_ct", "checkpoint"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache", type=Path, default=SVD_CACHE)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build encoder
    print(f"Building encoder ({args.encoder})...")
    if args.encoder == "checkpoint":
        if not args.ckpt:
            raise ValueError("--ckpt required")
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v
                 for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)
    else:
        encoder = build_htsat_encoder(use_csaf=True)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device).eval()

    # Collect subjects
    subjects = collect_subjects(args.cache)
    print(f"Subjects: {len(subjects)} "
          f"(healthy={sum(1 for s in subjects if s['label']==0)}, "
          f"pathological={sum(1 for s in subjects if s['label']==1)})")

    # Extract features
    print("Extracting features (TPA-CSAF)...")
    X_vowel, X_phrase, y, groups = extract_features_batch(
        encoder, subjects, device, use_csaf=True)
    feat_dim = X_vowel.shape[1]
    print(f"Feature dim: {feat_dim}")
    groups_arr = np.array(groups)

    variants = ["vowel_only", "phrase_only", "concat", "mean", "attention_fusion"]
    results = {}

    for variant in variants:
        print(f"\n{'='*50}")
        print(f"  {variant}")
        print(f"{'='*50}")

        skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
        fold_aucs = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(
                skf.split(X_vowel, y, groups_arr)):
            # Split train_val into train and val (85/15)
            tv_labels = y[train_val_idx]
            n_val = max(1, int(len(train_val_idx) * 0.15))
            perm = torch.randperm(len(train_val_idx),
                                  generator=torch.Generator().manual_seed(fold_idx))
            val_local = perm[:n_val]
            train_local = perm[n_val:]
            train_idx = train_val_idx[train_local]
            val_idx = train_val_idx[val_local]

            X_train = (X_vowel[train_idx], X_phrase[train_idx])
            y_train = y[train_idx]
            X_val = (X_vowel[val_idx], X_phrase[val_idx])
            y_val = y[val_idx]
            X_test = (X_vowel[test_idx], X_phrase[test_idx])
            y_test = y[test_idx]

            auc = train_fold(X_train, y_train, X_val, y_val, X_test, y_test,
                             variant, feat_dim, epochs=args.epochs, lr=args.lr,
                             device=device, seed=fold_idx)
            fold_aucs.append(auc)
            print(f"  Fold {fold_idx}: AUC={auc:.4f}")

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        results[variant] = {
            "auc_mean": round(mean_auc, 4),
            "auc_std": round(std_auc, 4),
            "per_fold": [round(a, 4) for a in fold_aucs],
        }
        print(f"  {variant}: {mean_auc:.4f} ± {std_auc:.4f}")

    # Also extract stage4-only features for comparison
    print("\n\nExtracting Stage-4 only features...")

    @torch.no_grad()
    def extract_stage4(encoder, subjects, device):
        encoder.eval()
        Xv, Xp, ys, gs = [], [], [], []
        for i, s in enumerate(subjects):
            mv = torch.load(s["vowel"], map_location="cpu").unsqueeze(0).to(device)
            mp = torch.load(s["phrase"], map_location="cpu").unsqueeze(0).to(device)
            # Stage4 only
            x = encoder._preprocess(mv)
            x = encoder.htsat.patch_embed(x)
            if encoder.htsat.ape:
                x = x + encoder.htsat.absolute_pos_embed
            x = encoder.htsat.pos_drop(x)
            for layer in encoder.htsat.layers:
                x, _ = layer(x)
            zv = encoder.htsat.norm(x).mean(dim=1).squeeze(0).cpu()

            x = encoder._preprocess(mp)
            x = encoder.htsat.patch_embed(x)
            if encoder.htsat.ape:
                x = x + encoder.htsat.absolute_pos_embed
            x = encoder.htsat.pos_drop(x)
            for layer in encoder.htsat.layers:
                x, _ = layer(x)
            zp = encoder.htsat.norm(x).mean(dim=1).squeeze(0).cpu()

            Xv.append(zv); Xp.append(zp)
            ys.append(s["label"]); gs.append(s["subj"])
            if (i+1) % 200 == 0:
                print(f"  {i+1}/{len(subjects)}")
        return torch.stack(Xv), torch.stack(Xp), torch.tensor(ys, dtype=torch.long), gs

    X_vowel_s4, X_phrase_s4, y_s4, groups_s4 = extract_stage4(encoder, subjects, device)

    stage4_results = {}
    for variant in ["concat", "attention_fusion"]:
        skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
        fold_aucs = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(
                skf.split(X_vowel_s4, y_s4, np.array(groups_s4))):
            tv_labels = y_s4[train_val_idx]
            n_val = max(1, int(len(train_val_idx) * 0.15))
            perm = torch.randperm(len(train_val_idx),
                                  generator=torch.Generator().manual_seed(fold_idx))
            train_idx = train_val_idx[perm[n_val:]]
            val_idx = train_val_idx[perm[:n_val]]
            X_train = (X_vowel_s4[train_idx], X_phrase_s4[train_idx])
            X_val = (X_vowel_s4[val_idx], X_phrase_s4[val_idx])
            X_test = (X_vowel_s4[test_idx], X_phrase_s4[test_idx])
            auc = train_fold(X_train, y_s4[train_idx], X_val, y_s4[val_idx],
                             X_test, y_s4[test_idx], variant, 768,
                             epochs=args.epochs, lr=args.lr,
                             device=device, seed=fold_idx)
            fold_aucs.append(auc)
        stage4_results[f"stage4_{variant}"] = {
            "auc_mean": round(float(np.mean(fold_aucs)), 4),
            "auc_std": round(float(np.std(fold_aucs)), 4),
            "per_fold": [round(a, 4) for a in fold_aucs],
        }
        print(f"  stage4_{variant}: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    results.update(stage4_results)

    # Summary
    print(f"\n{'='*60}")
    print("  SVD FULL EVALUATION SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:25s}: {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    print(f"  {'MVP HuBERT IFF-TE ref':25s}: 0.958")

    output = {
        "encoder": args.encoder,
        "checkpoint": args.ckpt,
        "n_subjects": len(subjects),
        "protocol": "trainable MLP classifier, 10-fold speaker-independent CV",
        "results": results,
        "mvp_reference": 0.958,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

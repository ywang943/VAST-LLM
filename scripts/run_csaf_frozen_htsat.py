"""
CSAF Contribution Experiment: Frozen HTS-AT + Trainable CSAF + Linear Probe.

Design rationale:
  HTS-AT backbone is COLA-pretrained (OPERA-CT).
  Fine-tuning it with LeJEPA for only 5 epochs disrupts COLA features (LP drops 0.812→0.551).
  The clean experiment isolates CSAF's contribution:

  Condition A (baseline): Frozen HTS-AT Stage-4 only + linear head   → OPERA-CT repro 0.812
  Condition B (ours):     Frozen HTS-AT + CSAF (trainable) + linear  → does CSAF help?

  If B > A: CSAF multi-scale fusion extracts better features than Stage-4 alone.
  This is the direct contribution of the multi-scale design.

Protocol:
  - HTS-AT: fully frozen (initialized from OPERA-CT checkpoint)
  - CSAF: trainable (randomly initialized, learns to fuse 4 scales)
  - Linear head: trainable
  - ICBHI official split, 5 seeds
"""

import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.downstream import AttentionPool
from scripts.run_opera_icbhi_disease import official_split
from scripts.run_full_local import (
    class_weights_from_labels, evaluate_binary, labels_from_subset
)

LABEL_CACHE = "./data/mel_cache/opera_icbhi_disease"
D, BS = 768, 16
EPOCHS = 64
SEEDS = [0, 1, 2, 3, 4]


class CSAFLinearProbe(nn.Module):
    """
    Frozen HTS-AT + Trainable CSAF + Linear head.
    Only CSAF and head parameters are updated.
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        # Frozen backbone (OPERA-CT COLA pretrained, all 4 Swin stages)
        self.encoder = build_htsat_encoder(
            use_csaf=True,       # CSAF is ACTIVE
            freeze_backbone=True,  # HTS-AT frozen
        )
        # CSAF is trainable (already part of encoder.csaf)
        # Unfreeze CSAF and poolers
        for p in self.encoder.csaf.parameters():
            p.requires_grad = True
        for p in self.encoder.pool1.parameters():
            p.requires_grad = True
        for p in self.encoder.pool2.parameters():
            p.requires_grad = True

        # Linear probe head: pool + linear
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, n_classes)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen (HTS-AT): {n_frozen:,} params")
        print(f"  Trainable (CSAF + head): {n_train:,} params")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Run HTS-AT stages (frozen)
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
            x = self.encoder.htsat.norm(x)
            e4 = x

        # CSAF fusion (trainable) — learn multi-scale combination
        z = self.encoder.csaf([e1, e2, e3, e4])   # (B, 64, 768)

        # Pool and classify
        h = self.pool(z)                            # (B, 768)
        return self.head(h)                         # (B, n_classes)


def run_seed(seed: int, device, out_dir: str) -> float:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    label_ds = CachedMelDataset(
        root=LABEL_CACHE,
        meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2).to(device)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    model = CSAFLinearProbe(n_classes=2).to(device)
    # Only optimize CSAF and head parameters
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=3e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            mel    = batch["mel"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad()
            logits = model(mel)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            loss_sum += loss.item() * labels.numel()
            n += labels.numel()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                lbl = batch["label"].to(device)
                logits = model(mel)
                probs = F.softmax(logits, dim=1)[:, 1]
                val_probs.append(probs.cpu()); val_labels.append(lbl.cpu())
        from sklearn.metrics import roc_auc_score
        vp = torch.cat(val_probs).numpy(); vl = torch.cat(val_labels).numpy()
        try:
            val_auc = roc_auc_score(vl, vp)
        except Exception:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 16 == 0 or epoch == EPOCHS:
            print(f"    Ep{epoch}: loss={loss_sum/n:.4f}  val_auc={val_auc:.4f}  best={best_auc:.4f}")

    # Evaluate best on test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            mel = batch["mel"].to(device)
            lbl = batch["label"].to(device)
            logits = model(mel)
            probs = F.softmax(logits, dim=1)[:, 1]
            test_probs.append(probs.cpu()); test_labels.append(lbl.cpu())
    tp = torch.cat(test_probs).numpy(); tl = torch.cat(test_labels).numpy()
    try:
        test_auc = float(roc_auc_score(tl, tp))
    except Exception:
        test_auc = 0.5

    # Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "val_auc": best_auc,
                "test_auc": test_auc, "seed": seed},
               str(Path(out_dir) / f"csaf_lp_seed{seed}.pt"))
    return test_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "./checkpoints/csaf_frozen_htsat"
    print(f"=== CSAF + Frozen HTS-AT Linear Probe ===")
    print(f"  HTS-AT: FROZEN (OPERA-CT COLA pretrained)")
    print(f"  CSAF: TRAINABLE (learn multi-scale fusion)")
    print(f"  Protocol: ICBHI official split, {EPOCHS} epochs, {len(SEEDS)} seeds")
    print()

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        auc = run_seed(seed, device, out_dir)
        print(f"  seed {seed}: test AUROC = {auc:.4f}")
        results.append(auc)

    m_auc = float(np.mean(results))
    s_auc = float(np.std(results))
    print(f"\n{'='*55}")
    print(f"  CSAF + Frozen HTS-AT: {m_auc:.3f} +- {s_auc:.3f}")
    print(f"  per-seed: {[round(a,4) for a in results]}")
    print()
    print(f"  Compare:")
    print(f"    Stage-4 only (OPERA-CT repro, bs=32): 0.812 +- 0.011")
    print(f"    D128 LeJEPA (custom backbone):        0.770 +- 0.034")
    if m_auc > 0.812:
        print(f"  >>> CSAF improves over Stage-4 baseline by +{m_auc-0.812:.3f} !")
    print(f"{'='*55}")

    out = {
        "method": "Frozen HTS-AT + Trainable CSAF + Linear head",
        "htsat": "frozen (OPERA-CT COLA pretrained)",
        "csaf": "trainable (multi-scale fusion, 9.9M params)",
        "auroc_mean": round(m_auc, 4),
        "auroc_std": round(s_auc, 4),
        "per_seed": [round(a, 4) for a in results],
    }
    with open(f"{out_dir}/csaf_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_dir}/csaf_results.json")


if __name__ == "__main__":
    main()

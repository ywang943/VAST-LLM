"""
CSAF Ablation: Isolate contributions of multi-scale vs cross-scale attention.

Answers reviewer questions:
  1. Does any single intermediate stage beat Stage-4?
  2. Does parameter-free concat fusion match CSAF?
  3. Is the gain from the attention mechanism specifically?

All conditions: frozen HTS-AT (OPERA-CT COLA), same protocol (5 seeds, 64 epochs).

Conditions:
  A: Stage-4 only + linear head         (0.812, known baseline)
  B: Stage-1 only + linear head         (fine-grained only)
  C: Stage-3 only + linear head         (medium scale)
  D: Concat of 4 stages + linear head  (multi-scale, zero fusion params)
  E: CSAF + linear head                 (0.920, our method)
"""

import json, random, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.downstream import AttentionPool
from respvoice.csa_fusion import (
    CrossScaleAttentionFusion, PoolToResolution,
)
from scripts.run_opera_icbhi_disease import official_split
from scripts.run_full_local import class_weights_from_labels, labels_from_subset

LABEL_CACHE = "./data/mel_cache/opera_icbhi_disease"
D, BS, EPOCHS = 768, 16, 64
SEEDS = [0, 1, 2, 3, 4]


# ── Feature extractors ────────────────────────────────────────────────────────

def extract_htsat_stages(htsat, mel, pool1, pool2):
    """Extract features from all 4 frozen HTS-AT stages."""
    with torch.no_grad():
        # Preprocessing
        x = mel.transpose(2, 3)              # (B, 1, T, F)
        x = x.transpose(1, 3)
        x = htsat.bn0(x)
        x = x.transpose(1, 3)
        # reshape_wav2img
        B, C, T, F = x.shape
        target_T = int(htsat.spec_size * htsat.freq_ratio)
        if T < target_T:
            x = x.repeat(1, 1, (target_T // T) + 1, 1)
        x = x[:, :, :target_T, :]
        x = htsat.reshape_wav2img(x)

        x = htsat.patch_embed(x)
        if htsat.ape:
            x = x + htsat.absolute_pos_embed
        x = htsat.pos_drop(x)

        x, _ = htsat.layers[0](x)  # (B, 1024, 192)
        e1 = pool1(x)               # (B, 64, 192)
        x, _ = htsat.layers[1](x)  # (B, 256, 384)
        e2 = pool2(x)               # (B, 64, 384)
        x, _ = htsat.layers[2](x)  # (B, 64, 768)
        e3 = x
        x, _ = htsat.layers[3](x)  # (B, 64, 768)
        e4 = htsat.norm(x)
    return e1, e2, e3, e4


# ── Probe models ──────────────────────────────────────────────────────────────

class SingleStageProbe(nn.Module):
    """Linear probe on a single HTS-AT stage."""
    def __init__(self, stage_dim: int, n_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(stage_dim, D) if stage_dim != D else nn.Identity()
        self.norm = nn.LayerNorm(D)
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, n_classes)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.proj(e))  # (B, 64, D)
        h = self.pool(z)             # (B, D)
        return self.head(h)


class ConcatProbe(nn.Module):
    """Concatenate all 4 stages along feature dim, mean-pool, linear head only."""
    CONCAT_DIM = 192 + 384 + 768 + 768  # 2112

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.head = nn.Linear(self.CONCAT_DIM, n_classes)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"    ConcatProbe params: {n_params:,}")

    def forward(self, stages: list) -> torch.Tensor:
        z = torch.cat(stages, dim=-1)   # (B, 64, 2112)
        h = z.mean(dim=1)               # (B, 2112)
        return self.head(h)


class CSAFProbe(nn.Module):
    """CSAF (cross-scale attention) probe — same as run_csaf_frozen_htsat.py."""
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.csaf = CrossScaleAttentionFusion(
            D=D, n_scales=4, n_heads=8, depth=2,
            scale_dims=(192, 384, 768, 768)
        )
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, n_classes)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"    CSAFProbe params: {n_params:,}")

    def forward(self, stages: list) -> torch.Tensor:
        z = self.csaf(stages)   # (B, 64, D)
        h = self.pool(z)
        return self.head(h)


# ── Training ──────────────────────────────────────────────────────────────────

def run_probe(probe, stage_selector, htsat, pool1, pool2, train_ds, val_ds, test_ds,
              weights, device, seed):
    """Train and evaluate one probe variant."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    probe = probe.to(device)
    params = list(probe.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        probe.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            mel    = batch["mel"].to(device)
            labels = batch["label"].to(device)
            stages = extract_htsat_stages(htsat, mel, pool1, pool2)
            feats  = stage_selector(stages)
            opt.zero_grad()
            logits = probe(feats) if isinstance(feats, list) else probe(feats)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            loss_sum += loss.item() * labels.numel()
            n += labels.numel()

        # Validation
        probe.eval()
        vprobs, vlabels = [], []
        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                lbl = batch["label"].to(device)
                stages = extract_htsat_stages(htsat, mel, pool1, pool2)
                feats = stage_selector(stages)
                logits = probe(feats)
                vprobs.append(F.softmax(logits, dim=1)[:, 1].cpu())
                vlabels.append(lbl.cpu())
        vp = torch.cat(vprobs).numpy(); vl = torch.cat(vlabels).numpy()
        try:
            val_auc = roc_auc_score(vl, vp)
        except Exception:
            val_auc = 0.5
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    # Test evaluation
    if best_state:
        probe.load_state_dict(best_state)
    probe.eval()
    tprobs, tlabels = [], []
    with torch.no_grad():
        for batch in test_loader:
            mel = batch["mel"].to(device)
            lbl = batch["label"].to(device)
            stages = extract_htsat_stages(htsat, mel, pool1, pool2)
            feats = stage_selector(stages)
            logits = probe(feats)
            tprobs.append(F.softmax(logits, dim=1)[:, 1].cpu())
            tlabels.append(lbl.cpu())
    tp = torch.cat(tprobs).numpy(); tl = torch.cat(tlabels).numpy()
    try:
        return float(roc_auc_score(tl, tp))
    except Exception:
        return 0.5


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen HTS-AT
    print("\nLoading frozen HTS-AT (OPERA-CT)...")
    enc = build_htsat_encoder(use_csaf=False)  # just need the backbone
    htsat = enc.htsat.to(device)
    pool1 = enc.pool1.to(device)  # PoolToResolution(8,8)
    pool2 = enc.pool2.to(device)
    for p in htsat.parameters():
        p.requires_grad = False
    n_frozen = sum(p.numel() for p in htsat.parameters())
    print(f"  HTS-AT frozen: {n_frozen:,} params")

    # Dataset
    label_ds = CachedMelDataset(
        root=LABEL_CACHE,
        meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
        include_labels=True)
    train_ds, val_ds, test_ds = official_split(label_ds)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2).to(device)

    # Ablation conditions: (selector_fn, probe_factory, description)
    conditions = {
        "B_stage1_only":   (lambda s: s[0],  lambda: SingleStageProbe(192), "Stage-1 only (~1s, jitter/shimmer)"),
        "C_stage3_only":   (lambda s: s[2],  lambda: SingleStageProbe(768), "Stage-3 only (~4s, F0 scale)"),
        "D_concat":        (lambda s: s,     lambda: ConcatProbe(),         "Concat of 4 stages, linear head only"),
        "E_csaf":          (lambda s: s,     lambda: CSAFProbe(),           "CSAF cross-scale attention (ours)"),
    }

    results = {
        "A_stage4_only": {
            "description": "Stage-4 only + linear head (OPERA-CT repro baseline)",
            "auroc_mean": 0.812, "auroc_std": 0.011,
            "per_seed": [], "note": "from OPERA-CT reproduction run"
        }
    }

    for cond_key, (selector, probe_factory, desc) in conditions.items():
        print(f"\n{'='*55}")
        print(f"  Condition {cond_key}: {desc}")
        # Print param count once
        probe_sample = probe_factory()
        print(f"    Trainable params: {sum(p.numel() for p in probe_sample.parameters()):,}")
        del probe_sample

        seed_aurocs = []
        for seed in SEEDS:
            probe = probe_factory()  # fresh probe per seed
            auc = run_probe(probe, selector, htsat, pool1, pool2,
                            train_ds, val_ds, test_ds, weights, device, seed)
            print(f"    seed {seed}: AUROC={auc:.4f}")
            seed_aurocs.append(auc)

        m = float(np.mean(seed_aurocs)); s = float(np.std(seed_aurocs))
        print(f"  {desc}: {m:.3f} +- {s:.3f}")
        results[cond_key] = {
            "description": desc,
            "auroc_mean": round(m, 4),
            "auroc_std": round(s, 4),
            "per_seed": [round(a, 4) for a in seed_aurocs],
        }

    # Summary
    print(f"\n{'='*60}")
    print("  CSAF ABLATION SUMMARY")
    print(f"{'='*60}")
    order = ["A_stage4_only", "B_stage1_only", "C_stage3_only",
             "D_concat", "E_csaf"]
    labels_map = {
        "A_stage4_only":   "A: Stage-4 only (OPERA baseline)",
        "B_stage1_only":   "B: Stage-1 only (~1s, fine-grained)",
        "C_stage3_only":   "C: Stage-3 only (~4s, F0 scale)",
        "D_concat":        "D: Concat 4 stages, linear head only",
        "E_csaf":          "E: CSAF cross-scale attention (ours)",
    }
    for k in order:
        if k in results:
            r = results[k]
            print(f"  {labels_map[k]:<45}  {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}")

    out = Path("checkpoints/csaf_ablation/csaf_ablation_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

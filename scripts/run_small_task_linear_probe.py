"""
Pure linear probe for SMALL downstream tasks (KAUH, COPD severity).

Problem: TPA_CSAFProbe (10M trainable CSAF params) overfits tiny datasets
(KAUH train=162, COPD train=312), underperforming OPERA-CT (0.722 / 0.625).

Fix: OPERA-style pure linear probe — FREEZE the entire encoder (HTS-AT Stage-4
output), train ONLY a linear head. This matches OPERA's protocol exactly and
should be competitive on small data.

Two encoder feature sources (selectable):
  - stage4:  frozen HTS-AT Stage-4 (like OPERA-CT, 768-d)  [default]
  - csaf_frozen: frozen HTS-AT + frozen randomly-init CSAF (no train) — not used

Linear probe head:
  z (B,64,768) -> mean-pool -> Linear(768, n_classes)   (only ~1.5K-4K params)

Protocol: OPERA official-style, 5 seeds, AUROC (binary or macro-OvR).

Usage:
  python scripts/run_small_task_linear_probe.py --task kauh
  python scripts/run_small_task_linear_probe.py --task copd
  python scripts/run_small_task_linear_probe.py --task both
"""

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder

D, BS, EPOCHS = 768, 32, 64
SEEDS = [0, 1, 2, 3, 4]

TASKS = {
    "kauh": ("data/mel_cache/opera_kauh", 2, "KAUH Obstructive", 0.722),
    "copd": ("data/mel_cache/opera_copd", 5, "COPD Severity 5-class", 0.625),
}


@torch.no_grad()
def stage4_feature(htsat, mel, device):
    """Frozen HTS-AT Stage-4 output, mean-pooled to (B, 768) — OPERA-style."""
    x = mel.transpose(2, 3).transpose(1, 3)
    x = htsat.bn0(x).transpose(1, 3)
    B, C, T, Fr = x.shape
    target_T = int(htsat.spec_size * htsat.freq_ratio)
    if T < target_T:
        x = x.repeat(1, 1, (target_T // T) + 1, 1)
    x = x[:, :, :target_T, :]
    x = htsat.reshape_wav2img(x)
    x = htsat.patch_embed(x)
    if htsat.ape:
        x = x + htsat.absolute_pos_embed
    x = htsat.pos_drop(x)
    x, _ = htsat.layers[0](x)
    x, _ = htsat.layers[1](x)
    x, _ = htsat.layers[2](x)
    x, _ = htsat.layers[3](x)
    x = htsat.norm(x)            # (B, 64, 768)
    return x.mean(dim=1)         # (B, 768) — global mean pool


class LinearHead(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.head = nn.Linear(D, n_classes)
    def forward(self, feat):
        return self.head(feat)


def split_dataset(ds, seed=1337):
    labels = [int(s["label"]) for s in ds.samples]
    idx = list(range(len(ds)))
    # honor existing split field if present, else stratified
    has_split = all("split" in s for s in ds.samples)
    if has_split:
        tr = [i for i in idx if ds.samples[i]["split"] == "train"]
        va = [i for i in idx if ds.samples[i]["split"] == "val"]
        te = [i for i in idx if ds.samples[i]["split"] == "test"]
        if len(te) >= 5 and len(tr) >= 10:
            return Subset(ds, tr), Subset(ds, va or tr[:max(1,len(tr)//5)]), Subset(ds, te)
    trv, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    trl = [labels[i] for i in trv]
    tr, va = train_test_split(trv, test_size=0.2, random_state=seed, stratify=trl)
    return Subset(ds, tr), Subset(ds, va), Subset(ds, te)


def class_weights(subset, n_classes, device):
    labels = [int(subset.dataset.samples[i]["label"]) for i in subset.indices]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.sum() * n_classes
    return torch.tensor(w, device=device)


def auc_score(y, p, n_classes):
    try:
        if n_classes == 2:
            return roc_auc_score(y, p[:, 1])
        return roc_auc_score(y, p, multi_class="ovr", average="macro")
    except Exception:
        return 0.5


def run_task(task, device):
    cache, n_classes, name, opera_ref = TASKS[task]
    if not (Path(cache) / "metadata.json").exists():
        print(f"  SKIP {name}: cache {cache} not found"); return None
    ds = CachedMelDataset(root=cache, meta_file=str(Path(cache)/"metadata.json"), include_labels=True)
    # normalize labels to int
    uniq = sorted(set(str(s["label"]) for s in ds.samples))
    lmap = {l: i for i, l in enumerate(uniq)}
    for s in ds.samples:
        s["label"] = lmap.get(str(s["label"]), 0)
    print(f"\n=== {name} (pure linear probe, frozen encoder) ===")
    print(f"  {len(ds)} samples, {n_classes} classes")

    enc = build_htsat_encoder(use_csaf=False)
    htsat = enc.htsat.to(device).eval()
    for p in htsat.parameters(): p.requires_grad = False

    aucs = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        tr, va, te = split_dataset(ds, seed=1337)
        if len(te) < 5: print("  test too small, skip"); return None
        w = class_weights(tr, n_classes, device)
        g = torch.Generator().manual_seed(seed)
        trL = DataLoader(tr, batch_size=BS, shuffle=True, num_workers=0, generator=g)
        vaL = DataLoader(va, batch_size=BS, shuffle=False, num_workers=0)
        teL = DataLoader(te, batch_size=BS, shuffle=False, num_workers=0)

        head = LinearHead(n_classes).to(device)
        opt = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(trL))

        best, best_state = -1, None
        for ep in range(EPOCHS):
            head.train()
            for b in trL:
                feat = stage4_feature(htsat, b["mel"].to(device), device)
                opt.zero_grad()
                F.cross_entropy(head(feat), b["label"].to(device), weight=w).backward()
                opt.step(); sched.step()
            head.eval()
            vp, vl = [], []
            with torch.no_grad():
                for b in vaL:
                    feat = stage4_feature(htsat, b["mel"].to(device), device)
                    vp.append(F.softmax(head(feat), 1).cpu()); vl.append(b["label"])
            va_auc = auc_score(torch.cat(vl).numpy(), torch.cat(vp).numpy(), n_classes)
            if va_auc > best: best = va_auc; best_state = {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state); head.eval()
        tp, tl = [], []
        with torch.no_grad():
            for b in teL:
                feat = stage4_feature(htsat, b["mel"].to(device), device)
                tp.append(F.softmax(head(feat), 1).cpu()); tl.append(b["label"])
        ta = auc_score(torch.cat(tl).numpy(), torch.cat(tp).numpy(), n_classes)
        print(f"    seed {seed}: test AUROC={ta:.4f}")
        aucs.append(float(ta))

    m, s = float(np.mean(aucs)), float(np.std(aucs))
    print(f"  {name}: {m:.3f} ± {s:.3f}   (OPERA-CT ref: {opera_ref})")
    if m > opera_ref: print(f"  >>> BEATS OPERA-CT by +{m-opera_ref:.3f}")
    out = {"task": name, "protocol": "pure linear probe (frozen HTS-AT Stage-4)",
           "auroc_mean": round(m, 4), "auroc_std": round(s, 4),
           "per_seed": [round(a, 4) for a in aucs], "opera_ct_ref": opera_ref}
    od = Path(f"checkpoints/small_task_lp/{task}")
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["kauh", "copd", "both"], default="both")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ["kauh", "copd"] if args.task == "both" else [args.task]
    results = {}
    for t in tasks:
        r = run_task(t, device)
        if r: results[t] = r
    print("\n" + "=" * 55)
    print("  SMALL-TASK LINEAR PROBE SUMMARY")
    print("=" * 55)
    for t, r in results.items():
        ref = r["opera_ct_ref"]
        flag = "✓ beats OPERA" if r["auroc_mean"] > ref else "below OPERA"
        print(f"  {r['task']:25s} {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}  (OPERA-CT {ref})  {flag}")


if __name__ == "__main__":
    main()

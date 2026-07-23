"""
Run TPA-CSAF evaluation on all available downstream tasks.
Skips tasks whose result JSON already exists.

Usage:
    python scripts/run_all_downstream.py
    python scripts/run_all_downstream.py --seeds 0 1 2  # quick 3-seed run
"""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.csa_fusion import CrossScaleAttentionFusion
from respvoice.downstream import AttentionPool

SEEDS = [0, 1, 2, 3, 4]
D, BS, EPOCHS = 768, 32, 64

# All downstream tasks (cache_dir, result_dir, n_classes, description)
ALL_TASKS = [
    # Already done / running — use their existing result dirs
    ("data/mel_cache/opera_icbhi_disease",   "checkpoints/csaf_frozen_htsat",         2,  "ICBHI Healthy-vs-COPD"),
    ("data/mel_cache/opera_copd",            "checkpoints/downstream_copd_d128_cont_ft", 5, "COPD Severity 5-class"),
    ("data/mel_cache/opera_kauh",            "checkpoints/downstream_kauh_d128_cont_ft", 2, "KAUH Obstructive"),
    ("data/mel_cache/coughvid_covid",        "checkpoints/coughvid_tasks",             2,  "CoughVID COVID Detection"),
    ("data/mel_cache/coughvid_sex",          "checkpoints/coughvid_tasks",             2,  "CoughVID Sex Detection"),
    # New Coswara tasks
    ("data/mel_cache/coswara_covid_all",     "checkpoints/tasks/coswara_covid_all",    2,  "Coswara COVID (all modalities)"),
    ("data/mel_cache/coswara_covid_breathing","checkpoints/tasks/coswara_covid_breath",2,  "Coswara COVID (breathing)"),
    ("data/mel_cache/coswara_covid_cough",   "checkpoints/tasks/coswara_covid_cough",  2,  "Coswara COVID (cough)"),
    ("data/mel_cache/coswara_covid_voice",   "checkpoints/tasks/coswara_covid_voice",  2,  "Coswara COVID (voice/vowel)"),
    ("data/mel_cache/coswara_modality",      "checkpoints/tasks/coswara_modality",     4,  "Coswara Modality 4-class"),
    ("data/mel_cache/coswara_severity",      "checkpoints/tasks/coswara_severity",     3,  "Coswara COVID Severity 3-class"),
    ("data/mel_cache/icbhi_hf_4class",       "checkpoints/tasks/icbhi_4class",         7,  "ICBHI Respiratory 7-class"),
]


def split_dataset(ds):
    train, val, test = [], [], []
    for i, s in enumerate(ds.samples):
        sp = s.get("split", "train")
        if sp == "test":   test.append(i)
        elif sp == "val":  val.append(i)
        else:              train.append(i)
    if not test:  # random split if no test
        random.shuffle(train)
        n = len(train); n_t = max(1, int(n*0.2)); n_v = max(1, int(n*0.1))
        test = train[:n_t]; val = train[n_t:n_t+n_v]; train = train[n_t+n_v:]
    if not val:   # carve val from train if missing
        random.shuffle(train)
        n_v = max(1, int(len(train)*0.15))
        val = train[:n_v]; train = train[n_v:]
    return Subset(ds, train), Subset(ds, val), Subset(ds, test)


def extract_stages(htsat, pool1, pool2, mel, device):
    with torch.no_grad():
        x = mel.transpose(2,3).transpose(1,3)
        x = htsat.bn0(x); x = x.transpose(1,3)
        B,C,T,F2 = x.shape; target_T = int(htsat.spec_size * htsat.freq_ratio)
        if T < target_T: x = x.repeat(1,1,(target_T//T)+1,1)
        x = x[:,:,:target_T,:]
        x = htsat.reshape_wav2img(x)
        x = htsat.patch_embed(x)
        if htsat.ape: x = x + htsat.absolute_pos_embed
        x = htsat.pos_drop(x)
        x, _ = htsat.layers[0](x); e1 = pool1(x)
        x, _ = htsat.layers[1](x); e2 = pool2(x)
        x, _ = htsat.layers[2](x); e3 = x
        x, _ = htsat.layers[3](x); e4 = htsat.norm(x)
    return e1, e2, e3, e4


class TPA_CSAFProbe(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.csaf = CrossScaleAttentionFusion(D=D, n_scales=4, n_heads=8, depth=2,
                                               scale_dims=(192,384,768,768))
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, n_classes)

    def forward(self, stages):
        return self.head(self.pool(self.csaf(stages)))


def class_weights_from_subset(subset, n_classes, device):
    labels = [subset.dataset.samples[i]["label"] for i in subset.indices]
    try:
        labels_int = [int(l) for l in labels]
    except Exception:
        unique = sorted(set(labels))
        lmap = {l: i for i, l in enumerate(unique)}
        labels_int = [lmap[l] for l in labels]
    counts = torch.bincount(torch.tensor(labels_int), minlength=n_classes).float()
    w = counts.sum() / (counts.clamp_min(1) * n_classes)
    return w.to(device)


def run_task(cache_dir, result_dir, n_classes, task_name, device, seeds):
    # Check multiple possible result filenames
    result_path = None
    for fname in ["results.json", "csaf_results.json", "coughvid_covid_results.json",
                  "coughvid_sex_results.json", "summary.json"]:
        p = Path(result_dir) / fname
        if p.exists():
            result_path = p
            break
    if result_path:
        try:
            r = json.loads(result_path.read_text())
            auc = r.get("auroc_mean") or r.get("best_test_auroc_mean", "?")
            if auc and float(auc) > 0.3:
                print(f"  SKIP {task_name}: already done (AUROC={auc})")
                return r
        except Exception: pass
    result_path = Path(result_dir) / "results.json"

    if not Path(cache_dir).exists() or not (Path(cache_dir) / "metadata.json").exists():
        print(f"  SKIP {task_name}: cache not found")
        return None

    ds = CachedMelDataset(root=cache_dir, meta_file=str(Path(cache_dir)/"metadata.json"),
                           include_labels=True)
    if len(ds) < 20:
        print(f"  SKIP {task_name}: too few samples ({len(ds)})")
        return None

    # Fix string labels to int
    for s in ds.samples:
        try: s["label"] = int(s["label"])
        except Exception:
            unique = sorted(set(int(x["label"]) if isinstance(x["label"], (int,float))
                               else x["label"] for x in ds.samples))
            lmap = {l: i for i, l in enumerate(unique)}
            for ss in ds.samples:
                ss["label"] = lmap.get(ss["label"], 0)
            break

    train_ds, val_ds, test_ds = split_dataset(ds)
    print(f"  {task_name}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} n_cls={n_classes}")
    if len(test_ds) < 5:
        print(f"  SKIP: test set too small")
        return None

    enc = build_htsat_encoder(use_csaf=False)  # backbone only
    htsat = enc.htsat.to(device)
    pool1, pool2 = enc.pool1.to(device), enc.pool2.to(device)
    for p in htsat.parameters(): p.requires_grad = False

    weights = class_weights_from_subset(train_ds, n_classes, device)
    val_loader  = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    seed_aurocs = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        probe = TPA_CSAFProbe(n_classes).to(device)
        params = list(probe.parameters())
        opt = AdamW(params, lr=3e-4, weight_decay=1e-2)
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                                   num_workers=0, generator=g)
        sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

        best_auc, best_state = -1.0, None
        for epoch in range(1, EPOCHS + 1):
            probe.train()
            for batch in train_loader:
                mel = batch["mel"].to(device); lbl = batch["label"].to(device)
                stages = extract_stages(htsat, pool1, pool2, mel, device)
                opt.zero_grad()
                F.cross_entropy(probe(list(stages)), lbl, weight=weights).backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step()

            probe.eval()
            vp, vl = [], []
            with torch.no_grad():
                for batch in val_loader:
                    mel = batch["mel"].to(device); lbl = batch["label"].to(device)
                    stages = extract_stages(htsat, pool1, pool2, mel, device)
                    probs = F.softmax(probe(list(stages)), dim=1)
                    vp.append(probs.cpu()); vl.append(lbl.cpu())
            vp_np = torch.cat(vp).numpy(); vl_np = torch.cat(vl).numpy()
            try:
                va = roc_auc_score(vl_np, vp_np[:,1] if n_classes==2 else vp_np,
                                   multi_class="ovr", average="macro") if n_classes > 2 else \
                     roc_auc_score(vl_np, vp_np[:,1])
            except Exception: va = 0.5
            if va > best_auc: best_auc = va; best_state = {k:v.clone() for k,v in probe.state_dict().items()}

        if best_state: probe.load_state_dict(best_state)
        probe.eval()
        tp, tl = [], []
        with torch.no_grad():
            for batch in test_loader:
                mel = batch["mel"].to(device); lbl = batch["label"].to(device)
                stages = extract_stages(htsat, pool1, pool2, mel, device)
                probs = F.softmax(probe(list(stages)), dim=1)
                tp.append(probs.cpu()); tl.append(lbl.cpu())
        tp_np = torch.cat(tp).numpy(); tl_np = torch.cat(tl).numpy()
        try:
            ta = roc_auc_score(tl_np, tp_np[:,1] if n_classes==2 else tp_np,
                               multi_class="ovr", average="macro") if n_classes > 2 else \
                 roc_auc_score(tl_np, tp_np[:,1])
        except Exception: ta = 0.5
        print(f"    seed {seed}: val={best_auc:.4f} test={ta:.4f}")
        seed_aurocs.append(float(ta))

    m, s = float(np.mean(seed_aurocs)), float(np.std(seed_aurocs))
    print(f"  {task_name}: AUROC {m:.3f} +/- {s:.3f}")

    result = {"task": task_name, "auroc_mean": round(m,4), "auroc_std": round(s,4),
              "per_seed": [round(a,4) for a in seed_aurocs], "n_seeds": len(seeds)}
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--tasks", nargs="+", default=None, help="subset of task indices")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}")
    print()

    task_list = ALL_TASKS
    if args.tasks:
        task_list = [ALL_TASKS[int(i)] for i in args.tasks]

    all_results = {}
    for cache_dir, result_dir, n_classes, task_name in task_list:
        print(f"\n{'='*55}")
        print(f"  Task: {task_name}")
        r = run_task(cache_dir, result_dir, n_classes, task_name, device, args.seeds)
        if r: all_results[task_name] = r

    print(f"\n{'='*55}")
    print("  COMPLETE DOWNSTREAM RESULTS SUMMARY")
    print(f"{'='*55}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1].get("auroc_mean", 0)):
        print(f"  {name:<45}  {r['auroc_mean']:.3f} +/- {r['auroc_std']:.3f}")

    out = Path("checkpoints/all_downstream_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

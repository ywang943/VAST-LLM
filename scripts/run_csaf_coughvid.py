"""
Evaluate CSAF + Frozen HTS-AT on CoughVID tasks.
  - COVID detection: healthy vs COVID-19
  - Sex detection: female vs male

Protocol: same as run_csaf_frozen_htsat.py (5 seeds, 64 epochs)
"""
import json, random, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.downstream import AttentionPool

SEEDS = [0, 1, 2, 3, 4]
D, BS, EPOCHS = 768, 32, 64


def split_by_meta(ds):
    train, val, test = [], [], []
    for i, s in enumerate(ds.samples):
        sp = s.get("split", "train")
        if sp == "test":   test.append(i)
        elif sp == "val":  val.append(i)
        else:              train.append(i)
    return Subset(ds, train), Subset(ds, val), Subset(ds, test)


def extract_htsat_stages(htsat, pool1, pool2, mel, device):
    with torch.no_grad():
        x = mel.transpose(2, 3).transpose(1, 3)
        x = htsat.bn0(x)
        x = x.transpose(1, 3)
        B, C, T, F2 = x.shape
        target_T = int(htsat.spec_size * htsat.freq_ratio)
        if T < target_T:
            x = x.repeat(1, 1, (target_T // T) + 1, 1)
        x = x[:, :, :target_T, :]
        x = htsat.reshape_wav2img(x)
        x = htsat.patch_embed(x)
        if htsat.ape:
            x = x + htsat.absolute_pos_embed
        x = htsat.pos_drop(x)
        x, _ = htsat.layers[0](x); e1 = pool1(x)
        x, _ = htsat.layers[1](x); e2 = pool2(x)
        x, _ = htsat.layers[2](x); e3 = x
        x, _ = htsat.layers[3](x); e4 = htsat.norm(x)
    return e1, e2, e3, e4


class CSAFProbe(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        from respvoice.csa_fusion import CrossScaleAttentionFusion
        self.csaf = CrossScaleAttentionFusion(D=D, n_scales=4, n_heads=8, depth=2,
                                               scale_dims=(192, 384, 768, 768))
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, n_classes)

    def forward(self, stages):
        z = self.csaf(stages)
        return self.head(self.pool(z))


def class_weights(ds_subset, n_classes, device):
    labels = [ds_subset.dataset.samples[i]["label"] for i in ds_subset.indices]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=n_classes).float()
    w = counts.sum() / (counts.clamp_min(1) * n_classes)
    return w.to(device)


def run_task(cache_dir, task_name, n_classes, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {task_name} (n_classes={n_classes}) ===")

    enc = build_htsat_encoder(use_csaf=False)
    htsat = enc.htsat.to(device)
    pool1, pool2 = enc.pool1.to(device), enc.pool2.to(device)
    for p in htsat.parameters(): p.requires_grad = False

    ds = CachedMelDataset(root=cache_dir, meta_file=str(Path(cache_dir)/"metadata.json"),
                           include_labels=True)
    train_ds, val_ds, test_ds = split_by_meta(ds)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    weights = class_weights(train_ds, n_classes, device)
    val_loader  = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    seed_results = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        probe = CSAFProbe(n_classes).to(device)
        params = list(probe.parameters())
        opt = AdamW(params, lr=3e-4, weight_decay=1e-2)
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, generator=g)
        sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

        best_auc, best_state = -1.0, None
        for epoch in range(1, EPOCHS + 1):
            probe.train()
            for batch in train_loader:
                mel, labels = batch["mel"].to(device), batch["label"].to(device)
                stages = extract_htsat_stages(htsat, pool1, pool2, mel, device)
                opt.zero_grad()
                logits = probe(list(stages))
                F.cross_entropy(logits, labels, weight=weights).backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step()

            probe.eval()
            vp, vl = [], []
            with torch.no_grad():
                for batch in val_loader:
                    mel = batch["mel"].to(device); lbl = batch["label"].to(device)
                    stages = extract_htsat_stages(htsat, pool1, pool2, mel, device)
                    probs = F.softmax(probe(list(stages)), dim=1)
                    if n_classes == 2:
                        vp.append(probs[:, 1].cpu()); vl.append(lbl.cpu())
                    else:
                        vp.append(probs.cpu()); vl.append(lbl.cpu())
            vp_np = torch.cat(vp).numpy(); vl_np = torch.cat(vl).numpy()
            try:
                val_auc = roc_auc_score(vl_np, vp_np) if n_classes == 2 else \
                          roc_auc_score(vl_np, vp_np, multi_class='ovr', average='macro')
            except Exception: val_auc = 0.5
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if best_state: probe.load_state_dict(best_state)
        probe.eval()
        tp, tl = [], []
        with torch.no_grad():
            for batch in test_loader:
                mel = batch["mel"].to(device); lbl = batch["label"].to(device)
                stages = extract_htsat_stages(htsat, pool1, pool2, mel, device)
                probs = F.softmax(probe(list(stages)), dim=1)
                if n_classes == 2: tp.append(probs[:, 1].cpu())
                else:              tp.append(probs.cpu())
                tl.append(lbl.cpu())
        tp_np = torch.cat(tp).numpy(); tl_np = torch.cat(tl).numpy()
        try:
            test_auc = float(roc_auc_score(tl_np, tp_np) if n_classes == 2 else
                             roc_auc_score(tl_np, tp_np, multi_class='ovr', average='macro'))
        except Exception: test_auc = 0.5
        print(f"  seed {seed}: val={best_auc:.4f}  test={test_auc:.4f}")
        seed_results.append(test_auc)

    m = float(np.mean(seed_results)); s = float(np.std(seed_results))
    print(f"\n  {task_name}: {m:.3f} +- {s:.3f}")

    out = Path(out_dir) / f"{task_name}_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"task": task_name, "auroc_mean": round(m,4), "auroc_std": round(s,4),
                   "per_seed": [round(a,4) for a in seed_results]}, f, indent=2)
    return m, s


def main():
    run_task("./data/mel_cache/coughvid_covid", "coughvid_covid",  2, "./checkpoints/coughvid_tasks")
    run_task("./data/mel_cache/coughvid_sex",   "coughvid_sex",    2, "./checkpoints/coughvid_tasks")
    print("\nDone.")


if __name__ == "__main__":
    main()

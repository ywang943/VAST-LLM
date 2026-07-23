"""Run CSAF ablation E seeds 3 and 4 only, then save final results."""
import json, random, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.downstream import AttentionPool
from respvoice.csa_fusion import CrossScaleAttentionFusion
from scripts.run_opera_icbhi_disease import official_split
from scripts.run_full_local import class_weights_from_labels, labels_from_subset

LABEL_CACHE = "./data/mel_cache/opera_icbhi_disease"
D, BS, EPOCHS = 768, 16, 64


def extract_stages(htsat, pool1, pool2, mel):
    with torch.no_grad():
        x = mel.transpose(2, 3).transpose(1, 3)
        x = htsat.bn0(x); x = x.transpose(1, 3)
        B, C, T, F2 = x.shape
        target_T = int(htsat.spec_size * htsat.freq_ratio)
        if T < target_T:
            x = x.repeat(1, 1, (target_T // T) + 1, 1)
        x = x[:, :, :target_T, :]
        x = htsat.reshape_wav2img(x)
        x = htsat.patch_embed(x)
        if htsat.ape: x = x + htsat.absolute_pos_embed
        x = htsat.pos_drop(x)
        x, _ = htsat.layers[0](x); e1 = pool1(x)
        x, _ = htsat.layers[1](x); e2 = pool2(x)
        x, _ = htsat.layers[2](x); e3 = x
        x, _ = htsat.layers[3](x); e4 = htsat.norm(x)
    return [e1, e2, e3, e4]


class CSAFProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.csaf = CrossScaleAttentionFusion(D=D, n_scales=4, n_heads=8, depth=2,
                                               scale_dims=(192, 384, 768, 768))
        self.pool = AttentionPool(D)
        self.head = nn.Linear(D, 2)
    def forward(self, stages): return self.head(self.pool(self.csaf(stages)))


def run_seed(seed, htsat, pool1, pool2, train_ds, val_ds, test_ds, weights, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator().manual_seed(seed)
    train_l = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, generator=g)
    val_l = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=0)
    test_l = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    probe = CSAFProbe().to(device)
    params = list(probe.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS * len(train_l))

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        probe.train()
        for batch in train_l:
            mel = batch["mel"].to(device); lbl = batch["label"].to(device)
            stages = extract_stages(htsat, pool1, pool2, mel)
            opt.zero_grad()
            F.cross_entropy(probe(stages), lbl, weight=weights).backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        probe.eval()
        vp, vl = [], []
        with torch.no_grad():
            for b in val_l:
                s = extract_stages(htsat, pool1, pool2, b["mel"].to(device))
                vp.append(F.softmax(probe(s), dim=1)[:, 1].cpu()); vl.append(b["label"])
        try:
            vauc = float(roc_auc_score(torch.cat(vl).numpy(), torch.cat(vp).numpy()))
        except: vauc = 0.5
        if vauc > best_auc: best_auc = vauc; best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state: probe.load_state_dict(best_state)
    probe.eval()
    tp, tl = [], []
    with torch.no_grad():
        for b in test_l:
            s = extract_stages(htsat, pool1, pool2, b["mel"].to(device))
            tp.append(F.softmax(probe(s), dim=1)[:, 1].cpu()); tl.append(b["label"])
    try:
        return float(roc_auc_score(torch.cat(tl).numpy(), torch.cat(tp).numpy()))
    except: return 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = build_htsat_encoder(use_csaf=False)
    htsat = enc.htsat.to(device); pool1 = enc.pool1.to(device); pool2 = enc.pool2.to(device)
    for p in htsat.parameters(): p.requires_grad = False

    label_ds = CachedMelDataset(root=LABEL_CACHE,
                                 meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
                                 include_labels=True)
    train_ds, val_ds, test_ds = official_split(label_ds)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2).to(device)

    # Seeds 0-2 already done
    seed_aurocs = [0.9024, 0.9410, 0.8399]
    for seed in [3, 4]:
        auc = run_seed(seed, htsat, pool1, pool2, train_ds, val_ds, test_ds, weights, device)
        print(f"  E_csaf seed {seed}: AUROC={auc:.4f}")
        seed_aurocs.append(auc)

    m = float(np.mean(seed_aurocs)); s = float(np.std(seed_aurocs))
    print(f"\nE_csaf FINAL: {m:.3f} +/- {s:.3f}")
    print(f"  per-seed: {[round(a, 4) for a in seed_aurocs]}")

    result = {
        "A_stage4_only":  {"description": "Stage-4 only (OPERA baseline)", "auroc_mean": 0.812, "auroc_std": 0.011, "per_seed": []},
        "B_stage1_only":  {"description": "Stage-1 only (~1s)", "auroc_mean": 0.777, "auroc_std": 0.086, "per_seed": [0.724, 0.650, 0.835, 0.780, 0.898]},
        "C_stage3_only":  {"description": "Stage-3 only (~4s)", "auroc_mean": 0.906, "auroc_std": 0.012, "per_seed": [0.904, 0.927, 0.888, 0.904, 0.909]},
        "D_concat_nosttn":{"description": "Concat 4 stages (no attention)", "auroc_mean": 0.922, "auroc_std": 0.029, "per_seed": [0.927, 0.949, 0.912, 0.951, 0.873]},
        "E_csaf":         {"description": "CSAF cross-scale attention (ours)", "auroc_mean": round(m, 4), "auroc_std": round(s, 4), "per_seed": [round(a, 4) for a in seed_aurocs]},
    }
    Path("checkpoints/csaf_ablation").mkdir(parents=True, exist_ok=True)
    with open("checkpoints/csaf_ablation/csaf_ablation_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved: checkpoints/csaf_ablation/csaf_ablation_results.json")


if __name__ == "__main__":
    main()

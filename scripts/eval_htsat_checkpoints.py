"""
Evaluate saved HTS-AT + LeJEPA linear probe checkpoints.
Handles both old (single-scale) and new (CSAF) encoder architectures
by matching keys with strict=False.

Usage:
    python scripts/eval_htsat_checkpoints.py
"""
import sys, json, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

from data.respvoice_datasets import CachedMelDataset
from respvoice.model import RespVoiceModel
from respvoice.downstream import DownstreamHead
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from scripts.run_opera_icbhi_disease import official_split
from scripts.run_full_local import evaluate_binary
from torch.utils.data import DataLoader

LABEL_CACHE = "./data/mel_cache/opera_icbhi_disease"
ENC_CKPT    = "checkpoints/htsat_lejepa/htsat_lejepa_encoder.pt"
SEED_DIR    = "checkpoints/htsat_lejepa"
D, BS = 768, 16


def build_compatible_model(ckpt_path: str, device):
    """
    Load checkpoint with arch-compatible fallback.
    The saved checkpoint uses old HTSATLeJEPAEncoder (no CSAF).
    We rebuild the model by matching available keys only.
    """
    from respvoice.htsat_encoder import build_htsat_encoder

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_m = ckpt["config"].model
    saved_state = ckpt["model_state"]

    # Build base RespVoiceModel with HTS-AT encoder
    class LegacyHTSATModel(RespVoiceModel):
        """Wrapper that loads HTS-AT + linear head from checkpoint."""
        def __init__(self, cfg_m, enc_ckpt):
            super().__init__(cfg_m)
            # use_csaf=False: Stage-4 only, matches what was trained
            self.encoder = build_htsat_encoder(use_csaf=False)

    m = LegacyHTSATModel(cfg_m, ENC_CKPT)
    m.set_downstream_head(DownstreamHead(D, n_classes=2, use_regression=False))

    # Load only keys that shape-match (handles old↔new architecture mismatch)
    cur_state = m.state_dict()
    compatible = {
        k: v for k, v in saved_state.items()
        if k in cur_state and cur_state[k].shape == v.shape
    }
    skipped = [k for k in saved_state if k not in compatible]
    missing  = [k for k in cur_state if k not in compatible]

    m.load_state_dict(compatible, strict=False)
    m.to(device)

    if skipped:
        print(f"  [load] skipped {len(skipped)} shape-mismatched keys")
    if missing:
        print(f"  [load] {len(missing)} keys at random init (CSAF if new arch)")

    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_ds = CachedMelDataset(
        root=LABEL_CACHE,
        meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
        include_labels=True,
    )
    _, _, test_ds = official_split(label_ds)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

    print(f"=== HTS-AT + LeJEPA Linear Probe Results ===")
    print(f"  (Architecture: single-scale HTS-AT Stage-4 only, no CSAF)")
    print(f"  Note: CSAF experiment was queued but pyc cache used old code")
    print()

    results = []
    for seed in range(5):
        ckpt_path = f"{SEED_DIR}/htsat_lp_seed{seed}/stage3_best_auc.pt"
        if not Path(ckpt_path).exists():
            print(f"seed {seed}: missing checkpoint")
            continue
        m = build_compatible_model(ckpt_path, device)
        res = evaluate_binary(m, test_loader, device, use_quantized=False)
        auc = float(res.get("auroc", 0))
        print(f"  seed {seed}: AUROC={auc:.4f}  acc={res.get('accuracy',0):.4f}")
        results.append(auc)

    if results:
        m_auc = float(np.mean(results))
        s_auc = float(np.std(results))
        print(f"\n  HTS-AT + LeJEPA (no CSAF): {m_auc:.3f} +- {s_auc:.3f}")
        print(f"  per-seed: {[round(a,4) for a in results]}")
        print()
        print("  Compare:")
        print("    D128 LeJEPA (custom):           0.770 +- 0.034")
        print("    OPERA-CT (COLA, HTS-AT, bs=32): 0.812 +- 0.011")

        out = {
            "method": "HTS-AT Stage-4 + LeJEPA (no CSAF, pyc cache issue)",
            "note": "Ran with old single-scale encoder. CSAF rerun needed.",
            "auroc_mean": round(m_auc, 4),
            "auroc_std": round(s_auc, 4),
            "per_seed": [round(a, 4) for a in results],
        }
        p = Path(SEED_DIR) / "htsat_lejepa_results.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {p}")


if __name__ == "__main__":
    main()

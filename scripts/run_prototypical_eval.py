"""
Prototypical (zero-training) evaluation of pretrained representations.

For each downstream task and each k in {1, 5, 10, 20, 50}:
  - Randomly sample k examples per class from the support set (train split)
  - Compute class prototype = mean of L2-normalised features
  - Classify test set by cosine similarity to nearest prototype
  - Repeat 1000 episodes, report mean AUROC ± std

This measures the intrinsic geometric quality of the pretrained feature space
with zero downstream training — the closest equivalent to a k-NN probe.

Encoders compared:
  - opera_ct   : OPERA-CT COLA baseline (frozen HTS-AT, no CSAF)
  - checkpoint : RespVoice LeJEPA checkpoint (frozen HTS-AT + TPA-CSAF)

Representations compared per encoder:
  - stage4     : final HTS-AT stage only (768-dim, mean-pooled)
  - tpa_csaf   : TPA-CSAF output         (768-dim, mean-pooled)

Usage:
  # RespVoice scratch LeJEPA
  python scripts/run_prototypical_eval.py \
      --encoder checkpoint \
      --ckpt checkpoints/htsat_lejepa_scratch_clean/htsat_lejepa_best.pt \
      --out checkpoints/prototypical/scratch_results.json

  # OPERA-CT baseline
  python scripts/run_prototypical_eval.py \
      --encoder opera_ct \
      --out checkpoints/prototypical/opera_ct_results.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from respvoice.htsat_encoder import build_htsat_encoder
from scripts.run_frozen_multitask_benchmark import (
    TASKS, MultiCacheDataset, encode_stages, split_indices,
)

K_SHOTS = [1, 5, 10, 20, 50]
N_EPISODES = 1000
BATCH_SIZE = 128


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_features(encoder, dataset, device, representation):
    """Extract mean-pooled features for every sample. Returns (N, D) tensor."""
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    feats, labels = [], []
    encoder.eval()
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        if representation == "tpa_csaf":
            z = encoder(mel)            # (B, T, 768)
        else:
            stages = encode_stages(encoder, mel)
            z = stages[-1]              # stage4: (B, T, 768)
        feats.append(z.mean(dim=1).cpu())
        labels.append(batch["label"])
    return torch.cat(feats), torch.cat(labels)


# ── Prototypical episode ──────────────────────────────────────────────────────

def prototypical_auroc(support_feats, support_labels, query_feats, query_labels,
                       k, n_classes, rng):
    """
    Sample k examples per class from support, compute prototypes,
    classify query by cosine similarity, return AUROC.
    Returns None if a class has no support examples.
    """
    prototypes = []
    for c in range(n_classes):
        idx = (support_labels == c).nonzero(as_tuple=True)[0].tolist()
        if len(idx) < k:
            return None
        chosen = rng.sample(idx, k)
        proto = support_feats[chosen].mean(dim=0)
        prototypes.append(F.normalize(proto, dim=0))
    proto_mat = torch.stack(prototypes)             # (n_classes, D)

    q_norm = F.normalize(query_feats, dim=1)        # (N_q, D)
    sims = q_norm @ proto_mat.T                     # (N_q, n_classes)

    y = query_labels.numpy()
    try:
        if n_classes == 2:
            scores = sims[:, 1].numpy()
            return float(roc_auc_score(y, scores))
        else:
            probs = F.softmax(sims, dim=1).numpy()
            return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def run_prototypical_task(support_feats, support_labels,
                          query_feats, query_labels,
                          n_classes, seed=1337,
                          k_shots=None, n_episodes=None):
    """Run all k-shot settings for one task. Returns dict k→{mean, std}."""
    if k_shots is None:
        k_shots = K_SHOTS
    if n_episodes is None:
        n_episodes = N_EPISODES
    rng = random.Random(seed)
    results = {}
    for k in k_shots:
        aucs = []
        for _ in range(n_episodes):
            auc = prototypical_auroc(
                support_feats, support_labels,
                query_feats, query_labels,
                k, n_classes, rng,
            )
            if auc is not None:
                aucs.append(auc)
        if aucs:
            results[k] = {
                "auroc_mean": round(float(np.mean(aucs)), 4),
                "auroc_std":  round(float(np.std(aucs)),  4),
                "n_episodes": len(aucs),
            }
            print(f"    {k:>2}-shot: AUROC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
                  f"  (n={len(aucs)})")
        else:
            results[k] = {"auroc_mean": None, "auroc_std": None,
                          "n_episodes": 0, "note": "not enough support samples"}
            print(f"    {k:>2}-shot: skipped (not enough support samples)")
    return results


# ── Encoder loading ───────────────────────────────────────────────────────────

def load_encoder(encoder_type, ckpt_path, device):
    if encoder_type == "opera_ct":
        enc = build_htsat_encoder(use_csaf=False)
        for p in enc.parameters():
            p.requires_grad = False
        return enc.to(device).eval(), ["stage4"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    enc = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  Encoder load: missing={len(missing)}, unexpected={len(unexpected)}")
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device).eval(), ["stage4", "tpa_csaf"]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="opera_ct",
                        choices=["opera_ct", "checkpoint"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS),
                        default=list(TASKS))
    parser.add_argument("--k-shots", nargs="+", type=int, default=K_SHOTS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--out", default="checkpoints/prototypical/results.json")
    args = parser.parse_args()

    if args.encoder == "checkpoint" and not args.ckpt:
        parser.error("--ckpt required for checkpoint encoder")

    k_shots = args.k_shots
    n_episodes = args.episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    if args.ckpt:
        print(f"Checkpoint: {args.ckpt}")

    encoder, representations = load_encoder(args.encoder, args.ckpt, device)

    output = {
        "encoder": args.encoder,
        "checkpoint": args.ckpt,
        "protocol": "prototypical (zero training, cosine similarity)",
        "k_shots": k_shots,
        "n_episodes": n_episodes,
        "tasks": {},
    }

    for task_key in args.tasks:
        cfg = TASKS[task_key]
        dataset = MultiCacheDataset(cfg["roots"])
        train_idx, val_idx, test_idx = split_indices(dataset, cfg["split"])
        # Support = train + val (more support candidates → fairer episodes)
        support_idx = train_idx + val_idx
        n_classes = cfg["n_classes"]
        print(f"\n{'='*55}")
        print(f"Task: {cfg['name']}")
        print(f"  support={len(support_idx)}, query={len(test_idx)}, classes={n_classes}")

        task_result = {"note": cfg.get("note"), "representations": {}}

        for rep in representations:
            print(f"\n  Representation: {rep}")
            from torch.utils.data import Subset
            sup_feats, sup_labels = extract_all_features(
                encoder, Subset(dataset, support_idx), device, rep
            )
            qry_feats, qry_labels = extract_all_features(
                encoder, Subset(dataset, test_idx), device, rep
            )
            print(f"  Feature dim: {sup_feats.shape[1]}")

            task_result["representations"][rep] = run_prototypical_task(
                sup_feats, sup_labels, qry_feats, qry_labels,
                n_classes, seed=1337,
                k_shots=k_shots, n_episodes=n_episodes,
            )

        output["tasks"][task_key] = task_result

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

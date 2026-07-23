"""
RQ4 — Codebook-Event Interpretability Analysis.

Three-layer analysis:
  Layer 1: Codebook-Event Contingency Table (pure statistics, no LLM)
           Map each VQ token to its time window, look up ICBHI/SPRSound annotations,
           build a contingency table of codebook entries vs clinical events.
  Layer 2: Identify event-specific tokens (>70% occurrence in one event type).
  Layer 3: Compute Mutual Information between codebook and events.

Usage:
  python scripts/run_codebook_event_analysis.py \
    --encoder-ckpt checkpoints/htsat_lejepa_full/htsat_lejepa_best.pt \
    --vq-ckpt checkpoints/vq/speechgpt_vq.pt \
    --icbhi-dir opera_src/datasets/icbhi/ICBHI_final_database \
    --output checkpoints/codebook_event_analysis.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer


SR = 16000
TARGET_SEC = 8.0
N_TOKENS = 64  # number of VQ tokens per clip


def load_icbhi_annotations(icbhi_dir):
    """Load per-cycle annotations from ICBHI .txt files.
    Format: start_time end_time crackle(0/1) wheeze(0/1)
    """
    icbhi_dir = Path(icbhi_dir)
    annotations = {}
    for txt_file in sorted(icbhi_dir.glob("*.txt")):
        rec_id = txt_file.stem
        cycles = []
        for line in txt_file.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) < 4:
                parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            crackle, wheeze = int(parts[2]), int(parts[3])
            if crackle and wheeze:
                event = "both"
            elif crackle:
                event = "crackle"
            elif wheeze:
                event = "wheeze"
            else:
                event = "normal"
            cycles.append({"start": start, "end": end, "event": event})
        if cycles:
            annotations[rec_id] = cycles
    return annotations


def load_sprsound_annotations(sprsound_dir):
    """Load event-level annotations from SPRSound JSON files."""
    sprsound_dir = Path(sprsound_dir)
    annotations = {}
    json_dirs = list(sprsound_dir.glob("**/train_detection_json")) + \
                list(sprsound_dir.glob("**/valid_detection_json")) + \
                list(sprsound_dir.glob("**/test*_detection_json"))
    for json_dir in json_dirs:
        for jf in sorted(json_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
                events = data.get("event_annotation", [])
                if events:
                    rec_id = jf.stem
                    cycles = []
                    for ev in events:
                        start_ms = int(ev["start"])
                        end_ms = int(ev["end"])
                        etype = ev["type"].lower()
                        if "wheeze" in etype and "crackle" in etype:
                            event = "both"
                        elif "wheeze" in etype:
                            event = "wheeze"
                        elif "crackle" in etype:
                            event = "crackle"
                        elif "rhonchi" in etype:
                            event = "wheeze"
                        elif "stridor" in etype:
                            event = "wheeze"
                        else:
                            event = "normal"
                        cycles.append({
                            "start": start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "event": event,
                        })
                    annotations[rec_id] = cycles
            except Exception:
                continue
    return annotations


def get_event_at_time(cycles, t_start, t_end):
    """Return the dominant event type for a time window."""
    event_durations = defaultdict(float)
    for cycle in cycles:
        overlap_start = max(t_start, cycle["start"])
        overlap_end = min(t_end, cycle["end"])
        if overlap_start < overlap_end:
            event_durations[cycle["event"]] += overlap_end - overlap_start

    if not event_durations:
        return "normal"

    dominant = max(event_durations, key=event_durations.get)
    total = sum(event_durations.values())
    if event_durations.get("wheeze", 0) + event_durations.get("crackle", 0) + \
       event_durations.get("both", 0) > 0.3 * total:
        if "both" in event_durations:
            return "both"
        elif "wheeze" in event_durations and "crackle" in event_durations:
            return "both"
        elif "wheeze" in event_durations:
            return "wheeze"
        elif "crackle" in event_durations:
            return "crackle"
    return dominant


def load_encoder_and_vq(encoder_ckpt, vq_ckpt, device):
    """Load frozen encoder and VQ codebook."""
    ckpt = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    vq_ckpt_data = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    K = vq_ckpt_data["codebook_size"]
    D = vq_ckpt_data["D"]
    vq = VectorQuantizer(codebook_size=K, D=D)
    vq.load_state_dict(vq_ckpt_data["vq_state"], strict=False)
    vq = vq.to(device).eval()

    return encoder, vq, K


def tokenize_audio(encoder, vq, mel_tensor, device):
    """Get VQ token IDs for a mel spectrogram."""
    mel = mel_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        z_cont = encoder(mel)  # (1, L, D)
        vq_out = vq(z_cont)
        token_ids = vq_out["ids"].squeeze(0).cpu().numpy()  # (L,)
    return token_ids


def build_contingency_table(
    encoder, vq, K, mel_dir, wav_dir, annotations, source_name,
    device, target_sec=8.0
):
    """Build codebook-event contingency table."""
    mel_dir = Path(mel_dir)
    meta_path = mel_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  No metadata.json in {mel_dir}, skipping")
        return None, []

    meta = json.loads(meta_path.read_text())
    samples = meta.get("samples", meta if isinstance(meta, list) else [])

    contingency = np.zeros((K, 4), dtype=np.int64)  # K x [normal, crackle, wheeze, both]
    event_map = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}
    token_event_pairs = []
    n_processed = 0

    for sample in samples:
        mel_path = mel_dir / sample["path"]
        if not mel_path.exists():
            continue

        # Extract original recording ID from metadata
        orig_path = sample.get("original_path", "")
        if orig_path:
            rec_id = Path(orig_path).stem
        else:
            rec_id = sample["path"].rsplit(".", 1)[0]

        matched_ann = annotations.get(rec_id)

        if matched_ann is None:
            continue

        try:
            mel = torch.load(str(mel_path), map_location="cpu")
            token_ids = tokenize_audio(encoder, vq, mel, device)
        except Exception:
            continue

        n_tokens = len(token_ids)
        token_duration = target_sec / n_tokens

        for pos, tid in enumerate(token_ids):
            t_start = pos * token_duration
            t_end = (pos + 1) * token_duration
            event = get_event_at_time(matched_ann, t_start, t_end)
            eidx = event_map.get(event, 0)
            contingency[tid, eidx] += 1
            token_event_pairs.append((int(tid), event))

        n_processed += 1

    print(f"  {source_name}: processed {n_processed} recordings, "
          f"{len(token_event_pairs)} token-event pairs")
    return contingency, token_event_pairs


def compute_mutual_information(contingency):
    """Compute MI between codebook entries and events."""
    total = contingency.sum()
    if total == 0:
        return 0.0
    p_joint = contingency / total
    p_code = contingency.sum(axis=1, keepdims=True) / total
    p_event = contingency.sum(axis=0, keepdims=True) / total

    mi = 0.0
    for k in range(contingency.shape[0]):
        for e in range(contingency.shape[1]):
            if p_joint[k, e] > 0 and p_code[k, 0] > 0 and p_event[0, e] > 0:
                mi += p_joint[k, e] * np.log2(
                    p_joint[k, e] / (p_code[k, 0] * p_event[0, e])
                )
    return float(mi)


def identify_event_specific_tokens(contingency, threshold=0.7):
    """Find tokens that are >threshold concentrated in one event type."""
    event_names = ["normal", "crackle", "wheeze", "both"]
    specific_tokens = []
    row_sums = contingency.sum(axis=1)

    for k in range(contingency.shape[0]):
        if row_sums[k] < 5:
            continue
        fracs = contingency[k] / row_sums[k]
        max_idx = fracs.argmax()
        if fracs[max_idx] >= threshold:
            specific_tokens.append({
                "token_id": int(k),
                "dominant_event": event_names[max_idx],
                "fraction": float(fracs[max_idx]),
                "counts": {event_names[i]: int(contingency[k, i]) for i in range(4)},
            })
    return specific_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--vq-ckpt", required=True)
    parser.add_argument("--icbhi-dir", default="opera_src/datasets/icbhi/ICBHI_final_database")
    parser.add_argument("--sprsound-dir", default="data/SPRSound")
    parser.add_argument("--output", default="checkpoints/codebook_event_analysis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder and VQ
    print("Loading encoder and VQ...")
    encoder, vq, K = load_encoder_and_vq(args.encoder_ckpt, args.vq_ckpt, device)
    print(f"Codebook size: {K}")

    # Load annotations
    print("Loading ICBHI annotations...")
    icbhi_ann = load_icbhi_annotations(args.icbhi_dir)
    print(f"  {len(icbhi_ann)} recordings with annotations")

    print("Loading SPRSound annotations...")
    sprsound_ann = load_sprsound_annotations(args.sprsound_dir)
    print(f"  {len(sprsound_ann)} recordings with annotations")

    # Build contingency tables
    print("\n=== ICBHI Contingency Table ===")
    icbhi_contingency, icbhi_pairs = build_contingency_table(
        encoder, vq, K,
        "data/mel_cache/opera_icbhi_disease",
        "data/wav_cache/opera_icbhi_disease",
        icbhi_ann, "ICBHI", device
    )

    # Combine results
    total_contingency = np.zeros((K, 4), dtype=np.int64)
    if icbhi_contingency is not None:
        total_contingency += icbhi_contingency

    # Compute metrics
    mi = compute_mutual_information(total_contingency)
    specific = identify_event_specific_tokens(total_contingency, threshold=0.7)

    event_names = ["normal", "crackle", "wheeze", "both"]
    event_totals = {event_names[i]: int(total_contingency[:, i].sum()) for i in range(4)}

    n_specific_by_event = Counter(t["dominant_event"] for t in specific)
    active_codes = int((total_contingency.sum(axis=1) > 0).sum())

    results = {
        "codebook_size": K,
        "active_codes": active_codes,
        "total_token_event_pairs": int(total_contingency.sum()),
        "event_distribution": event_totals,
        "mutual_information_bits": mi,
        "n_event_specific_tokens_70pct": len(specific),
        "specific_by_event": dict(n_specific_by_event),
        "top_specific_tokens": sorted(specific, key=lambda x: -x["fraction"])[:20],
    }

    # Random baseline MI (permutation test)
    print("\nComputing random baseline MI (100 permutations)...")
    random_mis = []
    rng = np.random.RandomState(42)
    flat = total_contingency.flatten()
    for _ in range(100):
        perm = rng.permutation(flat).reshape(total_contingency.shape)
        random_mis.append(compute_mutual_information(perm))
    results["random_baseline_mi_mean"] = float(np.mean(random_mis))
    results["random_baseline_mi_std"] = float(np.std(random_mis))
    results["mi_significant"] = mi > np.mean(random_mis) + 3 * np.std(random_mis)

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Active codes: {active_codes}/{K}")
    print(f"Total token-event pairs: {total_contingency.sum()}")
    print(f"Event distribution: {event_totals}")
    print(f"Mutual Information: {mi:.4f} bits")
    print(f"Random baseline MI: {np.mean(random_mis):.4f} ± {np.std(random_mis):.4f}")
    print(f"MI significant (>3σ): {results['mi_significant']}")
    print(f"Event-specific tokens (>70%): {len(specific)}/{active_codes}")
    print(f"  Normal-dominant: {n_specific_by_event.get('normal', 0)}")
    print(f"  Crackle-dominant: {n_specific_by_event.get('crackle', 0)}")
    print(f"  Wheeze-dominant: {n_specific_by_event.get('wheeze', 0)}")
    print(f"  Both-dominant: {n_specific_by_event.get('both', 0)}")

    if specific:
        print(f"\nTop 10 event-specific tokens:")
        for t in sorted(specific, key=lambda x: -x["fraction"])[:10]:
            print(f"  rv_{t['token_id']:05d}: {t['dominant_event']} "
                  f"({t['fraction']:.1%}) counts={t['counts']}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x)))
    print(f"\nSaved to {args.output}")

    # Save full contingency table as numpy
    np.save(args.output.replace(".json", "_contingency.npy"), total_contingency)
    print(f"Contingency table saved: {total_contingency.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate time-localized VAST token evidence for saved RQ3 examples.

This is an occlusion analysis for qualitative paper examples.  For each saved
classification prompt, the script removes one coarse time column of VAST tokens
at a time and re-scores the candidate labels with the trained LoRA model.  A
time window is treated as supporting the selected answer if removing its tokens
decreases the selected-answer probability or the selected-vs-alternative margin.

The 64 VAST tokens come from the HTS-AT/Swin final 8x8 grid.  With the mel-only
8-second inputs used here, each grid column is approximately one second.  The
mapping is therefore approximate: token index i maps to row i//8 and time column
i%8.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise RuntimeError("peft is required to load the saved adapter") from exc


ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"<rv_(\d{3})>")


def load_examples(path: Path, mode: str, task: str):
    data = json.loads(path.read_text())
    return data["results"][mode]["tasks"][task]["examples"]


def split_prompt(prompt: str):
    marker = "Audio tokens: "
    start = prompt.index(marker) + len(marker)
    suffix_start = prompt.index("\nClassify as one of:", start)
    prefix = prompt[:start]
    token_text = prompt[start:suffix_start]
    suffix = prompt[suffix_start:]
    tokens = [int(x) for x in TOKEN_RE.findall(token_text)]
    return prefix, tokens, suffix


def format_tokens(tokens):
    return " ".join(f"<rv_{tid:03d}>" for tid in tokens)


def rebuild_prompt(prefix, tokens, suffix):
    return prefix + format_tokens(tokens) + suffix


def score_candidates(model, tokenizer, prompt, candidates, device, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cand_scores = []
    for cand in candidates:
        cand_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        full_ids = prompt_ids + cand_ids
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
        input_tensor = torch.tensor([full_ids], device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_tensor).logits[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        score = 0.0
        count = 0
        start = len(full_ids) - len(cand_ids)
        for pos, tid in enumerate(cand_ids, start=start):
            if pos > 0 and (pos - 1) < logits.shape[0]:
                score += log_probs[pos - 1, tid].item()
                count += 1
        cand_scores.append(score / max(count, 1))
    probs = F.softmax(torch.tensor(cand_scores), dim=0).cpu().numpy()
    return {c: float(p) for c, p in zip(candidates, probs)}


def margin(probs, selected, alternative):
    return float(probs[selected] - probs[alternative])


def summarize_example(example, full_probs, windows, selected, alternative):
    top_support = sorted(windows, key=lambda x: x["margin_drop"], reverse=True)
    top_oppose = sorted(windows, key=lambda x: x["margin_drop"])
    best = top_support[0]
    second = top_support[1] if len(top_support) > 1 else None

    pieces = [
        f"Prediction: {selected} "
        f"(P={full_probs[selected]:.3f} vs {alternative}={full_probs[alternative]:.3f})."
    ]
    if best["margin_drop"] > 0:
        pieces.append(
            f"Strongest supporting window: {best['time_window']} "
            f"with tokens {best['token_text_short']}; removing it lowers the "
            f"{selected} margin by {best['margin_drop']:.3f}."
        )
    else:
        pieces.append(
            "No removed time window reduced the selected-answer margin; "
            "this sample has no positive token-time evidence under the occlusion test."
        )
    if second and second["margin_drop"] > 0:
        pieces.append(
            f"Secondary support: {second['time_window']} "
            f"(margin drop {second['margin_drop']:.3f})."
        )
    if top_oppose[0]["margin_drop"] < 0:
        pieces.append(
            f"The most opposing/uncertain window is {top_oppose[0]['time_window']}, "
            f"where removal increases the selected margin by {-top_oppose[0]['margin_drop']:.3f}."
        )
    return " ".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--examples-json", required=True)
    parser.add_argument("--mode", default="audio_text")
    parser.add_argument("--task", default="coswara_smoker_breathing")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    adapter_dir = ROOT / args.adapter_dir
    examples_path = ROOT / args.examples_json
    examples = load_examples(examples_path, args.mode, args.task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.to(device)
    model.eval()

    outputs = []
    for ex_idx, example in enumerate(examples):
        prefix, tokens, suffix = split_prompt(example["prompt"])
        if len(tokens) != 64:
            print(f"WARNING: {example['mel_path']} has {len(tokens)} tokens, expected 64")

        candidates = list(example["candidate_probs"].keys())
        selected = example["pred_answer"]
        alternative = next(c for c in candidates if c != selected)
        full_prompt = rebuild_prompt(prefix, tokens, suffix)
        full_probs = score_candidates(
            model, tokenizer, full_prompt, candidates, device, args.max_length
        )
        full_margin = margin(full_probs, selected, alternative)

        n_cols = 8
        seconds_per_col = args.duration_sec / n_cols
        windows = []
        for col in range(n_cols):
            removed_indices = [i for i in range(len(tokens)) if (i % n_cols) == col]
            kept_tokens = [tid for i, tid in enumerate(tokens) if i not in set(removed_indices)]
            occluded_prompt = rebuild_prompt(prefix, kept_tokens, suffix)
            occ_probs = score_candidates(
                model, tokenizer, occluded_prompt, candidates, device, args.max_length
            )
            occ_margin = margin(occ_probs, selected, alternative)
            removed_tokens = [tokens[i] for i in removed_indices]
            start_t = col * seconds_per_col
            end_t = (col + 1) * seconds_per_col
            windows.append({
                "time_column": col,
                "time_window": f"{start_t:.1f}-{end_t:.1f}s",
                "removed_token_indices": removed_indices,
                "removed_token_rows": [i // n_cols for i in removed_indices],
                "removed_token_ids": removed_tokens,
                "removed_tokens": [f"<rv_{tid:03d}>" for tid in removed_tokens],
                "token_text_short": " ".join(f"<rv_{tid:03d}>" for tid in removed_tokens[:4])
                + (" ..." if len(removed_tokens) > 4 else ""),
                "occluded_candidate_probs": occ_probs,
                "selected_prob_after_removal": occ_probs[selected],
                "selected_prob_drop": float(full_probs[selected] - occ_probs[selected]),
                "margin_after_removal": occ_margin,
                "margin_drop": float(full_margin - occ_margin),
            })

        top_support = sorted(windows, key=lambda x: x["margin_drop"], reverse=True)
        out = {
            "index": ex_idx,
            "mel_path": example["mel_path"],
            "true_answer": example["true_answer"],
            "pred_answer": selected,
            "alternative_answer": alternative,
            "dms_text": example["dms_text"],
            "time_mapping_note": (
                "Approximate mapping from the final 8x8 HTS-AT/VAST token grid: "
                "token index i maps to row i//8 and time column i%8; with 8-second "
                "mel inputs, each column is about 1.0 second."
            ),
            "full_candidate_probs_saved": example["candidate_probs"],
            "full_candidate_probs_rescored": full_probs,
            "full_margin": full_margin,
            "windows": windows,
            "top_support_windows": top_support[:3],
            "top_opposing_windows": sorted(windows, key=lambda x: x["margin_drop"])[:3],
            "paper_style_summary": summarize_example(
                example, full_probs, windows, selected, alternative
            ),
            "classification_prompt": example["prompt"],
        }
        outputs.append(out)

        print(f"\n=== Example {ex_idx}: {example['mel_path']} ===")
        print(out["paper_style_summary"])
        print("Top support windows:")
        for w in out["top_support_windows"]:
            print(
                f"  {w['time_window']}: margin_drop={w['margin_drop']:.4f}, "
                f"prob_drop={w['selected_prob_drop']:.4f}, tokens={w['token_text_short']}"
            )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "examples_json": str(examples_path),
        "task": args.task,
        "mode": args.mode,
        "duration_sec": args.duration_sec,
        "occlusion": (
            "For each time column, remove the 8 VAST tokens in that column from "
            "the prompt and re-score the candidate answers. Positive margin_drop "
            "means the removed window supported the original prediction."
        ),
        "examples": outputs,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    sys.exit(main())

"""
RQ3: LLM-based multi-task diagnosis with VAST discrete tokens + DMS text.

Architecture: frozen VAST encoder → VQ tokens → LLM vocabulary expansion → LoRA fine-tuning.
Following RespLLM protocol: train on seen tasks, evaluate on seen + zero-shot unseen tasks.

Fixes over run_llm_multitask.py:
  - bf16 training (not fp16) to prevent NaN
  - Proper new-token embedding initialization (mean of existing embeddings)
  - Gradient clipping
  - Gradient accumulation for effective larger batch
  - Warmup scheduler
  - Fixed candidate_scores evaluation bug
  - Multiple input modes: audio_only, text_only, audio_text
  - Coswara COVID + Smoker tasks
  - Support for OpenBioLLM-8B and other Llama models
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from respvoice.dual_input_encoder import build_dual_input_encoder
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    LoraConfig = PeftModel = TaskType = get_peft_model = None

SR = 16000
WAV_LEN = SR * 8

TASKS = {
    "icbhi_copd": {
        "name": "ICBHI COPD Detection",
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "wav_root": "data/wav_cache/opera_icbhi_disease",
        "dms_source": "icbhi",
        "instruction": (
            "Dataset: ICBHI Respiratory Sound Database. "
            "Task: classify whether this person has COPD based on the audio and clinical information."
        ),
        "labels": {0: "healthy", 1: "copd"},
    },
    "svd_pathology": {
        "name": "SVD Voice Pathology",
        "mel_root": "data/mel_cache/svd_full",
        "dms_source": "svd",
        "instruction": (
            "Dataset: Saarbruecken Voice Database. "
            "Task: classify whether this voice recording indicates vocal fold pathology."
        ),
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
    "copd_severity": {
        "name": "COPD Severity",
        "mel_root": "data/mel_cache/opera_copd",
        "dms_source": "icbhi",
        "instruction": (
            "Dataset: Respiratory@TR. "
            "Task: classify COPD severity level."
        ),
        "labels": {0: "severity zero", 1: "severity one", 2: "severity two",
                   3: "severity three", 4: "severity four"},
    },
    "kauh_obstructive": {
        "name": "KAUH Obstructive Disease",
        "mel_root": "data/mel_cache/opera_kauh",
        "dms_source": "kauh",
        "instruction": (
            "Dataset: KAUH Respiratory Database. "
            "Task: classify whether this person has obstructive airway disease."
        ),
        "labels": {0: "healthy", 1: "obstructive disease"},
    },
    "coswara_covid_cough": {
        "name": "Coswara COVID (Cough)",
        "mel_root": "data/mel_cache/coswara_covid_cough",
        "dms_source": "coswara_covid_cough",
        "instruction": (
            "Dataset: Coswara COVID-19 Sounds. "
            "Task: classify whether this person has COVID-19 based on the cough audio and clinical information."
        ),
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "coswara_covid_exhale": {
        "name": "Coswara COVID (Exhalation)",
        "mel_root": "data/mel_cache/coswara_covid_exhale",
        "dms_source": "coswara_covid_breathing",
        "instruction": (
            "Dataset: Coswara COVID-19 Sounds. "
            "Task: classify whether this person has COVID-19 based on the exhalation audio and clinical information."
        ),
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "coswara_covid_breathing": {
        "name": "Coswara COVID (Breathing)",
        "mel_root": "data/mel_cache/coswara_covid_breathing",
        "dms_source": "coswara_covid_breathing",
        "instruction": (
            "Dataset: Coswara COVID-19 Sounds. "
            "Task: classify whether this person has COVID-19 based on the breathing audio and clinical information."
        ),
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "coswara_smoker_cough": {
        "name": "Coswara Smoker (Cough)",
        "mel_root": "data/mel_cache/coswara_smoker_cough",
        "dms_source": "coswara_smoker_cough",
        "instruction": (
            "Dataset: Coswara COVID-19 Sounds. "
            "Task: classify whether this person is a smoker based on the cough audio and clinical information."
        ),
        "labels": {0: "non smoker", 1: "smoker"},
    },
    "coswara_smoker_breathing": {
        "name": "Coswara Smoker (Breathing)",
        "mel_root": "data/mel_cache/coswara_smoker_breathing",
        "dms_source": "coswara_smoker_breathing",
        "instruction": (
            "Dataset: Coswara COVID-19 Sounds. "
            "Task: classify whether this person is a smoker based on the breathing audio and clinical information."
        ),
        "labels": {0: "non smoker", 1: "smoker"},
    },
    "uk_covid_cough": {
        "name": "UK COVID-19 Sounds (Cough)",
        "mel_root": "data/mel_cache/uk_covid_cough",
        "dms_source": None,
        "instruction": (
            "Dataset: UK COVID-19 Vocal Audio Dataset. "
            "Task: classify whether this person has COVID-19 based on the cough audio and clinical information."
        ),
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "b2ai_voice_pathology": {
        "name": "Bridge2AI Voice Pathology",
        "mel_root": "data/mel_cache/b2ai_voice_pathology",
        "dms_source": "b2ai",
        "instruction": (
            "Dataset: Bridge2AI Voice dataset. "
            "Task: classify whether this person has voice pathology based on the prolonged vowel audio and clinical information."
        ),
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
    "b2ai_laryngeal_cancer": {
        "name": "Bridge2AI Laryngeal Cancer",
        "mel_root": "data/mel_cache/b2ai_laryngeal_cancer",
        "dms_source": "b2ai",
        "instruction": (
            "Dataset: Bridge2AI Voice dataset. "
            "Task: classify whether this person has laryngeal cancer or precancerous lesions based on the prolonged vowel audio and clinical information."
        ),
        "labels": {0: "healthy voice", 1: "laryngeal cancer"},
    },
    "b2ai_benign_lesions": {
        "name": "Bridge2AI Benign Lesions",
        "mel_root": "data/mel_cache/b2ai_benign_lesions",
        "dms_source": "b2ai",
        "instruction": (
            "Dataset: Bridge2AI Voice dataset. "
            "Task: classify whether this person has benign lesions of the vocal cord based on the prolonged vowel audio and clinical information."
        ),
        "labels": {0: "healthy voice", 1: "benign vocal cord lesions"},
    },
    "b2ai_laryngeal_dystonia": {
        "name": "Bridge2AI Laryngeal Dystonia",
        "mel_root": "data/mel_cache/b2ai_laryngeal_dystonia",
        "dms_source": "b2ai",
        "instruction": (
            "Dataset: Bridge2AI Voice dataset. "
            "Task: classify whether this person has spasmodic dysphonia or laryngeal tremor based on the prolonged vowel audio and clinical information."
        ),
        "labels": {0: "healthy voice", 1: "spasmodic dysphonia"},
    },
    "svd_pathology_target": {
        "name": "SVD Voice Pathology Target",
        "mel_root": "data/mel_cache/svd_full_target_alltest",
        "dms_source": "svd",
        "instruction": (
            "Dataset: Saarbruecken Voice Database target split. "
            "Task: classify whether this voice recording indicates vocal fold pathology."
        ),
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
}

B2AI_STRICT_EXCLUDE_DIAGNOSES = {
    "laryngeal_cancer",
    "precancerous_lesions",
    "benign_lesions",
    "laryngeal_dystonia",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder_and_vq(encoder_ckpt, vq_ckpt, device, encoder_type="htsat"):
    ckpt = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("encoder.", "", 1): v
             for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    if encoder_type == "dual":
        encoder = build_dual_input_encoder(
            ckpt_path=None, freeze_backbone=True, freeze_cnn=True, use_csaf=True,
        )
    else:
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    vq_data = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    vq_encoder_type = vq_data.get("encoder_type")
    vq_encoder_ckpt = str(vq_data.get("encoder_checkpoint", ""))
    if encoder_type == "htsat":
        if vq_encoder_type and vq_encoder_type != "htsat_mel":
            raise RuntimeError(
                f"VQ/encoder mismatch: encoder_type=htsat but VQ was trained for "
                f"{vq_encoder_type} ({vq_ckpt})"
            )
        if not vq_encoder_type and "dual" in vq_encoder_ckpt.lower():
            raise RuntimeError(
                f"VQ/encoder mismatch: encoder_type=htsat but VQ metadata points to "
                f"dual encoder checkpoint {vq_encoder_ckpt}"
            )
    if encoder_type == "dual" and vq_encoder_type == "htsat_mel":
        raise RuntimeError(
            f"VQ/encoder mismatch: encoder_type=dual but VQ is mel-only ({vq_ckpt})"
        )
    K = vq_data["codebook_size"]
    D = vq_data["D"]
    l2_norm = vq_data.get("l2_normalize", True)
    vq = VectorQuantizer(codebook_size=K, D=D, l2_normalize=l2_norm)
    vq.load_state_dict(vq_data["vq_state"], strict=False)
    print(f"  VQ: K={K}, D={D}, l2_normalize={l2_norm}")
    vq = vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad = False

    return encoder, vq, K


def load_wav_for_mel(wav_root, mel_path):
    wav_path = ROOT / wav_root / mel_path.replace(".pt", ".npy")
    wav = np.load(str(wav_path)).astype(np.float32)
    wav = (wav - wav.mean()) / (wav.std() + 1e-8)
    if len(wav) >= WAV_LEN:
        wav = wav[:WAV_LEN]
    else:
        wav = np.pad(wav, (0, WAV_LEN - len(wav)))
    return torch.from_numpy(wav)


def tokenize_dataset(task_key, encoder, vq, device, encoder_type="htsat"):
    cfg = TASKS[task_key]
    mel_dir = ROOT / cfg["mel_root"]
    meta_path = mel_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  WARNING: {meta_path} not found, skipping {task_key}")
        return [], [], [], [], []

    meta = json.loads(meta_path.read_text())
    samples = meta.get("samples", [])

    token_ids_list, labels, splits, mel_paths, dms_texts, sample_infos = [], [], [], [], [], []

    for sample in samples:
        if "label" not in sample:
            continue
        mel_path = mel_dir / sample["path"]
        if not mel_path.exists():
            continue
        try:
            mel = torch.load(str(mel_path), map_location="cpu")
            mel = mel.unsqueeze(0).to(device)
            with torch.no_grad():
                if encoder_type == "dual":
                    wav_root = cfg.get("wav_root")
                    if not wav_root:
                        raise RuntimeError(f"{task_key} has no wav_root for dual encoder")
                    wav = load_wav_for_mel(wav_root, sample["path"]).unsqueeze(0).to(device)
                    z_cont = encoder(mel, wav)
                else:
                    z_cont = encoder(mel)
                vq_out = vq(z_cont)
                ids = vq_out["ids"].squeeze(0).cpu().numpy()
            token_ids_list.append(ids)
            labels.append(int(sample["label"]))
            splits.append(sample.get("split", "train"))
            mel_paths.append(sample["path"])
            dms_texts.append(sample.get("dms_text", ""))
            sample_infos.append({
                "pid": str(sample.get("pid", sample.get("participant_id", ""))),
                "participant_id": str(sample.get("participant_id", sample.get("pid", ""))),
                "diagnosis": str(sample.get("diagnosis", "")),
            })
        except Exception:
            continue

    return token_ids_list, labels, splits, mel_paths, dms_texts, sample_infos


def token_cache_path(cache_dir, task_key, encoder_ckpt, vq_ckpt, encoder_type):
    if not cache_dir:
        return None
    key = "|".join([
        task_key,
        encoder_type,
        str(Path(encoder_ckpt).resolve()),
        str(Path(vq_ckpt).resolve()),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return ROOT / cache_dir / f"{task_key}_{encoder_type}_{digest}.pt"


def tokenize_dataset_cached(task_key, encoder, vq, device, args):
    cache_path = token_cache_path(
        args.token_cache_dir, task_key, args.encoder_ckpt, args.vq_ckpt,
        args.encoder_type,
    )
    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  Loaded token cache: {cache_path}")
        return (
            data["token_ids"], data["labels"], data["splits"],
            data["mel_paths"], data["dms_texts"], data["sample_infos"],
        )

    data = tokenize_dataset(task_key, encoder, vq, device, args.encoder_type)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "token_ids": data[0],
            "labels": data[1],
            "splits": data[2],
            "mel_paths": data[3],
            "dms_texts": data[4],
            "sample_infos": data[5],
        }, cache_path)
        print(f"  Saved token cache: {cache_path}")
    return data


def load_dms_from_templates(task_key):
    cfg = TASKS[task_key]
    source = cfg.get("dms_source", task_key)
    dms_path = ROOT / f"data/dms_templates/{source}_dms.json"
    if not dms_path.exists():
        return {}
    dms_data = json.loads(dms_path.read_text())
    dms_map = {}
    for item in dms_data:
        mel_path = item.get("mel_path", "")
        if mel_path:
            dms_map[mel_path] = item.get("dms_text", "")
    return dms_map


def simplify_dms_text(text, mode="rich"):
    """Control clinical-text strength for leakage/sensitivity ablations."""
    if mode == "rich":
        return text
    if mode == "empty":
        return ""
    if mode != "minimal":
        raise ValueError(f"Unknown DMS mode: {mode}")

    parts = []
    for field in ("Gender", "Age"):
        marker = f"{field}:"
        if marker in text:
            after = text.split(marker, 1)[1]
            value = after.split(".", 1)[0].strip()
            if value:
                parts.append(f"{field}: {value}.")
    return " ".join(parts)


def audio_token_text(ids):
    toks = " ".join(f"<rv_{int(x):03d}>" for x in ids)
    return f"<AUD_BOS> {toks} <AUD_EOS>"


def build_prompt(task_key, token_ids, dms_text="", input_mode="audio_text"):
    cfg = TASKS[task_key]
    parts = [
        "You are a medical acoustic diagnostic assistant. Output only the label text.\n",
        f"Task: {cfg['instruction']}\n",
    ]

    if input_mode == "audio_text":
        parts.append(
            "Use both modalities. The audio tokens encode time-ordered acoustic "
            "patterns; the clinical information provides reliable subject context. "
            "When clinical information contains disease-relevant symptoms, voice "
            "quality, or risk factors, treat it as primary evidence and use the "
            "audio tokens to corroborate or refine the decision.\n"
        )
        if token_ids is not None:
            parts.append(f"Audio tokens: {audio_token_text(token_ids)}\n")
        if dms_text:
            parts.append(f"Clinical information: {dms_text}\n")
    elif input_mode == "audio_only":
        parts.append("Use only the acoustic token sequence; no clinical metadata is available.\n")
        if token_ids is not None:
            parts.append(f"Audio tokens: {audio_token_text(token_ids)}\n")
    elif input_mode == "text_only":
        parts.append("Use only the clinical information; no audio tokens are available.\n")
        if dms_text:
            parts.append(f"Clinical information: {dms_text}\n")

    label_options = ", ".join(f'"{v}"' for v in cfg["labels"].values())
    parts.append(f"Classify as one of: {label_options}\n")
    parts.append("Answer:")

    return "".join(parts)


def strict_target_pids(all_task_data):
    pids = set()
    for task_key in (
        "b2ai_laryngeal_cancer",
        "b2ai_benign_lesions",
        "b2ai_laryngeal_dystonia",
    ):
        for sample in all_task_data.get(task_key, []):
            pid = sample.get("pid") or sample.get("participant_id")
            if pid:
                pids.add(str(pid))
    return pids


def keep_strict_train_sample(sample, all_task_data):
    """Filter training samples for strict disease/participant-disjoint RQ3."""
    if sample["task_key"] != "b2ai_voice_pathology":
        return True

    diagnosis = sample.get("diagnosis", "")
    if diagnosis in B2AI_STRICT_EXCLUDE_DIAGNOSES:
        return False

    pid = sample.get("pid") or sample.get("participant_id")
    if pid and str(pid) in strict_target_pids(all_task_data):
        return False

    return True


def keep_strict_eval_sample(sample, task_key):
    """Use held-out participant test split for cross-modal Coswara targets."""
    if task_key in ("coswara_covid_breathing", "coswara_smoker_breathing"):
        return sample.get("split") == "test"
    return True


class MultiTaskDataset(Dataset):
    def __init__(self, task_samples, tokenizer, max_length, input_mode="audio_text",
                 modality_dropout=0.0):
        self.samples = task_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = input_mode
        self.modality_dropout = modality_dropout

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        s = self.samples[item]
        mode = self.input_mode
        if mode == "audio_text" and self.modality_dropout > 0:
            r = random.random()
            if r < self.modality_dropout / 2:
                mode = "audio_only"
            elif r < self.modality_dropout:
                mode = "text_only"
        prompt = build_prompt(
            s["task_key"], s["token_ids"], s.get("dms_text", ""), mode
        )
        answer = " " + s["label_text"]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        eos = [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids + eos
        labels = [-100] * len(prompt_ids) + answer_ids + eos

        if len(input_ids) > self.max_length:
            keep_prompt = max(self.max_length - len(answer_ids) - len(eos), 1)
            prompt_ids = prompt_ids[-keep_prompt:]
            input_ids = prompt_ids + answer_ids + eos
            labels = [-100] * len(prompt_ids) + answer_ids + eos
        pad_id = self.tokenizer.pad_token_id or 0
        pad_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_label": s["label"],
            "task_key": s["task_key"],
        }


def candidate_scores(model, tokenizer, prompts, candidates, device, max_length):
    all_probs = []
    pred = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        cand_scores = []
        for ci, cand in enumerate(candidates):
            cand_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
            full_ids = prompt_ids + cand_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[-max_length:]
            input_tensor = torch.tensor([full_ids], device=device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_tensor)
                logits = out.logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            score = 0.0
            count = 0
            start = len(full_ids) - len(cand_ids)
            for pos, tid in enumerate(cand_ids, start=start):
                if pos > 0 and (pos - 1) < logits.shape[0]:
                    score += log_probs[pos - 1, tid].item()
                    count += 1
            cand_scores.append(score / max(count, 1))
        probs = F.softmax(torch.tensor(cand_scores), dim=0).numpy()
        all_probs.append(probs)
        pred.append(int(probs.argmax()))
    return np.stack(all_probs), pred


def evaluate_task(model, tokenizer, task_samples, task_key, device, max_length, input_mode,
                  num_examples=0):
    cfg = TASKS[task_key]
    candidates = list(cfg["labels"].values())
    prompts = []
    true_labels = []

    for s in task_samples:
        prompt = build_prompt(
            task_key, s["token_ids"], s.get("dms_text", ""), input_mode
        )
        prompts.append(prompt)
        true_labels.append(s["label"])

    probs, predicted = candidate_scores(
        model, tokenizer, prompts, candidates, device, max_length,
    )

    y_true = np.array(true_labels)
    y_pred = np.array(predicted)
    acc = accuracy_score(y_true, y_pred)
    try:
        if len(candidates) == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    result = {"auroc": float(auc), "accuracy": float(acc), "n": len(y_true)}
    if num_examples > 0:
        examples = []
        for i in range(min(num_examples, len(task_samples))):
            s = task_samples[i]
            pred_idx = int(predicted[i])
            examples.append({
                "mel_path": s.get("mel_path", ""),
                "true_label": int(s["label"]),
                "true_answer": cfg["labels"].get(int(s["label"]), str(s["label"])),
                "pred_label": pred_idx,
                "pred_answer": candidates[pred_idx],
                "candidate_probs": {
                    cand: float(probs[i, j]) for j, cand in enumerate(candidates)
                },
                "dms_text": s.get("dms_text", ""),
                "audio_tokens": [int(x) for x in s["token_ids"]],
                "prompt": prompts[i],
            })
        result["examples"] = examples
    return result


def detect_target_modules(model):
    """Auto-detect LoRA target modules based on model architecture."""
    names = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in ("q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            names.add(last)
    if "q_proj" in names:
        return ["q_proj", "v_proj"]
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if "query" in last or "key" in last:
            names.add(last)
    return list(names)[:2] if names else ["q_proj", "v_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--encoder-type", choices=["htsat", "dual"], default="htsat")
    parser.add_argument("--vq-ckpt", required=True)
    parser.add_argument("--llm", default="aaditya/Llama3-OpenBioLLM-8B")
    parser.add_argument(
        "--train-tasks", nargs="+",
        default=["icbhi_copd", "svd_pathology",
                 "coswara_covid_cough", "coswara_smoker_cough",
                 "b2ai_voice_pathology"],
    )
    parser.add_argument(
        "--eval-tasks", nargs="+",
        default=["icbhi_copd", "svd_pathology",
                 "coswara_covid_cough", "coswara_smoker_cough",
                 "b2ai_voice_pathology",
                 "kauh_obstructive", "copd_severity",
                 "coswara_covid_breathing", "coswara_smoker_breathing",
                 "b2ai_laryngeal_cancer", "b2ai_benign_lesions",
                 "b2ai_laryngeal_dystonia"],
    )
    parser.add_argument("--input-mode", choices=["audio_only", "text_only", "audio_text"],
                        default="audio_text")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", default="",
                        help="Comma-separated LoRA target modules. Empty = auto q/v.")
    parser.add_argument("--modality-dropout", type=float, default=0.0,
                        help="For audio_text training, randomly replace this fraction with "
                             "audio_only/text_only prompts. Evaluation is unchanged.")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoints/rq3_llm/results.json")
    parser.add_argument("--save-examples-task", default="",
                        help="Task key for saving real prompt/prediction examples")
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--save-adapter-dir", default="",
                        help="Directory for saving LoRA adapters per mode")
    parser.add_argument("--load-adapter-dir", default="",
                        help="Load an existing LoRA adapter instead of training.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and only evaluate a loaded/base model.")
    parser.add_argument("--skip-seen-eval", action="store_true",
                        help="Do not evaluate train tasks; useful for target-only sweeps.")
    parser.add_argument("--token-cache-dir", default="checkpoints/rq3_llm/token_cache",
                        help="Cache VAST+VQ token ids per task/encoder/VQ.")
    parser.add_argument("--strict-rq3", action="store_true",
                        help="Use disease/participant-disjoint B2AI training and test-only Coswara targets")
    parser.add_argument("--run-all-modes", action="store_true",
                        help="Run audio_only, text_only, audio_text sequentially for ablation")
    parser.add_argument("--dms-mode", choices=["rich", "minimal", "empty"], default="rich",
                        help="Clinical text strength: rich=original DMS, minimal=Gender/Age only, empty=no DMS.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        print(f"GPU Memory: {mem / 1e9:.1f} GB")

    # Load encoder and VQ
    print("\nLoading encoder and VQ...")
    encoder, vq, K = load_encoder_and_vq(
        args.encoder_ckpt, args.vq_ckpt, device, args.encoder_type,
    )
    print(f"  Codebook size K={K}")

    # Tokenize all tasks
    all_task_data = {}
    all_tasks = sorted(set(args.train_tasks + args.eval_tasks))
    for task_key in all_tasks:
        if task_key not in TASKS:
            print(f"  Skipping unknown task: {task_key}")
            continue
        print(f"Tokenizing {task_key}...")
        token_ids, labels, splits, mel_paths, inline_dms, sample_infos = (
            tokenize_dataset_cached(task_key, encoder, vq, device, args)
        )
        if not token_ids:
            print(f"  No data for {task_key}, skipping")
            continue

        dms_map = load_dms_from_templates(task_key)

        samples = []
        for i, (tids, label, split, mpath, idms, sinfo) in enumerate(
            zip(token_ids, labels, splits, mel_paths, inline_dms, sample_infos)
        ):
            dms = simplify_dms_text(idms or dms_map.get(mpath, ""), args.dms_mode)
            samples.append({
                "task_key": task_key,
                "token_ids": tids,
                "label": label,
                "label_text": TASKS[task_key]["labels"].get(label, str(label)),
                "split": split,
                "dms_text": dms,
                "mel_path": mpath,
                "pid": sinfo.get("pid", ""),
                "participant_id": sinfo.get("participant_id", ""),
                "diagnosis": sinfo.get("diagnosis", ""),
            })
        all_task_data[task_key] = samples
        n_dms = sum(1 for s in samples if s["dms_text"])
        print(f"  {len(samples)} samples ({n_dms} with DMS)")

    # Free encoder/VQ GPU memory
    del encoder, vq
    gc.collect()
    torch.cuda.empty_cache()

    modes = ["audio_text", "audio_only", "text_only"] if args.run_all_modes else [args.input_mode]
    all_results = {}

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode}")
        print(f"{'='*70}")

        # Build train/test splits
        train_samples = []
        test_samples_by_task = {}
        for task_key in args.train_tasks:
            if task_key not in all_task_data:
                continue
            for s in all_task_data[task_key]:
                if s["split"] in ("train", "val"):
                    if (not args.strict_rq3) or keep_strict_train_sample(s, all_task_data):
                        train_samples.append(s)
            if not args.skip_seen_eval:
                test_samples_by_task[task_key] = [
                    s for s in all_task_data[task_key] if s["split"] == "test"
                ]

        for task_key in args.eval_tasks:
            if task_key not in args.train_tasks and task_key in all_task_data:
                if args.strict_rq3:
                    test_samples_by_task[task_key] = [
                        s for s in all_task_data[task_key]
                        if keep_strict_eval_sample(s, task_key)
                    ]
                else:
                    task_test = [
                        s for s in all_task_data[task_key]
                        if s.get("split") == "test"
                    ]
                    test_samples_by_task[task_key] = task_test or all_task_data[task_key]

        print(f"\nTraining: {len(train_samples)} samples from {args.train_tasks}")
        for tk, ts in test_samples_by_task.items():
            zs = "(zero-shot)" if tk not in args.train_tasks else "(seen)"
            print(f"  Eval {tk} {zs}: {len(ts)} samples")

        if not train_samples and not args.eval_only:
            print("  No training data! Skipping mode.")
            continue

        # Load LLM fresh for each mode
        print(f"\nLoading LLM: {args.llm}")
        tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.llm,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Add audio tokens with proper initialization
        audio_tokens = [f"<rv_{i:03d}>" for i in range(K)]
        special = ["<AUD_BOS>", "<AUD_EOS>"]
        all_new_tokens = special + audio_tokens
        old_vocab_size = len(tokenizer)
        print("  Adding audio tokens...", flush=True)
        tokenizer.add_tokens(all_new_tokens, special_tokens=False)

        print("  Resizing token embeddings...", flush=True)
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        print("  Initializing new token embeddings...", flush=True)
        new_embeddings = model.get_input_embeddings().weight.data
        with torch.no_grad():
            new_embeddings[old_vocab_size:len(tokenizer)].normal_(mean=0.0, std=0.02)

        if hasattr(model, "lm_head") and model.lm_head.weight.data.shape[0] == len(tokenizer):
            with torch.no_grad():
                model.lm_head.weight.data[old_vocab_size:len(tokenizer)].normal_(mean=0.0, std=0.02)

        print(f"  Added {len(all_new_tokens)} tokens (vocab: {old_vocab_size} → {len(tokenizer)})")

        if args.load_adapter_dir:
            if PeftModel is None:
                raise RuntimeError("peft is required for --load-adapter-dir")
            adapter_dir = ROOT / args.load_adapter_dir
            print(f"  Loading adapter: {adapter_dir}")
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            trainable_params = []
        # LoRA
        elif get_peft_model is not None:
            targets = (
                [x.strip() for x in args.lora_targets.split(",") if x.strip()]
                if args.lora_targets else detect_target_modules(model)
            )
            print(f"  LoRA targets: {targets}")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                lora_dropout=0.05,
                target_modules=targets,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            print("  WARNING: peft not available, training all parameters")

        model = model.to(device)

        if not args.eval_only:
            # Also make audio token embeddings trainable
            if hasattr(model, "base_model"):
                emb_layer = model.base_model.model.get_input_embeddings()
            else:
                emb_layer = model.get_input_embeddings()
            emb_layer.weight.requires_grad = True

            def mask_old_token_grads(grad):
                grad = grad.clone()
                grad[:old_vocab_size].zero_()
                return grad

            emb_layer.weight.register_hook(mask_old_token_grads)

            # Dataset & Loader
            train_dataset = MultiTaskDataset(
                train_samples, tokenizer, args.max_length, mode,
                args.modality_dropout if mode == "audio_text" else 0.0,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

            total_steps = len(train_loader) * args.epochs // args.grad_accum
            warmup = min(args.warmup_steps, total_steps // 5)

            def get_lr_scale(step):
                if step < warmup:
                    return (step + 1) / max(warmup, 1)
                progress = (step - warmup) / max(total_steps - warmup, 1)
                return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

            # Training loop
            print(f"\nTraining ({mode})...")
            print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}x{args.grad_accum}, "
                  f"LR: {args.lr}, Steps: {total_steps}")

            global_step = 0
            model.train()
            optimizer.zero_grad()

            for epoch in range(args.epochs):
                total_loss = 0
                n_batches = 0
                t0 = time.time()

                for bi, batch in enumerate(train_loader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_t = batch["labels"].to(device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
                        loss = out.loss / args.grad_accum

                    if torch.isnan(loss):
                        print(f"    WARNING: NaN loss at epoch {epoch+1} batch {bi}, skipping")
                        optimizer.zero_grad()
                        continue

                    loss.backward()

                    if (bi + 1) % args.grad_accum == 0 or (bi + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                    total_loss += loss.item() * args.grad_accum
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                elapsed = time.time() - t0
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
                      f"lr={current_lr:.2e} time={elapsed:.0f}s")

            if np.isnan(avg_loss):
                print("  FATAL: Training diverged (NaN). Stopping.")
                break

            if args.save_adapter_dir:
                adapter_dir = ROOT / args.save_adapter_dir / mode
                adapter_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(str(adapter_dir), save_embedding_layers=True)
                    tokenizer.save_pretrained(str(adapter_dir))
                    print(f"  Saved adapter/tokenizer to {adapter_dir}")
        else:
            print(f"\nEval-only ({mode}); skipping training.")

        # Evaluate
        print(f"\nEvaluation ({mode})...")
        model.eval()
        mode_results = {"train_tasks": args.train_tasks, "tasks": {}}

        for task_key, test_samples in test_samples_by_task.items():
            if not test_samples:
                continue
            is_zeroshot = task_key not in args.train_tasks
            tag = "zero-shot" if is_zeroshot else "seen"
            eval_result = evaluate_task(
                model, tokenizer, test_samples, task_key, device,
                args.max_length, mode,
                args.num_examples if task_key == args.save_examples_task else 0,
            )
            eval_result["type"] = tag
            mode_results["tasks"][task_key] = eval_result
            print(f"  {TASKS[task_key]['name']} ({tag}): "
                  f"AUROC={eval_result['auroc']:.4f} Acc={eval_result['accuracy']:.4f} "
                  f"n={eval_result['n']}")

        all_results[mode] = mode_results

        # Cleanup
        del model
        for obj_name in ("optimizer", "scheduler", "train_loader", "train_dataset"):
            try:
                exec(f"del {obj_name}")
            except NameError:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "llm": args.llm,
        "encoder_ckpt": args.encoder_ckpt,
        "encoder_type": args.encoder_type,
        "vq_ckpt": args.vq_ckpt,
        "codebook_size": K,
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "strict_rq3": args.strict_rq3,
        "results": all_results,
    }
    output_path.write_text(json.dumps(final, indent=2))
    print(f"\nAll results saved to {args.output}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mode, res in all_results.items():
        print(f"\n  Mode: {mode}")
        for tk, tr in res["tasks"].items():
            print(f"    {TASKS[tk]['name']:40s} ({tr['type']:9s}): "
                  f"AUROC={tr['auroc']:.4f}  Acc={tr['accuracy']:.4f}")


if __name__ == "__main__":
    main()

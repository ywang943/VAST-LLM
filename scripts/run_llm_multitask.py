"""
RQ3: LLM Multi-Task Instruction Tuning + Zero-Shot Cross-Task.

Extends run_speechgpt_style.py with:
  1. DMS text integration (audio + clinical text)
  2. Multi-task instruction tuning
  3. Zero-shot cross-task evaluation
  4. Input ablation (audio-only vs text-only vs audio+text)

Following RespLLM's experimental protocol.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = TaskType = get_peft_model = None


SR = 16000
WAV_LEN = SR * 8

TASKS = {
    "icbhi_copd": {
        "name": "ICBHI COPD Detection",
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "instruction": (
            "Dataset: ICBHI Respiratory Sound Database. "
            "Task: classify whether this person has COPD based on the audio and clinical information."
        ),
        "labels": {0: "healthy", 1: "copd"},
    },
    "copd_severity": {
        "name": "COPD Severity Classification",
        "mel_root": "data/mel_cache/opera_copd",
        "instruction": (
            "Dataset: Respiratory@TR. "
            "Task: classify the COPD severity level (zero to four) based on the audio and clinical information."
        ),
        "labels": {0: "severity zero", 1: "severity one", 2: "severity two",
                   3: "severity three", 4: "severity four"},
    },
    "kauh_obstructive": {
        "name": "KAUH Obstructive Disease",
        "mel_root": "data/mel_cache/opera_kauh",
        "instruction": (
            "Dataset: KAUH Respiratory Database. "
            "Task: classify whether this person has obstructive airway disease based on the audio and clinical information."
        ),
        "labels": {0: "healthy", 1: "obstructive disease"},
    },
    "svd_pathology": {
        "name": "SVD Voice Pathology",
        "mel_root": "data/mel_cache/svd_full",
        "instruction": (
            "Dataset: Saarbruecken Voice Database. "
            "Task: classify whether this voice recording indicates vocal fold pathology based on the audio and clinical information."
        ),
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder_and_vq(encoder_ckpt, vq_ckpt, device):
    ckpt = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("encoder.", "", 1): v
             for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    vq_data = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    K = vq_data["codebook_size"]
    D = vq_data["D"]
    vq = VectorQuantizer(codebook_size=K, D=D)
    vq.load_state_dict(vq_data["vq_state"], strict=False)
    vq = vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad = False

    return encoder, vq, K


def tokenize_dataset(task_key, encoder, vq, device):
    """Convert mel spectrograms to VQ token IDs."""
    cfg = TASKS[task_key]
    mel_dir = Path(cfg["mel_root"])
    meta = json.loads((mel_dir / "metadata.json").read_text())
    samples = meta.get("samples", [])

    token_ids_list = []
    labels = []
    splits = []
    mel_paths = []

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
                z_cont = encoder(mel)
                vq_out = vq(z_cont)
                ids = vq_out["ids"].squeeze(0).cpu().numpy()
            token_ids_list.append(ids)
            labels.append(int(sample["label"]))
            splits.append(sample.get("split", "train"))
            mel_paths.append(sample["path"])
        except Exception as e:
            continue

    return token_ids_list, labels, splits, mel_paths


def load_dms_texts(task_key):
    """Load DMS text templates for a task."""
    source_map = {
        "icbhi_copd": "icbhi",
        "copd_severity": "icbhi",
        "kauh_obstructive": "kauh",
        "svd_pathology": "svd",
    }
    source = source_map.get(task_key, task_key)
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


def audio_token_text(ids):
    return " ".join(f"<rv_{int(x):05d}>" for x in ids)


def build_prompt(task_key, token_ids, dms_text="", input_mode="audio_text"):
    cfg = TASKS[task_key]
    parts = [
        "You are a medical acoustic diagnostic assistant.\n",
        f"Task: {cfg['instruction']}\n",
    ]

    if input_mode in ("text_only", "audio_text") and dms_text:
        parts.append(f"Clinical information: {dms_text}\n")

    if input_mode in ("audio_only", "audio_text") and token_ids is not None:
        parts.append(f"Audio tokens: {audio_token_text(token_ids)}\n")

    label_options = ", ".join(f'"{v}"' for v in cfg["labels"].values())
    parts.append(f"Answer with one of: {label_options}\n")
    parts.append("Answer:")

    return "".join(parts)


class MultiTaskDataset(Dataset):
    def __init__(self, task_samples, tokenizer, max_length, input_mode="audio_text"):
        self.samples = task_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = input_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        s = self.samples[item]
        prompt = build_prompt(
            s["task_key"], s["token_ids"], s.get("dms_text", ""), self.input_mode
        )
        answer = " " + s["label_text"]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        eos = [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids + eos
        labels = [-100] * len(prompt_ids) + answer_ids + eos

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
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
    scores = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        best_score = -float("inf")
        for cand in candidates:
            cand_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
            full_ids = prompt_ids + cand_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            input_tensor = torch.tensor([full_ids], device=device)
            with torch.no_grad():
                out = model(input_tensor)
                logits = out.logits[0]
            log_probs = F.log_softmax(logits, dim=-1)
            score = 0.0
            for i, tid in enumerate(cand_ids):
                pos = len(prompt_ids) - 1 + i
                if pos < logits.shape[0]:
                    score += log_probs[pos, tid].item()
            if score > best_score:
                best_score = score
                best_idx = candidates.index(cand)
        scores.append(best_idx)
    return scores


def evaluate_task(model, tokenizer, task_samples, task_key, device, max_length, input_mode):
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

    predicted = candidate_scores(model, tokenizer, prompts, candidates, device, max_length)

    y_true = np.array(true_labels)
    y_pred = np.array(predicted)
    acc = accuracy_score(y_true, y_pred)
    try:
        if len(candidates) == 2:
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = roc_auc_score(y_true, np.eye(len(candidates))[y_pred],
                                multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {"auroc": float(auc), "accuracy": float(acc), "n": len(y_true)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--vq-ckpt", required=True)
    parser.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-tasks", nargs="+",
                        default=["icbhi_copd", "svd_pathology"])
    parser.add_argument("--eval-tasks", nargs="+",
                        default=["icbhi_copd", "svd_pathology", "kauh_obstructive", "copd_severity"])
    parser.add_argument("--input-mode", choices=["audio_only", "text_only", "audio_text"],
                        default="audio_text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoints/llm_multitask_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Load encoder and VQ
    print("Loading encoder and VQ...")
    encoder, vq, K = load_encoder_and_vq(args.encoder_ckpt, args.vq_ckpt, device)

    # Tokenize all tasks
    all_task_data = {}
    for task_key in set(args.train_tasks + args.eval_tasks):
        if task_key not in TASKS:
            print(f"  Skipping unknown task: {task_key}")
            continue
        print(f"Tokenizing {task_key}...")
        token_ids, labels, splits, mel_paths = tokenize_dataset(
            task_key, encoder, vq, device
        )
        dms_map = load_dms_texts(task_key)

        samples = []
        for i, (tids, label, split, mpath) in enumerate(
            zip(token_ids, labels, splits, mel_paths)
        ):
            samples.append({
                "task_key": task_key,
                "token_ids": tids,
                "label": label,
                "label_text": TASKS[task_key]["labels"][label],
                "split": split,
                "dms_text": dms_map.get(mpath, ""),
                "mel_path": mpath,
            })
        all_task_data[task_key] = samples
        print(f"  {len(samples)} samples ({sum(1 for s in samples if s['dms_text'])} with DMS)")

    # Build train/test splits
    train_samples = []
    test_samples_by_task = {}
    for task_key in args.train_tasks:
        if task_key not in all_task_data:
            continue
        for s in all_task_data[task_key]:
            if s["split"] in ("train", "val"):
                train_samples.append(s)
        test_samples_by_task[task_key] = [
            s for s in all_task_data[task_key] if s["split"] == "test"
        ]

    # Zero-shot tasks (eval only, not in training)
    for task_key in args.eval_tasks:
        if task_key not in args.train_tasks and task_key in all_task_data:
            test_samples_by_task[task_key] = all_task_data[task_key]

    print(f"\nTraining: {len(train_samples)} samples from {args.train_tasks}")
    for tk, ts in test_samples_by_task.items():
        zs = "(zero-shot)" if tk not in args.train_tasks else "(seen)"
        print(f"  Eval {tk} {zs}: {len(ts)} samples")

    # Load LLM
    print(f"\nLoading LLM: {args.llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.float16, trust_remote_code=True
    )

    # Add audio tokens
    tokens = ["<AUD_BOS>", "<AUD_EOS>"] + [f"<rv_{i:05d}>" for i in range(K)]
    old_vocab = len(tokenizer)
    tokenizer.add_tokens(tokens, special_tokens=False)
    model.resize_token_embeddings(len(tokenizer))

    # LoRA
    if get_peft_model is not None:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)

    # Train
    train_dataset = MultiTaskDataset(
        train_samples, tokenizer, args.max_length, args.input_mode
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    print(f"\n{'='*60}")
    print(f"Training ({args.input_mode} mode)...")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluation ({args.input_mode} mode)")
    print(f"{'='*60}")

    model.eval()
    results = {
        "input_mode": args.input_mode,
        "train_tasks": args.train_tasks,
        "seed": args.seed,
        "tasks": {},
    }

    for task_key, test_samples in test_samples_by_task.items():
        if not test_samples:
            continue
        is_zeroshot = task_key not in args.train_tasks
        tag = "zero-shot" if is_zeroshot else "seen"
        eval_result = evaluate_task(
            model, tokenizer, test_samples, task_key, device,
            args.max_length, args.input_mode
        )
        eval_result["type"] = tag
        results["tasks"][task_key] = eval_result
        print(f"  {TASKS[task_key]['name']} ({tag}): "
              f"AUROC={eval_result['auroc']:.4f} Acc={eval_result['accuracy']:.4f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

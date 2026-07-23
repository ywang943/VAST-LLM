#!/usr/bin/env python3
"""RespLLM-style baseline for the new S1-S7 table.

Local reimplementation based on the released RespLLM code:
  OPERA-COLA audio encoder -> linear aligner -> LLM input embeddings
  prompt/context text embeddings + continuous audio patch embeddings
  LLM hidden states over audio positions -> task-specific classifier heads.

This is used because the RespLLM repository does not provide a ready checkpoint.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder


TASKS = {
    "icbhi_copd": {
        "name": "ICBHI COPD Detection",
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "dms_source": "icbhi",
        "instruction": "Classify COPD versus healthy from lung sounds and clinical information.",
        "labels": {0: "healthy", 1: "copd"},
    },
    "copd_severity": {
        "name": "COPD Severity",
        "mel_root": "data/mel_cache/opera_copd",
        "dms_source": "icbhi",
        "instruction": "Classify the COPD severity level from lung sounds.",
        "labels": {0: "severity zero", 1: "severity one", 2: "severity two", 3: "severity three", 4: "severity four"},
    },
    "coswara_covid_exhale": {
        "name": "Coswara COVID (Exhalation)",
        "mel_root": "data/mel_cache/coswara_covid_exhale",
        "dms_source": "coswara_covid_breathing",
        "instruction": "Classify COVID positive versus non-COVID from exhalation audio and clinical information.",
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "coswara_covid_cough": {
        "name": "Coswara COVID (Cough)",
        "mel_root": "data/mel_cache/coswara_covid_cough",
        "dms_source": "coswara_covid_cough",
        "instruction": "Classify COVID positive versus non-COVID from cough audio and clinical information.",
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "coswara_smoker_cough": {
        "name": "Coswara Smoker (Cough)",
        "mel_root": "data/mel_cache/coswara_smoker_cough",
        "dms_source": "coswara_smoker_cough",
        "instruction": "Classify smoker versus non-smoker from cough audio and clinical information.",
        "labels": {0: "non smoker", 1: "smoker"},
    },
    "svd_pathology": {
        "name": "SVD Voice Pathology",
        "mel_root": "data/mel_cache/svd_full",
        "dms_source": "svd",
        "instruction": "Classify healthy voice versus voice pathology from vowel and sentence recordings.",
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
    "b2ai_voice_pathology": {
        "name": "Bridge2AI Voice Pathology",
        "mel_root": "data/mel_cache/b2ai_voice_pathology",
        "dms_source": "b2ai",
        "instruction": "Classify healthy voice versus voice pathology from sustained vowel audio and clinical information.",
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
    "b2ai_laryngeal_cancer": {
        "name": "Bridge2AI Laryngeal Cancer",
        "mel_root": "data/mel_cache/b2ai_laryngeal_cancer",
        "dms_source": "b2ai",
        "instruction": "Classify whether this person has laryngeal cancer or precancerous lesions from sustained vowel audio and clinical information.",
        "labels": {0: "healthy voice", 1: "laryngeal cancer"},
    },
    "b2ai_benign_lesions": {
        "name": "Bridge2AI Benign Vocal Fold Lesions",
        "mel_root": "data/mel_cache/b2ai_benign_lesions",
        "dms_source": "b2ai",
        "instruction": "Classify whether this person has benign vocal fold lesions from sustained vowel audio and clinical information.",
        "labels": {0: "healthy voice", 1: "benign vocal fold lesions"},
    },
    "b2ai_laryngeal_dystonia": {
        "name": "Bridge2AI Laryngeal Dystonia",
        "mel_root": "data/mel_cache/b2ai_laryngeal_dystonia",
        "dms_source": "b2ai",
        "instruction": "Classify whether this person has laryngeal dystonia or spasmodic dysphonia from sustained vowel audio and clinical information.",
        "labels": {0: "healthy voice", 1: "laryngeal dystonia"},
    },
    "coswara_covid_breathing": {
        "name": "Coswara COVID (Deep Breathing)",
        "mel_root": "data/mel_cache/coswara_covid_breathing",
        "dms_source": "coswara_covid_breathing",
        "instruction": "Classify COVID positive versus non-COVID from deep breathing audio and clinical information.",
        "labels": {0: "no covid", 1: "covid positive"},
    },
    "coswara_smoker_breathing": {
        "name": "Coswara Smoker (Deep Breathing)",
        "mel_root": "data/mel_cache/coswara_smoker_breathing",
        "dms_source": "coswara_smoker_cough",
        "instruction": "Classify smoker versus non-smoker from deep breathing audio and clinical information.",
        "labels": {0: "non smoker", 1: "smoker"},
    },
}

DEFAULT_TRAIN_TASKS = [
    "icbhi_copd",
    "copd_severity",
    "coswara_covid_exhale",
    "coswara_covid_cough",
    "coswara_smoker_cough",
    "svd_pathology",
    "b2ai_voice_pathology",
]

DEFAULT_EVAL_TASKS = [
    "b2ai_laryngeal_cancer",
    "b2ai_benign_lesions",
    "b2ai_laryngeal_dystonia",
    "coswara_covid_breathing",
    "coswara_smoker_breathing",
]

DEFAULT_EVAL_HEAD_MAP = {
    "b2ai_laryngeal_cancer": "b2ai_voice_pathology",
    "b2ai_benign_lesions": "b2ai_voice_pathology",
    "b2ai_laryngeal_dystonia": "b2ai_voice_pathology",
    "coswara_covid_breathing": "coswara_covid_exhale",
    "coswara_smoker_breathing": "coswara_smoker_cough",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_opera_encoder(device):
    ckpt_path = ROOT / "opera_src/cks/model/encoder-operaCT.ckpt"
    enc = build_htsat_encoder(ckpt_path=str(ckpt_path), freeze_backbone=True, use_csaf=False)
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device).eval()


def dms_map(task_key):
    src = TASKS[task_key].get("dms_source", task_key)
    path = ROOT / "data/dms_templates" / f"{src}_dms.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {x.get("mel_path", ""): x.get("dms_text", "") for x in raw if x.get("mel_path")}


@torch.no_grad()
def build_samples(task_keys, encoder, device, batch_size, use_dms=True):
    samples = []
    for task_key in task_keys:
        cfg = TASKS[task_key]
        root = ROOT / cfg["mel_root"]
        raw = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        meta = raw.get("samples", raw if isinstance(raw, list) else [])
        dm = dms_map(task_key)
        print(f"Tokenizing continuous OPERA features: {task_key} n={len(meta)}")
        batch_mels, batch_items = [], []
        for item in meta:
            if "label" not in item:
                continue
            path = root / item["path"]
            if not path.exists():
                continue
            batch_mels.append(torch.load(str(path), map_location="cpu"))
            batch_items.append(item)
            if len(batch_mels) == batch_size:
                _flush_batch(samples, task_key, batch_mels, batch_items, encoder, device, dm, use_dms)
                batch_mels, batch_items = [], []
        if batch_mels:
            _flush_batch(samples, task_key, batch_mels, batch_items, encoder, device, dm, use_dms)
    return samples


@torch.no_grad()
def _flush_batch(samples, task_key, batch_mels, batch_items, encoder, device, dm, use_dms=True):
    mel = torch.stack(batch_mels).to(device)
    z = encoder(mel).cpu().to(torch.float16)
    for i, item in enumerate(batch_items):
        dms = (item.get("dms_text") or dm.get(item.get("path", ""), "")) if use_dms else ""
        if dms:
            text = f"Task: {TASKS[task_key]['instruction']}\nClinical information: {dms}\nAnswer:"
        else:
            text = f"Task: {TASKS[task_key]['instruction']}\nAnswer:"
        samples.append({
            "task_key": task_key,
            "z": z[i],
            "label": int(item["label"]),
            "split": item.get("split", "train"),
            "text": text,
        })


class FeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    return {
        "z": torch.stack([b["z"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "task_keys": [b["task_key"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


class RespLLMStyle(nn.Module):
    def __init__(self, llm, tokenizer, hidden_size, task_classes, d_ff=32):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.aligner = nn.Linear(768, hidden_size)
        self.d_ff = d_ff
        self.heads = nn.ModuleDict({
            k: nn.Linear(64 * d_ff, n) for k, n in task_classes.items()
        })

    def forward(self, z, texts, task_keys):
        toks = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=192)
        toks = {k: v.to(z.device) for k, v in toks.items()}
        text_emb = self.llm.get_input_embeddings()(toks["input_ids"])
        audio_emb = self.aligner(z.float()).to(text_emb.dtype)
        embeds = torch.cat([text_emb, audio_emb], dim=1)
        mask = torch.cat([
            toks["attention_mask"],
            torch.ones((z.size(0), z.size(1)), device=z.device, dtype=toks["attention_mask"].dtype),
        ], dim=1)
        out = self.llm(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        audio_hidden = out[:, -64:, :self.d_ff].float().reshape(z.size(0), -1)
        logits = []
        for i, tk in enumerate(task_keys):
            logits.append(self.heads[tk](audio_hidden[i:i + 1]))
        return logits


def evaluate(model, tokenizer, samples, batch_size, device, head_map=None):
    loader = DataLoader(FeatureDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collate)
    by_task = {k: {"y": [], "probs": [], "pred": []} for k in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            mapped_task_keys = [head_map.get(tk, tk) for tk in batch["task_keys"]] if head_map else batch["task_keys"]
            logits_list = model(z, batch["texts"], mapped_task_keys)
            for logit, y, tk in zip(logits_list, batch["labels"], batch["task_keys"]):
                probs = F.softmax(logit.float(), dim=-1).cpu().numpy()[0]
                by_task[tk]["y"].append(int(y))
                by_task[tk]["probs"].append(probs)
                by_task[tk]["pred"].append(int(probs.argmax()))
    results = {}
    for tk, obj in by_task.items():
        if not obj["y"]:
            continue
        y = np.array(obj["y"])
        probs = np.stack(obj["probs"])
        pred = np.array(obj["pred"])
        try:
            if probs.shape[1] == 2:
                auc = roc_auc_score(y, probs[:, 1])
            else:
                auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        results[tk] = {"auroc": float(auc), "accuracy": float(accuracy_score(y, pred)), "n": int(len(y))}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="checkpoints/respllm_style/new_table_rq1.json")
    p.add_argument("--train-tasks", nargs="+", default=DEFAULT_TRAIN_TASKS, choices=list(TASKS.keys()))
    p.add_argument("--eval-tasks", nargs="+", default=None, choices=list(TASKS.keys()))
    p.add_argument("--map-target-heads", action="store_true",
                   help="Evaluate target tasks through semantically matched source-task heads.")
    p.add_argument("--no-dms", action="store_true",
                   help="Remove clinical/DMS text from prompts for audio-only evaluation.")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_opera_encoder(device)
    eval_tasks = args.eval_tasks or args.train_tasks
    all_keys = sorted(set(args.train_tasks + eval_tasks))
    all_samples = build_samples(all_keys, encoder, device, batch_size=32, use_dms=not args.no_dms)
    del encoder
    torch.cuda.empty_cache()

    train_set = set(args.train_tasks)
    eval_set = set(eval_tasks)
    train = [s for s in all_samples if s["task_key"] in train_set and s["split"] in ("train", "val")]
    test = [s for s in all_samples if s["task_key"] in eval_set and s["split"] == "test"]
    print(f"Train={len(train)} Test={len(test)}")

    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModel.from_pretrained(args.llm, torch_dtype=torch.bfloat16).to(device)
    llm = get_peft_model(llm, LoraConfig(r=args.lora_rank, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
    hidden = llm.config.hidden_size
    if args.map_target_heads:
        all_keys = sorted(set(all_keys + [DEFAULT_EVAL_HEAD_MAP[k] for k in eval_tasks if k in DEFAULT_EVAL_HEAD_MAP]))
    task_classes = {k: len(TASKS[k]["labels"]) for k in all_keys}
    model = RespLLMStyle(llm, tokenizer, hidden, task_classes).to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    loader = DataLoader(FeatureDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    steps = args.epochs * len(loader)
    print(f"Training RespLLM-style: epochs={args.epochs} steps={steps}")
    model.train()
    for ep in range(args.epochs):
        total = 0.0
        n = 0
        optim.zero_grad()
        for i, batch in enumerate(loader):
            z = batch["z"].to(device)
            labels = batch["labels"].to(device)
            logits_list = model(z, batch["texts"], batch["task_keys"])
            loss = 0.0
            for logit, y in zip(logits_list, labels):
                loss = loss + F.cross_entropy(logit, y.view(1))
            loss = loss / len(logits_list)
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                optim.step()
                optim.zero_grad()
            total += float(loss)
            n += 1
        print(f"  epoch {ep + 1}/{args.epochs}: loss={total / max(n, 1):.4f}")

    head_map = DEFAULT_EVAL_HEAD_MAP if args.map_target_heads else None
    results = evaluate(model, tokenizer, test, args.batch_size, device, head_map=head_map)
    for tk, r in results.items():
        print(f"{TASKS[tk]['name']}: AUROC={r['auroc']:.4f} Acc={r['accuracy']:.4f} n={r['n']}")
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "head_map": head_map or {}}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

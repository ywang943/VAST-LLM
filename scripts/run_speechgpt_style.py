"""
SpeechGPT-style downstream evaluation for RespVoice discrete tokens.

This is the LLM route we actually want:
  frozen dual-input RespVoice encoder -> VQ IDs -> added LLM vocabulary tokens
  -> instruction prompt -> diagnosis label.

The script can:
  1. train/load a VQ codebook on frozen encoder outputs;
  2. cache downstream audio token IDs;
  3. expand a causal LLM vocabulary with <rv_XXXXX> audio tokens;
  4. fine-tune with LoRA + trainable audio-token embeddings;
  5. evaluate by candidate-answer log-likelihood, not brittle first-token matching.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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

from respvoice.dual_input_encoder import build_dual_input_encoder
from respvoice.vq import VectorQuantizer
from scripts.run_dual_lejepa_pretrain import collect_datasets, collate_dual

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


SR = 16000
WAV_LEN = SR * 8

TASKS = {
    "icbhi_copd": {
        "name": "ICBHI Healthy-vs-COPD",
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "wav_root": "data/wav_cache/opera_icbhi_disease",
        "instruction": "Classify the respiratory recording as healthy or COPD.",
        "labels": {0: "healthy", 1: "copd"},
    },
    "copd_severity": {
        "name": "Respiratory@TR COPD Severity",
        "mel_root": "data/mel_cache/opera_copd",
        "wav_root": "data/wav_cache/opera_copd",
        "instruction": "Classify the COPD severity level from zero to four.",
        "labels": {
            0: "severity zero",
            1: "severity one",
            2: "severity two",
            3: "severity three",
            4: "severity four",
        },
    },
    "kauh_obstructive": {
        "name": "KAUH Obstructive Disease",
        "mel_root": "data/mel_cache/opera_kauh",
        "wav_root": "data/wav_cache/opera_kauh",
        "instruction": "Classify the respiratory recording as healthy or obstructive disease.",
        "labels": {0: "healthy", 1: "obstructive disease"},
    },
    "svd_pathology": {
        "name": "SVD Voice Pathology",
        "mel_root": "data/mel_cache/svd_full",
        "wav_root": "data/wav_cache/svd_full",
        "instruction": "Classify the voice recording as healthy voice or voice pathology.",
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_dual_input_encoder(
        ckpt_path=None,
        freeze_backbone=True,
        freeze_cnn=True,
        use_csaf=True,
    )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"  encoder loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


@torch.no_grad()
def train_or_load_vq(args, encoder, device):
    vq_path = Path(args.vq_ckpt)
    if vq_path.exists() and not args.retrain_vq:
        data = torch.load(vq_path, map_location="cpu", weights_only=False)
        vq = VectorQuantizer(codebook_size=data["codebook_size"], D=data["D"])
        vq.load_state_dict(data["vq_state"])
        print(f"  loaded VQ: {vq_path} (K={data['codebook_size']})")
        return vq.to(device).eval(), data

    print(f"  training VQ: K={args.codebook_size}, steps={args.vq_steps}")
    datasets, total = collect_datasets(None)
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = DataLoader(
        combined,
        batch_size=args.vq_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_dual,
    )
    print(f"  VQ data: {total} records, {len(loader)} batches/epoch")

    vq = VectorQuantizer(
        codebook_size=args.codebook_size,
        D=768,
        beta=0.25,
        use_ema=True,
        ema_decay=0.99,
        restart_threshold=1,
        restart_every=1,
        l2_normalize=True,
    ).to(device)

    step = 0
    stats = {"loss": 0.0, "util": 0.0, "perp": 0.0, "n": 0}
    while step < args.vq_steps:
        for batch in loader:
            if step >= args.vq_steps:
                break
            mel = batch["mel"].to(device, non_blocking=True)
            wav = batch["wav"].to(device, non_blocking=True)
            z = encoder(mel, wav)
            out = vq(z)
            step += 1
            stats["loss"] += float(out["loss"])
            stats["util"] += float(out["util"])
            stats["perp"] += float(out["perplexity"])
            stats["n"] += 1
            if step % args.log_every == 0:
                n = max(stats["n"], 1)
                print(
                    f"    VQ step {step}: loss={stats['loss']/n:.4f} "
                    f"util={stats['util']/n:.3f} perp={stats['perp']/n:.0f} "
                    f"restart={out['n_restarted']}"
                )
                stats = {"loss": 0.0, "util": 0.0, "perp": 0.0, "n": 0}

    vq_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vq_state": vq.state_dict(),
        "codebook_size": args.codebook_size,
        "D": 768,
        "steps": args.vq_steps,
        "encoder_checkpoint": args.encoder_ckpt,
    }
    torch.save(payload, vq_path)
    print(f"  saved VQ: {vq_path}")
    return vq.eval(), payload


def load_meta(mel_root):
    meta = json.loads((Path(mel_root) / "metadata.json").read_text())
    return meta.get("samples", meta if isinstance(meta, list) else [])


def load_wav(wav_root, mel_path):
    npy = Path(wav_root) / mel_path.replace(".pt", ".npy")
    wav = np.load(str(npy)).astype(np.float32)
    wav = (wav - wav.mean()) / (wav.std() + 1e-8)
    if len(wav) >= WAV_LEN:
        wav = wav[:WAV_LEN]
    else:
        wav = np.pad(wav, (0, WAV_LEN - len(wav)))
    return torch.from_numpy(wav)


@torch.no_grad()
def tokenize_task(task_key, args, encoder, vq, device):
    cfg = TASKS[task_key]
    cache_dir = Path(args.token_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{task_key}_K{args.codebook_size}_"
        f"{Path(args.encoder_ckpt).stem}_{Path(args.vq_ckpt).stem}.pt"
    )

    if cache_path.exists() and not args.retokenize:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  loaded token cache: {cache_path}")
        return cached

    samples = load_meta(cfg["mel_root"])
    token_ids, labels, splits, subjects = [], [], [], []
    print(f"  tokenizing {task_key}: {len(samples)} samples")
    for i, s in enumerate(samples):
        mel = torch.load(Path(cfg["mel_root"]) / s["path"], map_location="cpu")
        mel = mel.unsqueeze(0).to(device)
        wav = load_wav(cfg["wav_root"], s["path"]).unsqueeze(0).to(device)
        z = encoder(mel, wav)
        ids = vq(z)["ids"].squeeze(0).cpu().long()
        token_ids.append(ids)
        labels.append(int(s.get("label", 0)))
        splits.append(s.get("split", "train"))
        subjects.append(s.get("subject_id", s.get("user", str(i))))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(samples)}")

    cached = {
        "task": task_key,
        "token_ids": torch.stack(token_ids),
        "labels": labels,
        "splits": splits,
        "subjects": subjects,
        "codebook_size": args.codebook_size,
        "encoder_ckpt": args.encoder_ckpt,
    }
    torch.save(cached, cache_path)
    print(f"  saved token cache: {cache_path}")
    return cached


def split_indices(labels, splits, seed):
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx = [i for i, s in enumerate(splits) if s == "val"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]

    if not test_idx:
        train_idx = [i for i, s in enumerate(splits) if s != "test"]
    if not val_idx:
        rng = random.Random(seed)
        rng.shuffle(train_idx)
        n_val = max(1, int(len(train_idx) * 0.15))
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]
    return train_idx, val_idx, test_idx


def audio_token_text(ids):
    body = " ".join(f"<rv_{int(x):05d}>" for x in ids)
    return f"<AUD_BOS> {body} <AUD_EOS>"


def build_prompt(task_key, ids):
    cfg = TASKS[task_key]
    return (
        "You are given discrete respiratory/voice audio tokens produced by a "
        "medical acoustic encoder.\n"
        f"Audio tokens: {audio_token_text(ids)}\n"
        f"Task: {cfg['instruction']}\n"
        "Answer with only the class name.\n"
        "Answer:"
    )


class SpeechGPTDataset(Dataset):
    def __init__(self, token_ids, labels, indices, task_key, tokenizer, max_length):
        self.token_ids = token_ids
        self.labels = labels
        self.indices = indices
        self.task_key = task_key
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_text = TASKS[task_key]["labels"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        prompt = build_prompt(self.task_key, self.token_ids[idx])
        answer = " " + self.label_text[int(self.labels[idx])]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        eos = [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids + eos
        labels = [-100] * len(prompt_ids) + answer_ids + eos

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        pad_id = self.tokenizer.pad_token_id
        pad_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_label": int(self.labels[idx]),
            "sample_index": idx,
        }


def add_audio_tokens(tokenizer, codebook_size):
    tokens = ["<AUD_BOS>", "<AUD_EOS>"] + [
        f"<rv_{i:05d}>" for i in range(codebook_size)
    ]
    old_vocab = len(tokenizer)
    n_added = tokenizer.add_tokens(tokens, special_tokens=False)
    return old_vocab, n_added


def resolve_local_hf_path(model_name: str) -> str:
    """Resolve a HuggingFace repo id to its local cache snapshot path.

    Some transformers versions still call HuggingFace Hub metadata APIs during
    tokenizer loading even with local_files_only=True. Passing the snapshot
    directory avoids network access completely.
    """
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = cache_home / "hub" / ("models--" + model_name.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        candidates = sorted(
            [p for p in snapshots.iterdir() if (p / "config.json").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return model_name


def build_llm(args, codebook_size, device):
    llm_path = resolve_local_hf_path(args.llm_name)
    print(f"  LLM path: {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_vocab, n_added = add_audio_tokens(tokenizer, codebook_size)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if get_peft_model is None:
        raise RuntimeError("peft is required for this script in the pytorch env.")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","),
    )
    model = get_peft_model(model, lora_cfg)

    # Train the added audio-token input embeddings. A gradient hook keeps old
    # text-token rows frozen while allowing the new <rv_XXXXX> rows to learn.
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)

    def mask_old_token_grads(grad):
        grad = grad.clone()
        grad[:old_vocab].zero_()
        return grad

    emb.register_hook(mask_old_token_grads)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LLM vocab: old={old_vocab}, added={n_added}, new={len(tokenizer)}")
    print(f"  LLM trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M")
    return model, tokenizer


def candidate_scores(model, tokenizer, prompts, candidates, device, max_length):
    """Return probability over candidate answer strings for each prompt."""
    all_probs = []
    with torch.no_grad():
        for prompt in prompts:
            scores = []
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            for cand in candidates:
                cand_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
                ids = prompt_ids + cand_ids
                if len(ids) > max_length:
                    ids = ids[-max_length:]
                input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(input_ids=input_ids).logits[0]
                start = len(ids) - len(cand_ids)
                logp = 0.0
                count = 0
                for pos, tok in enumerate(cand_ids, start=start):
                    if pos == 0:
                        continue
                    lp = F.log_softmax(logits[pos - 1], dim=-1)[tok]
                    logp += float(lp)
                    count += 1
                scores.append(logp / max(count, 1))
            all_probs.append(F.softmax(torch.tensor(scores), dim=0).numpy())
    return np.stack(all_probs)


def evaluate(model, tokenizer, token_ids, labels, indices, task_key, device, max_length):
    cfg = TASKS[task_key]
    label_items = sorted(cfg["labels"].items())
    class_ids = [k for k, _ in label_items]
    candidates = [v for _, v in label_items]
    prompts = [build_prompt(task_key, token_ids[i]) for i in indices]
    probs = candidate_scores(model, tokenizer, prompts, candidates, device, max_length)
    y_true = np.array([labels[i] for i in indices])
    pred_pos = probs.argmax(axis=1)
    y_pred = np.array([class_ids[p] for p in pred_pos])
    acc = float(accuracy_score(y_true, y_pred))
    try:
        if len(class_ids) == 2:
            pos = class_ids.index(1) if 1 in class_ids else 1
            auc = float(roc_auc_score(y_true, probs[:, pos]))
        else:
            auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.5
    return {"accuracy": acc, "auroc": auc, "n": len(indices)}


def train_one_task(task_key, cached, args, seed, device):
    seed_everything(seed)
    token_ids = cached["token_ids"]
    labels = cached["labels"]
    splits = cached["splits"]
    train_idx, val_idx, test_idx = split_indices(labels, splits, seed)
    if not test_idx:
        raise RuntimeError(f"{task_key} has no test split in metadata.")

    model, tokenizer = build_llm(args, cached["codebook_size"], device)
    train_ds = SpeechGPTDataset(token_ids, labels, train_idx, task_key, tokenizer, args.max_length)
    val_ds = SpeechGPTDataset(token_ids, labels, val_idx, task_key, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val = float("inf")
    best_state = None
    no_improve = 0

    print(
        f"  split seed={seed}: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(out.loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += float(out.loss)
        train_loss = total_loss / max(len(train_loader), 1)
        val_loss = val_loss / max(len(val_loader), 1)
        print(f"    ep {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if "lora_" in k or "embed_tokens" in k
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"    early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    val_metrics = evaluate(model, tokenizer, token_ids, labels, val_idx, task_key, device, args.max_length)
    test_metrics = evaluate(model, tokenizer, token_ids, labels, test_idx, task_key, device, args.max_length)

    del model
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "best_val_loss": best_val,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", default="checkpoints/dual_lejepa_scratch/dual_lejepa_best.pt")
    p.add_argument("--vq-ckpt", default="checkpoints/vq/speechgpt_vq.pt")
    p.add_argument("--llm-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=["icbhi_copd"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--vq-steps", type=int, default=5000)
    p.add_argument("--vq-batch-size", type=int, default=64)
    p.add_argument("--retrain-vq", action="store_true")
    p.add_argument("--retokenize", action="store_true")
    p.add_argument("--token-cache-dir", default="checkpoints/speechgpt_style/token_cache")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", default="checkpoints/speechgpt_style/results.json")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.tasks = args.tasks[:1]
        args.seeds = args.seeds[:1]
        args.vq_steps = min(args.vq_steps, 2)
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LLM: {args.llm_name}")
    print(f"Tasks: {args.tasks}")

    encoder = load_encoder(args.encoder_ckpt, device)
    vq, vq_info = train_or_load_vq(args, encoder, device)

    results = {
        "method": "SpeechGPT-style RespVoice tokens -> expanded LLM vocabulary",
        "encoder_ckpt": args.encoder_ckpt,
        "vq_ckpt": args.vq_ckpt,
        "llm": args.llm_name,
        "codebook_size": int(vq_info["codebook_size"]),
        "tasks": {},
    }

    for task_key in args.tasks:
        print(f"\n{'=' * 70}\nTask: {TASKS[task_key]['name']}\n{'=' * 70}")
        cached = tokenize_task(task_key, args, encoder, vq, device)
        seed_runs = []
        for seed in args.seeds:
            print(f"\n  Seed {seed}")
            seed_runs.append(train_one_task(task_key, cached, args, seed, device))

        aucs = [r["test"]["auroc"] for r in seed_runs]
        accs = [r["test"]["accuracy"] for r in seed_runs]
        task_result = {
            "name": TASKS[task_key]["name"],
            "n_samples": len(cached["labels"]),
            "test_auroc_mean": round(float(np.mean(aucs)), 4),
            "test_auroc_std": round(float(np.std(aucs)), 4),
            "test_acc_mean": round(float(np.mean(accs)), 4),
            "test_acc_std": round(float(np.std(accs)), 4),
            "runs": seed_runs,
        }
        results["tasks"][task_key] = task_result
        print(
            f"  {task_key}: AUROC={task_result['test_auroc_mean']:.4f} "
            f"+/- {task_result['test_auroc_std']:.4f}, "
            f"ACC={task_result['test_acc_mean']:.4f}"
        )

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  partial saved: {out_path}")

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

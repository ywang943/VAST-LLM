"""
AudioLM-style: hierarchical discrete tokens → decoder Transformer → diagnosis.

Architecture:
  Level 1 (Semantic): CSAF output → VQ → semantic tokens (64 tokens, K=8192)
  Level 2 (Acoustic): Stage-4 output → smaller VQ → acoustic tokens (64 tokens, K=1024)

  Combined: [CLS] [semantic_0..63] [SEP] [acoustic_0..63] [SEP] [task_prompt]
  → Small decoder Transformer → classification

This is simpler than the full AudioLM (no audio generation), adapted for
discriminative health diagnosis using hierarchical discrete representations.

Reference: AudioLM (Borsos et al., 2023)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from respvoice.dual_input_encoder import build_dual_input_encoder
from respvoice.vq import VectorQuantizer


# ── Hierarchical Audio Tokenizer ─────────────────────────────────────────────

class HierarchicalTokenizer(nn.Module):
    """Two-level tokenizer: semantic (CSAF) + acoustic (Stage-4)."""

    def __init__(self, encoder_ckpt, K_semantic=8192, K_acoustic=1024):
        super().__init__()
        ckpt = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v
                 for k, v in ckpt["model_state"].items()
                 if k.startswith("encoder.")}
        self.encoder = build_dual_input_encoder(
            ckpt_path=None, freeze_backbone=True, freeze_cnn=True, use_csaf=True,
        )
        self.encoder.load_state_dict(state, strict=False)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.vq_semantic = VectorQuantizer(codebook_size=K_semantic, D=768,
                                            l2_normalize=True, use_ema=True)
        self.vq_acoustic = VectorQuantizer(codebook_size=K_acoustic, D=768,
                                            l2_normalize=True, use_ema=True)
        self.K_semantic = K_semantic
        self.K_acoustic = K_acoustic

    @torch.no_grad()
    def extract_stages(self, mel, waveform=None):
        """Get CSAF output (semantic) and Stage-4 output (acoustic)."""
        enc = self.encoder
        if waveform is not None:
            mel_img = enc._preprocess(mel)
            cnn_out = enc.hubert_cnn(waveform)
            wav_img = enc.wav_adapter(cnn_out)
            x = enc.fusion(mel_img, wav_img)
        else:
            x = enc._preprocess(mel)

        x = enc.htsat.patch_embed(x)
        if enc.htsat.ape:
            x = x + enc.htsat.absolute_pos_embed
        x = enc.htsat.pos_drop(x)

        x, _ = enc.htsat.layers[0](x)
        e1 = enc.pool1(x)
        x, _ = enc.htsat.layers[1](x)
        e2 = enc.pool2(x)
        x, _ = enc.htsat.layers[2](x)
        e3 = x
        x, _ = enc.htsat.layers[3](x)
        e4 = enc.htsat.norm(x)

        csaf_out = enc.csaf([e1, e2, e3, e4])  # semantic: (B, 64, 768)
        return csaf_out, e4  # semantic, acoustic

    def tokenize(self, mel, waveform=None):
        """Returns (semantic_ids, acoustic_ids), each (B, 64)."""
        semantic, acoustic = self.extract_stages(mel, waveform)
        sem_out = self.vq_semantic(semantic)
        aco_out = self.vq_acoustic(acoustic)
        return sem_out["ids"], aco_out["ids"], sem_out, aco_out

    def train_step(self, mel, waveform=None):
        """Train VQ codebooks, return losses."""
        semantic, acoustic = self.extract_stages(mel, waveform)
        self.vq_semantic.train()
        self.vq_acoustic.train()
        sem_out = self.vq_semantic(semantic)
        aco_out = self.vq_acoustic(acoustic)
        return {
            "loss": sem_out["loss"] + aco_out["loss"],
            "sem_util": sem_out["util"],
            "aco_util": aco_out["util"],
            "sem_perp": sem_out["perplexity"],
            "aco_perp": aco_out["perplexity"],
        }


# ── Decoder Transformer for Classification ───────────────────────────────────

class AudioLMClassifier(nn.Module):
    """
    Small decoder Transformer that takes hierarchical discrete tokens
    and classifies.

    Input: [CLS] [sem_0..63] [SEP] [aco_0..63] → Transformer → [CLS] → MLP → class
    """

    def __init__(self, K_semantic, K_acoustic, d_model=256, n_heads=4,
                 n_layers=4, n_classes=2, max_seq=132, dropout=0.3):
        super().__init__()
        # Separate embeddings for semantic and acoustic tokens
        self.sem_embed = nn.Embedding(K_semantic, d_model)
        self.aco_embed = nn.Embedding(K_acoustic, d_model)
        # Special tokens: CLS, SEP
        self.special_embed = nn.Embedding(3, d_model)  # 0=CLS, 1=SEP, 2=PAD
        self.pos_embed = nn.Embedding(max_seq, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, sem_ids, aco_ids):
        """
        sem_ids: (B, 64) semantic token IDs
        aco_ids: (B, 64) acoustic token IDs
        Returns: (B, n_classes) logits
        """
        B = sem_ids.size(0)
        device = sem_ids.device

        # Build sequence: [CLS] sem_0..63 [SEP] aco_0..63
        cls_emb = self.special_embed(torch.zeros(B, 1, dtype=torch.long, device=device))
        sep_emb = self.special_embed(torch.ones(B, 1, dtype=torch.long, device=device))

        sem_emb = self.sem_embed(sem_ids)    # (B, 64, D)
        aco_emb = self.aco_embed(aco_ids)    # (B, 64, D)

        # [CLS, sem*64, SEP, aco*64] = 130 tokens
        seq = torch.cat([cls_emb, sem_emb, sep_emb, aco_emb], dim=1)  # (B, 130, D)
        positions = torch.arange(seq.size(1), device=device).unsqueeze(0)
        seq = seq + self.pos_embed(positions)

        out = self.transformer(seq)          # (B, 130, D)
        cls_out = out[:, 0]                  # (B, D) — CLS token
        return self.head(cls_out)            # (B, n_classes)


# ── Dataset ──────────────────────────────────────────────────────────────────

class HierarchicalTokenDataset(Dataset):
    def __init__(self, sem_ids, aco_ids, labels):
        self.sem_ids = sem_ids
        self.aco_ids = aco_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sem_ids": self.sem_ids[idx],
            "aco_ids": self.aco_ids[idx],
            "label": self.labels[idx],
        }


# ── Pre-tokenize ─────────────────────────────────────────────────────────────

@torch.no_grad()
def tokenize_all(tokenizer, mel_root, wav_root, device):
    meta = json.loads((Path(mel_root) / "metadata.json").read_text())
    samples = meta.get("samples", [])

    all_sem, all_aco, all_labels, all_splits = [], [], [], []
    for i, s in enumerate(samples):
        mel = torch.load(Path(mel_root) / s["path"], map_location="cpu")
        mel = mel.unsqueeze(0).to(device)

        wav = None
        if wav_root:
            npy = s["path"].replace(".pt", ".npy")
            npy_path = Path(wav_root) / npy
            if npy_path.exists():
                w = np.load(str(npy_path)).astype(np.float32)
                w = (w - w.mean()) / (w.std() + 1e-8)
                target = 128000
                w = w[:target] if len(w) >= target else np.pad(w, (0, target - len(w)))
                wav = torch.from_numpy(w).unsqueeze(0).to(device)

        sem_ids, aco_ids, _, _ = tokenizer.tokenize(mel, wav)
        all_sem.append(sem_ids.squeeze(0).cpu())
        all_aco.append(aco_ids.squeeze(0).cpu())
        all_labels.append(s.get("label", 0))
        all_splits.append(s.get("split", "train"))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(samples)}")

    return torch.stack(all_sem), torch.stack(all_aco), all_labels, all_splits


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--encoder-ckpt", required=True)
    pa.add_argument("--task", required=True,
                    choices=["icbhi_copd", "copd_severity", "kauh_obstructive", "svd_pathology"])
    pa.add_argument("--mel-root", required=True)
    pa.add_argument("--wav-root", default=None)
    pa.add_argument("--K-semantic", type=int, default=8192)
    pa.add_argument("--K-acoustic", type=int, default=1024)
    pa.add_argument("--d-model", type=int, default=256)
    pa.add_argument("--n-layers", type=int, default=4)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=10)
    pa.add_argument("--vq-train-steps", type=int, default=3000)
    pa.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    pa.add_argument("--out", required=True)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = {"icbhi_copd": 2, "copd_severity": 5,
                 "kauh_obstructive": 2, "svd_pathology": 2}[args.task]

    # 1. Build hierarchical tokenizer
    print("Building hierarchical tokenizer...")
    tokenizer = HierarchicalTokenizer(
        args.encoder_ckpt, args.K_semantic, args.K_acoustic
    ).to(device)

    # 2. Train VQ codebooks
    print(f"\nTraining VQ codebooks ({args.vq_train_steps} steps)...")
    from scripts.run_dual_lejepa_pretrain import collect_datasets, collate_dual
    from torch.utils.data import ConcatDataset
    datasets, total = collect_datasets(None)
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    vq_loader = DataLoader(combined, batch_size=32, shuffle=True,
                           drop_last=True, num_workers=4, collate_fn=collate_dual)

    step = 0
    while step < args.vq_train_steps:
        for batch in vq_loader:
            if step >= args.vq_train_steps:
                break
            mel = batch["mel"].to(device)
            wav = batch["wav"].to(device) if batch["wav"] is not None else None
            out = tokenizer.train_step(mel, wav)
            step += 1
            if step % 500 == 0:
                print(f"  Step {step}: sem_util={out['sem_util']:.3f} "
                      f"aco_util={out['aco_util']:.3f} "
                      f"sem_perp={out['sem_perp']:.0f} aco_perp={out['aco_perp']:.0f}")

    # 3. Tokenize downstream data
    print(f"\nTokenizing {args.task}...")
    sem_ids, aco_ids, labels, splits = tokenize_all(
        tokenizer, args.mel_root, args.wav_root, device)
    print(f"  Tokenized: {len(labels)} samples")
    del tokenizer
    torch.cuda.empty_cache()

    # 4. Train classifier for each seed
    train_idx = [i for i, s in enumerate(splits) if s != "test"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]

    seed_aucs = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Train/val split
        idx_perm = train_idx.copy()
        random.shuffle(idx_perm)
        n_val = max(1, int(len(idx_perm) * 0.15))
        val_idx = idx_perm[:n_val]
        tr_idx = idx_perm[n_val:]

        train_ds = HierarchicalTokenDataset(
            sem_ids[tr_idx], aco_ids[tr_idx],
            torch.tensor([labels[i] for i in tr_idx]))
        val_ds = HierarchicalTokenDataset(
            sem_ids[val_idx], aco_ids[val_idx],
            torch.tensor([labels[i] for i in val_idx]))
        test_ds = HierarchicalTokenDataset(
            sem_ids[test_idx], aco_ids[test_idx],
            torch.tensor([labels[i] for i in test_idx]))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # Build classifier
        clf = AudioLMClassifier(
            args.K_semantic, args.K_acoustic,
            d_model=args.d_model, n_layers=args.n_layers,
            n_classes=n_classes, dropout=0.3,
        ).to(device)

        if seed == args.seeds[0]:
            n_params = sum(p.numel() for p in clf.parameters())
            print(f"\n  Classifier: {n_params/1e6:.2f}M params")

        # Class weights
        label_tensor = torch.tensor([labels[i] for i in tr_idx])
        counts = torch.bincount(label_tensor, minlength=n_classes).float()
        weights = (counts.sum() / (counts.clamp_min(1) * n_classes)).to(device)

        optimizer = AdamW(clf.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_auc, best_state, no_improve = -1.0, None, 0
        for epoch in range(args.epochs):
            clf.train()
            for batch in train_loader:
                logits = clf(batch["sem_ids"].to(device), batch["aco_ids"].to(device))
                loss = F.cross_entropy(logits, batch["label"].to(device), weight=weights)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            clf.eval()
            vp, vl = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = clf(batch["sem_ids"].to(device), batch["aco_ids"].to(device))
                    probs = F.softmax(logits, dim=1).cpu()
                    vp.append(probs)
                    vl.append(batch["label"])
            vp, vl = torch.cat(vp).numpy(), torch.cat(vl).numpy()
            try:
                val_auc = roc_auc_score(vl, vp[:, 1]) if n_classes == 2 else \
                    roc_auc_score(vl, vp, multi_class="ovr", average="macro")
            except:
                val_auc = 0.5

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                break

        # Test
        if best_state:
            clf.load_state_dict(best_state)
        clf.eval()
        tp, tl = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = clf(batch["sem_ids"].to(device), batch["aco_ids"].to(device))
                probs = F.softmax(logits, dim=1).cpu()
                tp.append(probs)
                tl.append(batch["label"])
        tp, tl = torch.cat(tp).numpy(), torch.cat(tl).numpy()
        try:
            test_auc = roc_auc_score(tl, tp[:, 1]) if n_classes == 2 else \
                roc_auc_score(tl, tp, multi_class="ovr", average="macro")
        except:
            test_auc = 0.5

        print(f"  seed {seed}: AUROC={test_auc:.4f}")
        seed_aucs.append(test_auc)
        del clf

    m, s = float(np.mean(seed_aucs)), float(np.std(seed_aucs))
    print(f"\n{'='*50}")
    print(f"  AudioLM-style: {args.task}")
    print(f"  AUROC: {m:.4f} ± {s:.4f}")
    print(f"{'='*50}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "method": "AudioLM-style (hierarchical tokens → decoder Transformer)",
        "task": args.task,
        "K_semantic": args.K_semantic,
        "K_acoustic": args.K_acoustic,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "auroc_mean": round(m, 4),
        "auroc_std": round(s, 4),
        "per_seed": [round(a, 4) for a in seed_aucs],
    }, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

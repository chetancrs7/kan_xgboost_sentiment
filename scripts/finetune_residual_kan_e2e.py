"""End-to-end Residual Bottleneck KAN over an unfrozen DeBERTa encoder.

This is the FULL architecture from the user's diagram, trained end-to-end on
raw tweets -- the thing that can actually make the KAN's accuracy climb
gradually instead of snapping to a frozen-embedding ceiling:

    Tweet tokens
      -> DeBERTa-v3 encoder (last N transformer layers UNFROZEN, rest frozen)
      -> CLS pooling  +  masked-mean pooling  -> concat (1536)
      -> LayerNorm(1536)
      -> Linear 1536 -> 256 -> GELU -> Dropout -> Linear 256 -> 64
      -> LayerNorm(64)                                       = z
      -> z' = z + gamma * KAN_64_64(z)   (gamma learnable)   [x n_blocks]
      -> LayerNorm(64)
      -> KAN_64_classes(z')  -> logits

Standalone by design: it only *imports* the project's KAN spline layer as a
reusable block and reads the raw-text CSV splits. It never modifies existing
code and writes solely to its own --output-dir, so nothing else can break.

Why this can give a gradual, non-plateauing curve (unlike the frozen-embedding
runs): the encoder's top layers keep improving the representation as training
proceeds, so the KAN has evolving nonlinear structure to fit -- gamma has a
reason to grow and val accuracy rises over many optimiser steps rather than
saturating in epoch 1. A step-level val curve is logged to show exactly that.

Usage:
    python scripts/finetune_residual_kan_e2e.py                    # defaults
    python scripts/finetune_residual_kan_e2e.py --train-size 40000 --epochs 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kan.layers import AdaptiveSplineKANLayer  # noqa: E402

from transformers import AutoModel, AutoTokenizer  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.labels[i])


def stratified_sample(df, n, label_col, seed):
    if n <= 0 or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _cls, g in df.groupby(label_col):
        k = min(len(g), max(1, round(len(g) * frac)))
        parts.append(g.sample(n=k, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResidualKANBlock(nn.Module):
    def __init__(self, dim, grid_size, spline_order, gamma_init):
        super().__init__()
        self.kan = AdaptiveSplineKANLayer(dim, dim, grid_size, spline_order)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, z):
        return z + self.gamma * self.kan(z)


class E2EResidualBottleneckKAN(nn.Module):
    def __init__(self, encoder_name, n_classes, *, unfrozen_layers=2,
                 proj_hidden=256, bottleneck=64, grid_size=3, spline_order=3,
                 dropout=0.2, gamma_init=0.1, n_blocks=1):
        super().__init__()
        # transformers v5 loads this checkpoint in fp16 by default; fp16 master
        # weights make AdamW's exp_avg_sq underflow to zero and NaN the encoder on
        # the first optimizer step (same fix as scripts/finetune_deberta.py).
        # Force fp32 master weights and let bf16 autocast handle throughput.
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size

        # Freeze everything, then unfreeze the last `unfrozen_layers` transformer
        # blocks so the top of the encoder keeps adapting to sentiment.
        for p in self.encoder.parameters():
            p.requires_grad = False
        layers = self.encoder.encoder.layer  # DeBERTa-v2 ModuleList
        self.n_unfrozen = min(unfrozen_layers, len(layers))
        for layer in layers[-self.n_unfrozen:]:
            for p in layer.parameters():
                p.requires_grad = True

        cat = hidden * 2  # CLS + masked-mean
        self.in_norm = nn.LayerNorm(cat)
        self.proj = nn.Sequential(
            nn.Linear(cat, proj_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_hidden, bottleneck),
        )
        self.post_proj_norm = nn.LayerNorm(bottleneck)
        self.blocks = nn.ModuleList(
            [ResidualKANBlock(bottleneck, grid_size, spline_order, gamma_init) for _ in range(n_blocks)]
        )
        self.block_norms = nn.ModuleList([nn.LayerNorm(bottleneck) for _ in range(n_blocks)])
        self.classifier = AdaptiveSplineKANLayer(bottleneck, n_classes, grid_size, spline_order)

    @staticmethod
    def _pool(last_hidden, attention_mask):
        cls = last_hidden[:, 0]                                   # [B, H]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-6)
        mean = summed / counts                                   # [B, H]
        return torch.cat([cls, mean], dim=-1)                    # [B, 2H]

    def _encode_pool(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def _head(self, pooled):
        # pykan spline layers are numerically fragile and NOT autocast-safe, so
        # the whole head runs in pure fp32 with autocast disabled.
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            pooled = pooled.float()
            z = self.post_proj_norm(self.proj(self.in_norm(pooled)))
            for blk, norm in zip(self.blocks, self.block_norms):
                z = norm(blk(z))
            return self.classifier(z)

    def forward(self, input_ids, attention_mask, **kw):
        return self._head(self._encode_pool(input_ids, attention_mask))

    @torch.no_grad()
    def seed_grids(self, input_ids, attention_mask):
        """Seed every KAN spline grid from a real batch's activations so the
        splines start centred on the data range (prevents early NaNs)."""
        was_training = self.training
        self.eval()
        pooled = self._encode_pool(input_ids, attention_mask).float()
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            z = self.post_proj_norm(self.proj(self.in_norm(pooled)))
            for blk, norm in zip(self.blocks, self.block_norms):
                blk.kan.adapt_grid(z)
                z = norm(blk(z))
            self.classifier.adapt_grid(z)
        if was_training:
            self.train()


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    correct, n, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        labels = batch["labels"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(ids, mask)
        logits = logits.float()
        loss_sum += crit(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--train-csv", default="data/splits/train.csv")
    ap.add_argument("--val-csv", default="data/splits/val.csv")
    ap.add_argument("--test-csv", default="data/splits/test.csv",
                    help="Held-out test split, evaluated once at the end with the best checkpoint.")
    ap.add_argument("--final-eval-size", type=int, default=0,
                    help="Rows for the final full val/test eval (0 = ALL rows).")
    ap.add_argument("--text-column", default="text", help="raw tweet column (diagram: 'Tweet tokens')")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--output-dir", default="outputs/e2e_residual_kan")
    ap.add_argument("--train-size", type=int, default=40000)
    ap.add_argument("--val-size", type=int, default=4000)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--unfrozen-layers", type=int, default=2)
    ap.add_argument("--proj-hidden", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--grid-size", type=int, default=3)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gamma-init", type=float, default=0.1)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100, help="eval val every N optimiser steps")
    ap.add_argument("--warmup-ratio", type=float, default=0.06,
                    help="Linear LR warmup fraction (stabilises DeBERTa's attention early).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"],
                    help="Encoder mixed precision. 'off' = pure fp32 (stable but heavier).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if (args.amp == "bf16" and device.type == "cuda"
                 and torch.cuda.is_bf16_supported()) else None)
    print(f"[INFO] E2E Residual KAN | encoder={args.encoder} | device={device} | amp={amp_dtype}")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    tr_df = pd.read_csv(args.train_csv, usecols=[args.text_column, args.label_column]).dropna()
    va_df = pd.read_csv(args.val_csv, usecols=[args.text_column, args.label_column]).dropna()
    tr_df = stratified_sample(tr_df, args.train_size, args.label_column, args.seed)
    va_df = stratified_sample(va_df, args.val_size, args.label_column, args.seed)
    n_classes = int(tr_df[args.label_column].nunique())
    print(f"[INFO] train={len(tr_df)} val={len(va_df)} classes={n_classes}")

    collate = make_collate(tok, args.max_length)
    train_loader = DataLoader(
        TweetDataset(tr_df[args.text_column], tr_df[args.label_column]),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        TweetDataset(va_df[args.text_column], va_df[args.label_column]),
        batch_size=64, shuffle=False, collate_fn=collate,
    )

    model = E2EResidualBottleneckKAN(
        args.encoder, n_classes, unfrozen_layers=args.unfrozen_layers,
        proj_hidden=args.proj_hidden, bottleneck=args.bottleneck,
        grid_size=args.grid_size, spline_order=args.spline_order,
        dropout=args.dropout, gamma_init=args.gamma_init, n_blocks=args.n_blocks,
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    print(f"[INFO] trainable params: {sum(p.numel() for p in trainable):,} "
          f"(encoder {sum(p.numel() for p in enc_params):,} in last {model.n_unfrozen} layers + "
          f"head {sum(p.numel() for p in head_params):,})")

    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.encoder_lr},
         {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(s):  # linear warmup -> linear decay (per the proven recipe)
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[INFO] {total_steps} optimiser steps, {warmup_steps} warmup.")
    criterion = nn.CrossEntropyLoss()

    # Seed KAN grids from one real batch so the splines start on the data range.
    seed_batch = next(iter(train_loader))
    model.seed_grids(seed_batch["input_ids"].to(device), seed_batch["attention_mask"].to(device))
    print("[INFO] Seeded KAN spline grids from a real batch.")

    history = {"step": [], "val_acc": [], "val_loss": [], "gamma": [], "train_run_acc": []}
    best_val, best_state = 0.0, copy.deepcopy(model.state_dict())
    step, t0 = 0, time.time()

    # Baseline (step 0) val accuracy before any training -- the start of the curve.
    vl, vacc = evaluate(model, val_loader, device, amp_dtype)
    history["step"].append(0); history["val_acc"].append(vacc)
    history["val_loss"].append(vl); history["gamma"].append(float(model.blocks[0].gamma))
    history["train_run_acc"].append(float("nan"))
    print(f"[step 000000] val_acc={vacc:.4f} val_loss={vl:.4f} (pre-training)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_correct, run_n = 0, 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(ids, mask)
            loss = criterion(logits.float(), labels)
            loss.backward()
            if args.clip_norm > 0:
                nn.utils.clip_grad_norm_(trainable, args.clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_n += labels.size(0)

            if step % args.eval_every == 0:
                vl, vacc = evaluate(model, val_loader, device, amp_dtype)
                gamma = float(model.blocks[0].gamma)
                run_acc = run_correct / max(1, run_n)
                history["step"].append(step); history["val_acc"].append(vacc)
                history["val_loss"].append(vl); history["gamma"].append(gamma)
                history["train_run_acc"].append(run_acc)
                if vacc > best_val:
                    best_val, best_state = vacc, copy.deepcopy(model.state_dict())
                print(f"[ep {epoch} step {step:06d}] train_run_acc={run_acc:.4f} "
                      f"val_acc={vacc:.4f} val_loss={vl:.4f} gamma={gamma:+.3f} "
                      f"({(time.time()-t0)/60:.1f} min)")
                # Incremental save so a multi-hour run's curve survives a crash.
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                (Path(args.output_dir) / "history.json").write_text(
                    json.dumps({"args": vars(args), "best_val_acc": best_val,
                                "history": history}, indent=2), encoding="utf-8")
                model.train()
                run_correct, run_n = 0, 0

        # End-of-epoch eval.
        vl, vacc = evaluate(model, val_loader, device, amp_dtype)
        gamma = float(model.blocks[0].gamma)
        history["step"].append(step); history["val_acc"].append(vacc)
        history["val_loss"].append(vl); history["gamma"].append(gamma)
        history["train_run_acc"].append(float("nan"))
        if vacc > best_val:
            best_val, best_state = vacc, copy.deepcopy(model.state_dict())
        print(f"== epoch {epoch} done | val_acc={vacc:.4f} val_loss={vl:.4f} gamma={gamma:+.3f} ==")

    va = history["val_acc"]

    # Restore the best checkpoint and evaluate on the FULL val + held-out test.
    model.load_state_dict(best_state)
    final = {}
    for name, csv in (("val", args.val_csv), ("test", args.test_csv)):
        df = pd.read_csv(csv, usecols=[args.text_column, args.label_column]).dropna()
        df = stratified_sample(df, args.final_eval_size, args.label_column, args.seed)
        loader = DataLoader(TweetDataset(df[args.text_column], df[args.label_column]),
                            batch_size=64, shuffle=False, collate_fn=collate)
        floss, facc = evaluate(model, loader, device, amp_dtype)
        final[name] = {"n": len(df), "acc": facc, "loss": floss}
        print(f"[FINAL] full {name}: n={len(df):,} acc={facc:.4f} loss={floss:.4f}")

    print("\n" + "=" * 64 + "\nRESULTS\n" + "=" * 64)
    print(f"  val_acc curve: pre-train(step0)={va[0]:.4f}  best={best_val:.4f}")
    print(f"  FINAL full val acc  = {final['val']['acc']:.4f}  (n={final['val']['n']:,})")
    print(f"  FINAL full test acc = {final['test']['acc']:.4f}  (n={final['test']['n']:,})")
    print(f"  gamma:   init={history['gamma'][0]:+.3f} -> final={history['gamma'][-1]:+.3f}")
    print(f"  climb: first->best delta = {best_val - va[0]:+.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "args": vars(args)}, out_dir / "best_model.pt")
    (out_dir / "history.json").write_text(
        json.dumps({"args": vars(args), "best_val_acc": best_val,
                    "final_full_eval": final, "history": history}, indent=2),
        encoding="utf-8",
    )
    # Save a learning-curve plot (best-effort).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["step"], history["val_acc"], marker="o", ms=3, label="val accuracy")
        ax.set_xlabel("optimiser step"); ax.set_ylabel("val accuracy")
        ax.set_title("E2E Residual Bottleneck KAN — val accuracy vs training step")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "learning_curve.png", dpi=150)
        print(f"[INFO] Saved learning curve to {out_dir / 'learning_curve.png'}")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")

    print(f"[INFO] Saved history to {(out_dir / 'history.json').resolve()}")


if __name__ == "__main__":
    main()

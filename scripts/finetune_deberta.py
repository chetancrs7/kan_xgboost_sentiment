"""Fine-tune DeBERTa-v3-base end-to-end for binary sentiment classification.

The frozen-embedding pipeline caps the KAN expert at ~83%; fine-tuning the
encoder on the task typically buys several points. The best checkpoint is
saved with ``save_pretrained``, so it drops straight into the existing
embedding pipeline::

    python scripts/finetune_deberta.py
    python src/embeddings/generate_embeddings.py \
        --model-name models/deberta_finetuned/best \
        --data-dir data/splitsfull --output-dir data/embeddings_finetuned

Sized for a 6 GB GPU: bf16 autocast (DeBERTa-v3 is numerically unstable in
fp16), batch 16 x grad-accum 2. Validation runs on a seeded subsample every
``--eval-every`` optimizer steps and the best checkpoint (by accuracy) is
saved each time it improves, so an interrupted run keeps its progress.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the same encoder presets as the embedding pipeline (conf/embeddings.yaml)
# so --method works identically here.
from src.embeddings.config import load_embedding_methods  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3-base for binary sentiment classification."
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Named encoder preset from conf/embeddings.yaml (e.g. 'bertweet'). "
        "Fills in model-name / text-column / tokenizer defaults; explicit flags "
        "still override. Omit for the default DeBERTa behaviour (unchanged).",
    )
    parser.add_argument("--model-name", default=None,
                        help="HuggingFace model id (default: preset or microsoft/deberta-v3-base).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "splitsfull")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "deberta_finetuned")
    parser.add_argument("--text-column", default=None,
                        help="Text column to fine-tune on (default: preset or clean_text).")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--eval-every", type=int, default=2000,
                        help="Optimizer steps between mid-epoch validations.")
    parser.add_argument("--eval-samples", type=int, default=20000,
                        help="Seeded val subsample size for mid-epoch evals (full val at the end).")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Subsample the train split (smoke tests).")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing if VRAM is tight.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] Split file not found: {path.resolve()}")
    df = pd.read_csv(path, usecols=[text_column, label_column])
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""].reset_index(drop=True)
    if before - len(df):
        logger.warning("Dropped %d rows with null/empty text from %s.", before - len(df), path.name)
    return df


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def make_collate(tokenizer, max_length: int):
    def collate(batch: list[tuple[str, int]]):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def linear_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> dict[str, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for batch in tqdm(loader, desc="Eval", unit="batch", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            out = model(**batch)
        n = batch["labels"].size(0)
        total_loss += out.loss.item() * n
        correct += (out.logits.argmax(dim=-1) == batch["labels"]).sum().item()
        count += n
    model.train()
    return {"loss": total_loss / count, "accuracy": correct / count}


def save_checkpoint(model, tokenizer, path: Path, retries: int = 5, wait: float = 4.0) -> None:
    """Save the model + tokenizer, retrying transient Windows file locks.

    On Windows, OneDrive sync or Defender can momentarily lock a freshly written
    ``.safetensors`` file, making ``save_pretrained`` raise "Access is denied"
    (os error 5). Rather than let that kill a long run, retry a few times with a
    short backoff. A *persistent* denial (e.g. Controlled Folder Access) will
    still raise after the retries — in that case save to a non-Documents dir.
    """
    path.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            return
        except (OSError, RuntimeError) as exc:
            if attempt == retries:
                logger.error(
                    "Checkpoint save to %s failed after %d attempts (%s). If this "
                    "is a Windows lock, save outside Documents/OneDrive via "
                    "--output-dir, or pause OneDrive / add a Defender exclusion.",
                    path, retries, exc,
                )
                raise
            logger.warning(
                "Checkpoint save failed (attempt %d/%d: %s); retrying in %.0fs ...",
                attempt, retries, exc, wait,
            )
            time.sleep(wait)


def resolve_encoder(args: argparse.Namespace) -> dict:
    """Merge a named ``--method`` preset with explicit flags (flags win)."""
    preset: dict = {}
    if args.method is not None:
        preset = load_embedding_methods().get(args.method) or {}
        if not preset:
            sys.exit(
                f"[ERROR] Unknown --method '{args.method}'. "
                "Add it to conf/embeddings.yaml or use an existing name."
            )
    args.model_name = args.model_name or preset.get("model_name") or "microsoft/deberta-v3-base"
    args.text_column = args.text_column or preset.get("text_column") or "clean_text"
    return dict(preset.get("tokenizer_kwargs", {}))


def main() -> None:
    args = parse_args()
    tokenizer_kwargs = resolve_encoder(args)
    logger.info(
        "Fine-tuning encoder: %s | model=%s | text_column=%s",
        args.method or "(flags)", args.model_name, args.text_column,
    )
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # bf16 sidesteps DeBERTa-v3's fp16 overflow issues; fall back to fp32 on CPU.
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s | autocast: %s", device, autocast_dtype)

    train_df = load_split(args.data_dir / "train.csv", args.text_column, args.label_column)
    val_df = load_split(args.data_dir / "val.csv", args.text_column, args.label_column)
    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = train_df.sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    # Seeded subsample keeps mid-epoch evals to ~1 min; the final eval uses full val.
    eval_df = val_df
    if args.eval_samples and args.eval_samples < len(val_df):
        eval_df = val_df.sample(n=args.eval_samples, random_state=args.seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    # transformers v5 loads this checkpoint in fp16 by default; fp16 master
    # weights make AdamW's eps/exp_avg_sq underflow to zero and NaN the whole
    # model on the first optimizer step. Force fp32 and let autocast handle speed.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, dtype=torch.float32
    )
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    collate = make_collate(tokenizer, args.max_length)
    labels_np = train_df[args.label_column].to_numpy(dtype=np.int64)
    train_loader = DataLoader(
        TextDataset(train_df[args.text_column].tolist(), labels_np),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(
        TextDataset(eval_df[args.text_column].tolist(), eval_df[args.label_column].to_numpy(dtype=np.int64)),
        batch_size=args.eval_batch_size, collate_fn=collate, pin_memory=device.type == "cuda",
    )

    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = linear_warmup_schedule(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    logger.info(
        "Optimizer steps: %d (%d/epoch) | effective batch %d | warmup %d",
        total_steps, steps_per_epoch, args.batch_size * args.grad_accum,
        int(total_steps * args.warmup_ratio),
    )

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    best_path = args.output_dir / "best"
    history: list[dict[str, float]] = []
    best_acc, global_step = 0.0, 0
    running_loss, running_correct, running_count = 0.0, 0, 0
    t0 = time.time()

    def run_eval() -> None:
        nonlocal best_acc
        metrics = evaluate(model, eval_loader, device, autocast_dtype)
        writer.add_scalar("eval/loss", metrics["loss"], global_step)
        writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step)
        history.append({"step": global_step, **metrics})
        improved = metrics["accuracy"] > best_acc
        if improved:
            best_acc = metrics["accuracy"]
            save_checkpoint(model, tokenizer, best_path)
        logger.info(
            "[Step %6d] eval_loss=%.4f eval_acc=%.4f%s",
            global_step, metrics["loss"], metrics["accuracy"],
            "  * new best (checkpoint saved)" if improved else "",
        )

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for i, batch in enumerate(progress, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                out = model(**batch)
            (out.loss / args.grad_accum).backward()

            running_loss += out.loss.item() * batch["labels"].size(0)
            running_correct += (out.logits.argmax(dim=-1) == batch["labels"]).sum().item()
            running_count += batch["labels"].size(0)

            if i % args.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 100 == 0:
                    train_loss = running_loss / running_count
                    train_acc = running_correct / running_count
                    writer.add_scalar("train/loss", train_loss, global_step)
                    writer.add_scalar("train/accuracy", train_acc, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    progress.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}")
                    running_loss, running_correct, running_count = 0.0, 0, 0

                if global_step % args.eval_every == 0:
                    run_eval()

    logger.info("Training done in %.1f min. Final evaluation on the FULL val split ...", (time.time() - t0) / 60)
    full_eval_loader = DataLoader(
        TextDataset(val_df[args.text_column].tolist(), val_df[args.label_column].to_numpy(dtype=np.int64)),
        batch_size=args.eval_batch_size, collate_fn=collate, pin_memory=device.type == "cuda",
    )
    final = evaluate(model, full_eval_loader, device, autocast_dtype)
    logger.info("Full val: loss=%.4f acc=%.4f (best subsample acc during training: %.4f)",
                final["loss"], final["accuracy"], best_acc)
    if final["accuracy"] >= best_acc:
        best_acc = final["accuracy"]
        save_checkpoint(model, tokenizer, best_path)
        logger.info("Final model saved as best -> %s", best_path.resolve())

    save_checkpoint(model, tokenizer, args.output_dir / "last")
    with open(args.output_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"history": history, "final_full_val": final, "best_accuracy": best_acc,
             "args": {k: str(v) for k, v in vars(args).items()}},
            fh, indent=2,
        )
    writer.close()
    logger.info("[DONE] Best checkpoint: %s | best val acc %.4f", best_path.resolve(), best_acc)


if __name__ == "__main__":
    main()

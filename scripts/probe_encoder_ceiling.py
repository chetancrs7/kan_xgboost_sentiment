"""Linear-probe ceiling of candidate encoders (read-only, standalone).

Answers one question before anyone invests in re-training: *is there accuracy
headroom above the current KAN (~0.872) in a better sentence representation?*

For a seeded stratified subsample it measures the **linear-probe accuracy**
(StandardScaler + logistic regression) of each encoder's sentence embeddings.
The probe accuracy is that encoder's practical ceiling — if a linear model can't
separate the classes, neither will a KAN on top of the same vectors.

Encoders compared:
  * ``deberta_ft`` — your current fine-tuned DeBERTa, read from the *existing*
    on-disk embeddings (``data/embeddings_finetuned``), indexed to the subsample.
    No re-embedding; this is the incumbent ceiling on the same rows.
  * ``bertweet_ft`` — fine-tuned BERTweet (``models/bertweet_finetuned/best``),
    embedded fresh with the KNOWN tokenizer fix (base ``vinai/bertweet-base``
    tokenizer, raw text, masked-mean pooling). Its saved tokenizer is broken, so
    we never use it.

This script imports nothing from the training pipeline and writes only to its
own output dir — it cannot affect any running job.

Usage::

    python scripts/probe_encoder_ceiling.py
    python scripts/probe_encoder_ceiling.py --train-size 20000 --val-size 5000 --test-size 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("train", "val", "test")
LABEL_COLUMN = "label"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-dir", default="data/splitsfull")
    p.add_argument("--deberta-emb-dir", default="data/embeddings_finetuned",
                   help="Existing DeBERTa embeddings (row-aligned with --splits-dir).")
    p.add_argument("--bertweet-dir", default="models/bertweet_finetuned/best")
    p.add_argument("--bertweet-tokenizer", default="vinai/bertweet-base",
                   help="Base hub tokenizer (the saved BERTweet tokenizer is broken).")
    p.add_argument("--bertweet-normalization", action="store_true",
                   help="Apply BERTweet's tweet normalization (@USER/HTTPURL) — the "
                   "preprocessing BERTweet was pretrained with.")
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--text-column", default="text", help="Raw text column for BERTweet.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="outputs/encoder_ceiling_probe")
    return p.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_idx(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Seeded, label-stratified row positions (sorted)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    per = (counts / counts.sum() * size).round().astype(int)
    per[0] += size - int(per.sum())  # fix rounding drift on the largest bucket
    chosen = []
    for cls, n_cls in zip(classes, per):
        pool = np.flatnonzero(labels == cls)
        chosen.append(rng.choice(pool, size=min(n_cls, len(pool)), replace=False))
    return np.sort(np.concatenate(chosen))


@torch.no_grad()
def embed_bertweet(texts, model, tokenizer, device, batch_size, max_length) -> np.ndarray:
    """Masked-mean-pooled last-hidden-state embeddings (raw text)."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        out.append(pooled.float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"    embedded {min(i + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    return np.concatenate(out).astype(np.float32)


def linear_probe(Xtr, ytr, Xval, yval, Xte, yte) -> dict:
    """StandardScaler + logistic-regression accuracy = the encoder's ceiling."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(scaler.transform(Xtr), ytr)
    acc = lambda X, y: float((clf.predict(scaler.transform(X)) == y).mean())
    return {"val_acc": acc(Xval, yval), "test_acc": acc(Xte, yte)}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[INFO] device={device}")

    # 1. Subsample indices per split (shared by both encoders) ---------------
    dfs, idx = {}, {}
    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    for s in SPLITS:
        df = pd.read_csv(Path(args.splits_dir) / f"{s}.csv")
        df[args.text_column] = df[args.text_column].fillna("").astype(str)
        labels = df[LABEL_COLUMN].to_numpy()
        idx[s] = stratified_idx(labels, sizes[s], args.seed)
        dfs[s] = df
        print(f"[INFO] {s}: sampled {len(idx[s]):,} of {len(df):,}")

    labels = {s: dfs[s][LABEL_COLUMN].to_numpy()[idx[s]].astype(np.int64) for s in SPLITS}
    results = {}

    # 2. DeBERTa (incumbent) from existing embeddings ------------------------
    try:
        emb_dir = Path(args.deberta_emb_dir)
        deb = {}
        for s in SPLITS:
            arr = np.load(emb_dir / f"{s}_embeddings.npy", mmap_mode="r")
            deb[s] = np.asarray(arr[idx[s]], dtype=np.float32)
        results["deberta_ft"] = linear_probe(
            deb["train"], labels["train"], deb["val"], labels["val"], deb["test"], labels["test"]
        )
        print(f"[deberta_ft] {results['deberta_ft']}")
    except Exception as exc:  # never let the incumbent block the candidate
        print(f"[WARN] DeBERTa probe skipped: {exc}")

    # 3. BERTweet fine-tuned (candidate), embedded fresh with the fix --------
    try:
        from transformers import AutoModel, AutoTokenizer
        print(f"[INFO] loading BERTweet model={args.bertweet_dir} "
              f"tokenizer={args.bertweet_tokenizer} (base hub tokenizer — saved one is broken)")
        try:
            tok = AutoTokenizer.from_pretrained(
                args.bertweet_tokenizer, use_fast=False,
                normalization=args.bertweet_normalization,
            )
            print(f"[INFO] BERTweet normalization={args.bertweet_normalization}")
        except Exception as tok_exc:
            print(f"[WARN] slow tokenizer failed ({tok_exc}); trying fast.")
            tok = AutoTokenizer.from_pretrained(args.bertweet_tokenizer)
        model = AutoModel.from_pretrained(args.bertweet_dir, dtype=torch.float32).to(device)
        bt = {}
        for s in SPLITS:
            texts = dfs[s][args.text_column].to_numpy()[idx[s]].tolist()
            print(f"[INFO] embedding {s} with BERTweet-ft ...")
            bt[s] = embed_bertweet(texts, model, tok, device, args.batch_size, args.max_length)
        results["bertweet_ft"] = linear_probe(
            bt["train"], labels["train"], bt["val"], labels["val"], bt["test"], labels["test"]
        )
        print(f"[bertweet_ft] {results['bertweet_ft']}")
    except Exception as exc:
        print(f"[WARN] BERTweet probe failed: {exc}")

    # 4. Report --------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"sizes": sizes, "seed": args.seed, "results": results,
               "reference": {"kan_test": 0.872, "deberta_head": 0.877}}
    (out / "probe_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 56 + "\nENCODER LINEAR-PROBE CEILING (higher = more headroom)\n" + "=" * 56)
    print(f"  {'encoder':<14} {'val_acc':>9} {'test_acc':>9}")
    for name, r in results.items():
        print(f"  {name:<14} {r['val_acc']:>9.4f} {r['test_acc']:>9.4f}")
    print(f"  {'(KAN now)':<14} {'—':>9} {0.872:>9.4f}")
    print(f"\n[INFO] saved {out / 'probe_results.json'}")


if __name__ == "__main__":
    main()

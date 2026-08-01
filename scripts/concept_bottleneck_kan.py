"""Concept-Bottleneck KAN -- a fully-semantic, interpretable KAN.

The problem with a KAN over raw embeddings: its inputs are 768 opaque
dimensions, so the learned spline curves have no human meaning. The fix here is
a *concept bottleneck*: project each tweet embedding onto a curated set of
NAMED semantic concepts (joy, anger, sarcasm, ...), then train a single-layer
KAN on those concept scores. The KAN is then a spline **GAM**:

    sentiment_logit = sum_c  phi_c( score_of_concept_c )

so every edge phi_c is a readable curve: "how the 'anger' score maps to
sentiment". These concepts are *meaning*-based (distinct from the XGBoost
expert's *form*-based structural features -- negation counts, punctuation, etc.),
so the two experts stay complementary.

Standalone: imports only the project's KAN spline layer, reads precomputed
embeddings + the fine-tuned encoder, and writes to its own --output-dir. No
existing code is modified.

Pipeline:
  1. embed ~30 named concept anchors with the fine-tuned DeBERTa encoder,
  2. concept score = cosine(tweet_embedding, concept_anchor)  (train-fit scaler),
  3. train a 1-layer KAN (spline GAM) on the concept scores,
  4. extract each concept's spline effect curve + an importance ranking.

Usage:
    python scripts/concept_bottleneck_kan.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kan.layers import AdaptiveSplineKANLayer  # noqa: E402
from scripts.diagnose_kan_vs_embeddings import (  # noqa: E402
    SPLITS, load_split, run_logreg, scores, set_seed, stratified_subsample,
)
from transformers import AutoModel, AutoTokenizer  # noqa: E402


# ~30 meaning-based concepts (NOT structural -- those belong to XGBoost).
CONCEPTS: dict[str, list[str]] = {
    "joy": ["joy", "happy", "delighted", "cheerful"],
    "love": ["love", "adore", "affection", "cherish"],
    "gratitude": ["grateful", "thankful", "appreciated"],
    "excitement": ["excited", "thrilled", "can't wait"],
    "pride": ["proud", "accomplished", "achievement"],
    "hope": ["hopeful", "optimistic", "looking forward"],
    "relief": ["relieved", "finally over", "phew"],
    "amusement": ["funny", "hilarious", "laughing"],
    "admiration": ["amazing", "awesome", "impressive"],
    "calm": ["calm", "peaceful", "relaxed"],
    "anger": ["anger", "furious", "rage", "annoyed"],
    "sadness": ["sadness", "grief", "heartbroken", "depressed"],
    "fear": ["afraid", "scared", "anxious", "worried"],
    "disgust": ["disgusting", "gross", "revolting"],
    "disappointment": ["disappointed", "let down", "frustrated"],
    "loneliness": ["lonely", "alone", "isolated"],
    "boredom": ["bored", "boring", "dull"],
    "jealousy": ["jealous", "envious"],
    "guilt": ["guilty", "ashamed", "regret"],
    "tiredness": ["tired", "exhausted", "sleepy"],
    "pain_illness": ["sick", "pain", "hurts", "ill"],
    "sarcasm": ["sarcasm", "ironic", "yeah right", "how wonderful"],
    "complaint": ["complaint", "terrible service", "worst"],
    "praise": ["praise", "well done", "compliment"],
    "sympathy": ["sorry for you", "sympathy", "hope you feel better"],
    "good_news": ["great news", "good news", "congratulations"],
    "bad_news": ["bad news", "unfortunately", "sad to say"],
    "success": ["success", "won", "achieved", "nailed it"],
    "failure": ["failure", "lost", "messed up", "ruined"],
    "celebration": ["celebrate", "party", "birthday"],
}


def embed_texts(encoder, tok, texts, device, max_len=16, batch=32):
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            e = tok(texts[i : i + batch], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(device)
            h = encoder(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).cpu().numpy())
    return np.concatenate(out)


def build_anchors(encoder, tok, device):
    names = list(CONCEPTS)
    anchors = np.stack([embed_texts(encoder, tok, CONCEPTS[n], device).mean(0) for n in names])
    anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    return names, anchors.astype(np.float32)


def concept_scores(X, anchors, chunk=100000):
    """Cosine sim of each row with each concept anchor, computed in chunks so a
    full 1.28M x 768 matrix never gets duplicated in RAM."""
    out = np.empty((len(X), anchors.shape[0]), dtype=np.float32)
    for i in range(0, len(X), chunk):
        Xi = X[i : i + chunk]
        Xn = Xi / (np.linalg.norm(Xi, axis=1, keepdims=True) + 1e-8)
        out[i : i + chunk] = Xn @ anchors.T
    return out


class KANGam(nn.Module):
    """Single KAN layer: concept scores -> class logits (an additive spline GAM)."""

    def __init__(self, n_concepts, n_classes, grid_size=3, spline_order=3):
        super().__init__()
        self.layer = AdaptiveSplineKANLayer(n_concepts, n_classes, grid_size, spline_order)

    def forward(self, x):
        return self.layer(x)

    @torch.no_grad()
    def seed(self, x):
        self.layer.adapt_grid(x)


@torch.no_grad()
def forward_logits(model, X, device, chunk=16384):
    """Chunked forward so a 160k-row eval never allocates one giant tensor."""
    model.eval()
    outs = []
    for i in range(0, len(X), chunk):
        outs.append(model(torch.from_numpy(X[i : i + chunk]).to(device)).cpu())
    return torch.cat(outs)


def train_gam(model, Xtr, ytr, Xva, yva, device, *, epochs, batch, lr, wd, patience, seed):
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, generator=g)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.seed(torch.from_numpy(Xtr[:512]).to(device))
    import copy
    import time
    best, best_state, wait = 1e9, copy.deepcopy(model.state_dict()), 0
    yva_t = torch.from_numpy(yva)
    hist = {"val_acc": []}
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        logits = forward_logits(model, Xva, device)
        vl = crit(logits, yva_t).item()
        va = (logits.argmax(1) == yva_t).float().mean().item()
        hist["val_acc"].append(va)
        marker = ""
        if vl < best - 1e-5:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            marker = " *"
        else:
            wait += 1
        print(f"  [ep {ep:03d}] val_acc={va:.4f} val_loss={vl:.4f} ({time.time()-t0:.0f}s){marker}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return hist


def predict(model, X, device):
    return forward_logits(model, X, device).argmax(1).numpy()


@torch.no_grad()
def concept_effect_curve(model, concept_idx, X, device, n=100):
    """Sweep one concept over its data range, holding others at 0 (the
    standardized mean). Returns (xs, effect) where effect is the concept's
    contribution to the positive-minus-negative logit (its sentiment push).
    Because the GAM is additive, sweeping one input isolates its edge."""
    lo, hi = np.percentile(X[:, concept_idx], [2, 98])
    xs = np.linspace(lo, hi, n).astype(np.float32)
    inp = np.zeros((n, X.shape[1]), dtype=np.float32)
    inp[:, concept_idx] = xs
    out = model(torch.from_numpy(inp).to(device)).cpu().numpy()
    eff = out[:, 1] - out[:, 0]
    return xs, eff - eff.mean()


SARCASM_MARKER_RE = r"#sarcas|#iron|#not\b|\bsarcas|yeah,? right|oh great|just great|oh joy|how wonderful|/s\b"


def sarcasm_slice_eval(model, anchors, scaler, embeddings_dir, test_csv, device):
    """Evaluate the model specifically on the test tweets that carry sarcasm
    cues/markers, vs the rest. These are HEURISTIC markers (sarcasm_flag,
    sarcasm_cue_count, explicit cue phrases) already in the dataset -- NOT
    human-verified sarcasm labels -- so this is 'accuracy on cue-flagged tweets',
    not a validated sarcasm benchmark. Runs on the FULL test split (row-aligned
    with the CSV) regardless of any --max-eval subsample used for the headline
    metrics."""
    import re
    import pandas as pd

    emb = Path(embeddings_dir)
    Xe = np.load(emb / "test_embeddings.npy").astype(np.float32)
    ye = np.load(emb / "test_labels.npy").astype(np.int64)
    scores = scaler.transform(concept_scores(Xe, anchors)).astype(np.float32)
    pred = predict(model, scores, device)
    correct = (pred == ye)

    df = pd.read_csv(test_csv, usecols=["text", "label", "sarcasm_flag", "sarcasm_cue_count"])
    if not np.array_equal(df["label"].to_numpy(), ye):
        print(f"[WARN] {test_csv} not row-aligned with {emb} test labels; skipping sarcasm slice.")
        return None
    t = df["text"].astype(str).str.lower()
    markers = t.str.contains(SARCASM_MARKER_RE, regex=True, na=False).to_numpy()
    flag = (df["sarcasm_flag"] == 1).to_numpy()
    cue1 = (df["sarcasm_cue_count"] >= 1).to_numpy()
    cue2 = (df["sarcasm_cue_count"] >= 2).to_numpy()

    def slice_stats(mask):
        return {
            "n_sarcastic": int(mask.sum()),
            "correct_sarcastic": int(correct[mask].sum()),
            "acc_sarcastic": float(correct[mask].mean()) if mask.any() else None,
            "acc_non_sarcastic": float(correct[~mask].mean()),
        }

    out = {
        "note": "heuristic sarcasm markers (cues/flags), NOT verified labels; "
                "Sentiment140 labels are emoticon-derived and may be wrong for sarcasm",
        "n_test": int(len(ye)),
        "overall_acc": float(correct.mean()),
        "slices": {
            "broad (flag | cue>=1 | text-marker)": slice_stats(flag | cue1 | markers),
            "strict (flag | cue>=2 | text-marker)": slice_stats(flag | cue2 | markers),
        },
    }
    return out


def save_plots(names, importance, ranked, curves, out_dir, top_bar=15, top_curves=12):
    """Save the two interpretation figures: concept-importance bar + per-concept
    spline curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def is_pos(n):
        return curves[n]["effect"][-1] > curves[n]["effect"][0]

    # 1. Importance bar (green = pushes positive, red = pushes negative, blue = sarcasm/non-monotonic).
    top = ranked[:top_bar][::-1]
    colors = ["#2a78d6" if n == "sarcasm" else ("#1baf7a" if is_pos(n) else "#eb6834") for n in top]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), [importance[n] for n in top], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n.replace("_", " ") for n in top])
    ax.set_xlabel("sentiment swing (learned importance)")
    ax.set_title("Concept importance — Concept-Bottleneck KAN")
    fig.tight_layout()
    fig.savefig(plots / "concept_importance.png", dpi=150)
    plt.close(fig)

    # 2. Per-concept spline curves (small multiples for the top concepts).
    tp = ranked[:top_curves]
    ncol = 4
    nrow = (len(tp) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow))
    for ax, n in zip(axes.ravel(), tp):
        color = "#2a78d6" if n == "sarcasm" else ("#1baf7a" if is_pos(n) else "#eb6834")
        ax.plot(curves[n]["x"], curves[n]["effect"], color=color, lw=2)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{n.replace('_', ' ')} (swing {importance[n]:.2f})", fontsize=10)
        ax.set_xlabel("concept score"); ax.set_ylabel("sentiment effect")
    for ax in axes.ravel()[len(tp):]:
        ax.axis("off")
    fig.suptitle("Learned spline curves per concept (up = more positive)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plots / "concept_curves.png", dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved graphs to {plots.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", default="data/embeddings_finetuned")
    ap.add_argument("--encoder", default="models/deberta_finetuned/best")
    ap.add_argument("--output-dir", default="outputs/concept_kan")
    ap.add_argument("--test-csv", default="data/splitsfull/test.csv",
                    help="Test split CSV (must match --embeddings-dir) for the sarcasm-cue slice.")
    ap.add_argument("--sarcasm-eval", action=argparse.BooleanOptionalAction, default=True,
                    help="Report accuracy on the sarcasm-cue-flagged test slice.")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-eval", type=int, default=5000)
    ap.add_argument("--grid-size", type=int, default=6)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Concept-Bottleneck KAN | encoder={args.encoder} | device={device}")

    # 1. Concept anchors -----------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder, dtype=torch.float32).to(device)
    names, anchors = build_anchors(enc, tok, device)
    print(f"[INFO] Built {len(names)} concept anchors.")
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Concept scores per split -------------------------------------------
    raw = {s: load_split(Path(args.embeddings_dir), s) for s in SPLITS}
    data = {}
    for s in SPLITS:
        cap = args.max_train if s == "train" else args.max_eval
        X, y = stratified_subsample(*raw[s], cap, args.seed)
        data[s] = (concept_scores(X, anchors), y)
    del raw  # free ~4.8 GB of raw embeddings before training (matters at full scale)
    import gc; gc.collect()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data["train"][0])
    data = {s: (scaler.transform(X).astype(np.float32), y) for s, (X, y) in data.items()}
    n_classes = int(len(np.unique(data["train"][1])))
    print(f"[INFO] concept scores: {data['train'][0].shape} | classes={n_classes}")

    # Sanity: per-concept correlation with the label (linear signal).
    corr = {}
    Xtr, ytr = data["train"]
    for i, name in enumerate(names):
        corr[name] = float(np.corrcoef(Xtr[:, i], ytr)[0, 1])

    # 3. Baselines + KAN GAM -------------------------------------------------
    splits_for_lr = {s: (data[s][0], data[s][1]) for s in SPLITS}
    lr = run_logreg(splits_for_lr, args.seed)
    print(f"[INFO] logreg on concepts: val={lr['val']['accuracy']:.4f}")

    set_seed(args.seed)
    model = KANGam(len(names), n_classes, args.grid_size, args.spline_order).to(device)
    train_gam(model, data["train"][0], data["train"][1], data["val"][0], data["val"][1],
              device, epochs=args.epochs, batch=args.batch_size, lr=args.lr,
              wd=args.weight_decay, patience=args.patience, seed=args.seed)
    kan_res = {s: scores(data[s][1], predict(model, data[s][0], device)) for s in SPLITS}

    # Export FULL val/test KAN probabilities (row-aligned with the splits, and thus
    # with the XGBoost expert) so the interpretable KAN can be fused downstream.
    # Independent of --max-eval so it always covers every row.
    import torch.nn.functional as F
    out_probs = Path(args.output_dir); out_probs.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        Xe = np.load(Path(args.embeddings_dir) / f"{split}_embeddings.npy").astype(np.float32)
        sc = scaler.transform(concept_scores(Xe, anchors)).astype(np.float32)
        probs = F.softmax(forward_logits(model, sc, device), dim=1).numpy()
        np.save(out_probs / f"{split}_probs.npy", probs)
    print(f"[INFO] Saved full val/test KAN probabilities to {out_probs} (for fusion).")

    # 4. Per-concept effect curves + importance -----------------------------
    curves, importance = {}, {}
    for i, name in enumerate(names):
        xs, eff = concept_effect_curve(model, i, data["train"][0], device)
        curves[name] = {"x": xs.tolist(), "effect": eff.tolist()}
        importance[name] = float(eff.max() - eff.min())  # total sentiment swing
    ranked = sorted(importance, key=importance.get, reverse=True)

    # 5. Report --------------------------------------------------------------
    print("\n" + "=" * 66)
    print("RESULTS  (concept-bottleneck KAN)")
    print("=" * 66)
    print(f"  logreg (linear on concepts)  val={lr['val']['accuracy']:.4f} test={lr['test']['accuracy']:.4f}")
    print(f"  KAN-GAM (spline on concepts) val={kan_res['val']['accuracy']:.4f} test={kan_res['test']['accuracy']:.4f}")
    print("\n  Top concepts by learned sentiment swing (importance):")
    for name in ranked[:12]:
        arrow = "positive" if curves[name]["effect"][-1] > curves[name]["effect"][0] else "negative"
        print(f"    {name:<15} swing={importance[name]:.3f}  corr={corr[name]:+.3f}  trend~{arrow}")

    # Sarcasm-cue slice: how many cue-flagged test tweets, and is the model right on them?
    sarcasm = None
    if args.sarcasm_eval:
        try:
            sarcasm = sarcasm_slice_eval(model, anchors, scaler, args.embeddings_dir, args.test_csv, device)
        except Exception as e:
            print(f"[WARN] sarcasm slice eval failed: {e}")
    if sarcasm:
        print("\n" + "=" * 66)
        print("SARCASM-CUE SLICE  (heuristic markers, not verified sarcasm labels)")
        print("=" * 66)
        print(f"  overall test acc = {sarcasm['overall_acc']:.4f}  (n={sarcasm['n_test']:,})")
        for name, s in sarcasm["slices"].items():
            acc = "n/a" if s["acc_sarcastic"] is None else f"{s['acc_sarcastic']:.4f}"
            print(f"  {name}")
            print(f"      sarcastic tweets: {s['n_sarcastic']:,}  |  model correct: {s['correct_sarcastic']:,}  "
                  f"|  acc={acc}  vs non-sarcastic acc={s['acc_non_sarcastic']:.4f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    save_plots(names, importance, ranked, curves, out)
    json.dump({
        "sarcasm_slice": sarcasm,
        "concepts": names, "correlation": corr, "importance": importance,
        "ranked": ranked, "curves": curves,
        "results": {"logreg": lr, "kan_gam": kan_res},
        "config": vars(args),
    }, open(out / "concept_kan_results.json", "w"), indent=2)
    print(f"\n[INFO] Saved to {(out / 'concept_kan_results.json').resolve()}")


if __name__ == "__main__":
    main()

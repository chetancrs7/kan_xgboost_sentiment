"""Interpretable additive Concept-Bottleneck KAN (Sentiment140 only).

A report-grade, mathematically decomposable KAN:

    sentiment_logit = bias + sum_c  phi_c( intensity_c )

Every named concept c connects to the output through ONE learnable cubic
B-spline phi_c, so the model is an additive GAM whose every edge is a readable
curve. Implements the additive-KAN methodology on Sentiment140:

  * concept extractor -> [0,1] intensity per concept (zero-shot: cosine to
    encoder-embedded anchor words, per-concept min-max calibrated on train),
  * additive spline head (grid 6 cubic B-spline) trained with sentiment BCE +
    SMOOTHNESS (penalise curvature) + SPARSITY (unimportant splines stay flat) +
    SOFT MONOTONICITY (joy/gratitude/... forced up, anger/sadness/... forced
    down; sarcasm/surprise/mixed left free),
  * robustness over multiple seeds -> confidence bands on every spline +
    accuracy mean/std,
  * per-prediction decomposition (intensities -> spline contributions -> logit),
  * exports FULL train/val/test probs so it fuses with the structural XGBoost.

STRICTLY Sentiment140: no external emotion dataset, no teacher model. Because
Sentiment140 has NO concept labels, the *supervised* concept-extractor training
and concept-level F1/AUROC/Brier/ECE/calibration described in the methodology
are NOT computable here and are reported as out-of-scope, never fabricated.

Standalone: imports only the project's KAN spline layer + read-only helpers;
writes to its own --output-dir.

Usage:
    python scripts/concept_kan_report.py
    python scripts/concept_kan_report.py --smoke   # tiny fast self-check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kan.layers import AdaptiveSplineKANLayer  # noqa: E402
from scripts.concept_bottleneck_kan import concept_scores, embed_texts  # noqa: E402
from scripts.diagnose_kan_vs_embeddings import (  # noqa: E402
    SPLITS, load_split, run_logreg, scores, set_seed, stratified_subsample,
)
from transformers import AutoModel, AutoTokenizer  # noqa: E402

# name -> (anchor phrases, direction, family). direction: +1 expect-positive,
# -1 expect-negative, 0 unconstrained (context concepts kept nonlinear).
CONCEPTS: dict[str, tuple[list[str], int, str]] = {
    # ---- positive emotions (monotone up) --------------------------------
    "joy": (["joy", "happy", "delighted", "cheerful"], +1, "positive"),
    "love": (["love", "adore", "affection", "cherish"], +1, "positive"),
    "gratitude": (["grateful", "thankful", "appreciated"], +1, "positive"),
    "excitement": (["excited", "thrilled", "can't wait"], +1, "positive"),
    "pride": (["proud", "accomplished", "achievement"], +1, "positive"),
    "hope": (["hopeful", "optimistic", "looking forward"], +1, "positive"),
    "relief": (["relieved", "finally over", "phew"], +1, "positive"),
    "amusement": (["funny", "hilarious", "laughing"], +1, "positive"),
    "admiration": (["amazing", "awesome", "impressive"], +1, "positive"),
    "calm": (["calm", "peaceful", "relaxed"], +1, "positive"),
    "contentment": (["content", "satisfied", "at ease"], +1, "positive"),
    "enthusiasm": (["enthusiastic", "eager", "pumped"], +1, "positive"),
    "optimism": (["optimistic", "bright future", "things will improve"], +1, "positive"),
    "affection": (["sweet", "caring", "warm"], +1, "positive"),
    "satisfaction": (["satisfied", "pleased", "just right"], +1, "positive"),
    # ---- negative emotions (monotone down) ------------------------------
    "anger": (["anger", "furious", "rage", "annoyed"], -1, "negative"),
    "sadness": (["sadness", "grief", "heartbroken", "depressed"], -1, "negative"),
    "fear": (["afraid", "scared", "anxious", "worried"], -1, "negative"),
    "disgust": (["disgusting", "gross", "revolting"], -1, "negative"),
    "disappointment": (["disappointed", "let down", "frustrated"], -1, "negative"),
    "loneliness": (["lonely", "alone", "isolated"], -1, "negative"),
    "boredom": (["bored", "boring", "dull"], -1, "negative"),
    "jealousy": (["jealous", "envious"], -1, "negative"),
    "guilt": (["guilty", "ashamed", "regret"], -1, "negative"),
    "tiredness": (["tired", "exhausted", "sleepy"], -1, "negative"),
    "pain_illness": (["sick", "pain", "hurts", "ill"], -1, "negative"),
    "frustration": (["frustrated", "fed up", "so annoying"], -1, "negative"),
    "anxiety": (["anxious", "nervous", "on edge"], -1, "negative"),
    "despair": (["hopeless", "despair", "give up"], -1, "negative"),
    "embarrassment": (["embarrassed", "humiliated", "cringe"], -1, "negative"),
    "contempt": (["contempt", "pathetic", "disdain"], -1, "negative"),
    "regret": (["regret", "wish i hadn't", "my mistake"], -1, "negative"),
    # ---- evaluative / discourse (directional) ---------------------------
    "praise": (["praise", "well done", "compliment"], +1, "social_pos"),
    "good_news": (["great news", "good news", "congratulations"], +1, "social_pos"),
    "success": (["success", "won", "achieved", "nailed it"], +1, "social_pos"),
    "celebration": (["celebrate", "party", "birthday"], +1, "social_pos"),
    "congratulation": (["congrats", "well deserved", "way to go"], +1, "social_pos"),
    "complaint": (["complaint", "terrible service", "worst"], -1, "social_neg"),
    "bad_news": (["bad news", "unfortunately", "sad to say"], -1, "social_neg"),
    "failure": (["failure", "lost", "messed up", "ruined"], -1, "social_neg"),
    "criticism": (["criticism", "not good enough", "letdown"], -1, "social_neg"),
    # ---- context concepts (LEFT UNCONSTRAINED / nonlinear) --------------
    "sarcasm": (["sarcasm", "ironic", "yeah right", "how wonderful"], 0, "context"),
    "surprise": (["surprised", "unexpected", "no way"], 0, "context"),
    "confusion": (["confused", "don't understand", "what is going on"], 0, "context"),
    "curiosity": (["curious", "wonder", "interesting question"], 0, "context"),
    "nostalgia": (["nostalgic", "miss the old days", "remember when"], 0, "context"),
    "sympathy": (["sorry for you", "sympathy", "hope you feel better"], 0, "context"),
    "apology": (["sorry", "my apologies", "forgive me"], 0, "context"),
    "mixed_emotion": (["bittersweet", "mixed feelings", "happy and sad"], 0, "context"),
}

FAMILY_COLOR = {
    "positive": "#1baf7a", "social_pos": "#5fbf8f",
    "negative": "#eb6834", "social_neg": "#d1743f",
    "context": "#2a78d6",
}


# --------------------------------------------------------------------------- #
# Concept extractor: cosine -> [0,1] intensity (per-concept min-max on train)
# --------------------------------------------------------------------------- #
def fit_intensity(cos_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(cos_train, 2, axis=0)
    hi = np.percentile(cos_train, 98, axis=0)
    hi = np.where(hi - lo < 1e-6, lo + 1e-6, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def to_intensity(cos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip((cos - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Additive concept KAN:  logit = bias + sum_c phi_c(intensity_c)
# --------------------------------------------------------------------------- #
class AdditiveConceptKAN(nn.Module):
    def __init__(self, n_concepts: int, grid_size: int = 6, spline_order: int = 3):
        super().__init__()
        self.layer = AdaptiveSplineKANLayer(n_concepts, 1, grid_size, spline_order)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias + self.layer(x).squeeze(-1)

    @torch.no_grad()
    def seed(self, x: torch.Tensor) -> None:
        self.layer.adapt_grid(x)


def concept_curves(model: AdditiveConceptKAN, n_concepts: int, device, G: int = 24,
                   baseline: float = 0.5) -> torch.Tensor:
    """[n_concepts, G] output as each concept sweeps [0,1], others at baseline.

    Additivity makes the shape over G exactly phi_c (up to a constant), so
    curvature/slope/range read straight off these rows.
    """
    g = torch.linspace(0.0, 1.0, G, device=device)
    base = torch.full((n_concepts, G, n_concepts), baseline, device=device)
    idx = torch.arange(n_concepts, device=device)
    base[idx, :, idx] = g.unsqueeze(0).expand(n_concepts, G)
    out = model(base.reshape(n_concepts * G, n_concepts)).reshape(n_concepts, G)
    return out


def regularizers(curves: torch.Tensor, direction: torch.Tensor):
    """Smoothness (2nd-diff), sparsity (curve range), soft monotonicity."""
    second = curves[:, 2:] - 2 * curves[:, 1:-1] + curves[:, :-2]
    smooth = (second ** 2).mean()
    sparse = (curves.max(1).values - curves.min(1).values).abs().mean()
    diffs = curves[:, 1:] - curves[:, :-1]
    up = (direction == 1).float().unsqueeze(1)     # want increasing
    down = (direction == -1).float().unsqueeze(1)  # want decreasing
    mono = (up * torch.relu(-diffs) + down * torch.relu(diffs)).mean()
    return smooth, sparse, mono


@torch.no_grad()
def forward_logits(model, X, device, chunk=16384):
    model.eval()
    out = []
    for i in range(0, len(X), chunk):
        out.append(model(torch.from_numpy(X[i:i + chunk]).to(device)).cpu())
    return torch.cat(out)


def train_one(model, Xtr, ytr, Xva, yva, direction, device, *, epochs, batch, lr,
              wd, patience, seed, l_smooth, l_sparse, l_mono):
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr),
                                        torch.from_numpy(ytr.astype(np.float32)))
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, generator=g)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.seed(torch.from_numpy(Xtr[:512]).to(device))
    n_concepts = Xtr.shape[1]
    yva_t = torch.from_numpy(yva)
    best, best_state, wait = 1e9, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = bce(logit, yb)
            smooth, sparse, mono = regularizers(
                concept_curves(model, n_concepts, device), direction)
            loss = loss + l_smooth * smooth + l_sparse * sparse + l_mono * mono
            loss.backward()
            opt.step()
        logits = forward_logits(model, Xva, device)
        vloss = bce(logits, yva_t.float()).item()
        vacc = ((logits > 0).long() == yva_t).float().mean().item()
        if vloss < best - 1e-5:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return vacc


@torch.no_grad()
def probs_full(model, cos, lo, hi, device):
    """[N,2] softmax-style probs from the scalar logit (for fusion)."""
    x = to_intensity(cos, lo, hi)
    logit = forward_logits(model, x, device).numpy()
    p = 1.0 / (1.0 + np.exp(-logit))
    return np.stack([1 - p, p], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@torch.no_grad()
def curve_effect(model, n_concepts, device, G=60):
    """Centred per-concept effect curves (bias removed) for plotting."""
    out = concept_curves(model, n_concepts, device, G=G).cpu().numpy()
    return out - out.mean(axis=1, keepdims=True)


def build_report(names, direction_map, family_map, per_seed_curves, corr,
                 densities, decomposition, seed_accs, out_dir, kan_res, lr_res,
                 fused=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)
    C = len(names)
    G = per_seed_curves.shape[2]
    xs = np.linspace(0, 1, G)
    mean_c = per_seed_curves.mean(0)                 # [C,G]
    lo_c = np.percentile(per_seed_curves, 10, axis=0)
    hi_c = np.percentile(per_seed_curves, 90, axis=0)
    swing = mean_c.max(1) - mean_c.min(1)            # [C]
    swing_std = (per_seed_curves.max(2) - per_seed_curves.min(2)).std(0)
    order = np.argsort(swing)[::-1]

    def trend(i):
        c = mean_c[i]
        if direction_map[names[i]] == 0 and (np.sign(np.diff(c)).sum() == 0 or
                                             (np.diff(c) > 0).any() and (np.diff(c) < 0).any()):
            return "non-monotonic"
        return "up" if c[-1] > c[0] else "down"

    # Figure A: grouped importance bar
    top = order[:18][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(range(len(top)), swing[top], xerr=swing_std[top],
            color=[FAMILY_COLOR[family_map[names[i]]] for i in top],
            error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([names[i].replace("_", " ") for i in top])
    ax.set_xlabel("sentiment swing (learned importance)  ·  error bar = ±1 std over seeds")
    ax.set_title("Concept importance — additive Concept-Bottleneck KAN")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ["#1baf7a", "#eb6834", "#2a78d6"]]
    ax.legend(handles, ["positive family", "negative family", "context (unconstrained)"],
              loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(plots / "concept_importance.png", dpi=150); plt.close(fig)

    # Figure B: spline curves w/ CI band + data density
    tp = order[:12]
    ncol, nrow = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow))
    for ax, i in zip(axes.ravel(), tp):
        col = FAMILY_COLOR[family_map[names[i]]]
        ax.fill_between(xs, lo_c[i], hi_c[i], color=col, alpha=0.2)
        ax.plot(xs, mean_c[i], color=col, lw=2)
        ax.axhline(0, color="gray", lw=0.6)
        dax = ax.twinx()
        dax.fill_between(np.linspace(0, 1, len(densities[i])), 0, densities[i],
                         color="gray", alpha=0.12, step="mid")
        dax.set_yticks([])
        ax.set_title(f"{names[i].replace('_', ' ')}  (swing {swing[i]:.2f}, {trend(i)})",
                     fontsize=9)
        ax.set_xlabel("concept intensity"); ax.set_ylabel("sentiment effect")
    for ax in axes.ravel()[len(tp):]:
        ax.axis("off")
    fig.suptitle("Learned spline per concept  (shaded = 10–90% band over seeds, "
                 "grey = data density; up = more positive)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plots / "concept_curves.png", dpi=150); plt.close(fig)

    # Figure C: per-prediction decomposition (waterfall-ish)
    if decomposition:
        n = len(decomposition)
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2))
        if n == 1:
            axes = [axes]
        for ax, ex in zip(axes, decomposition):
            contribs = sorted(ex["contributions"], key=lambda t: abs(t[1]), reverse=True)[:8]
            labels = [c[0].replace("_", " ") for c in contribs][::-1]
            vals = [c[1] for c in contribs][::-1]
            ax.barh(range(len(vals)), vals,
                    color=["#1baf7a" if v > 0 else "#eb6834" for v in vals])
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color="k", lw=0.6)
            gold = "pos" if ex["label"] == 1 else "neg"
            pred = "pos" if ex["logit"] > 0 else "neg"
            ax.set_title(f"logit={ex['logit']:+.2f} (pred {pred}, true {gold})\n"
                         f"“{ex['text'][:60]}”", fontsize=8)
            ax.set_xlabel("spline contribution")
        fig.suptitle("Per-prediction decomposition: bias + Σ concept contributions", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(plots / "prediction_decomposition.png", dpi=150); plt.close(fig)

    # Concept table (md + csv)
    rows = [("concept", "family", "constraint", "mean_swing", "swing_std",
             "corr_label", "trend", "reading")]
    dir_txt = {1: "↑ (forced up)", -1: "↓ (forced down)", 0: "free"}
    for i in order:
        n = names[i]
        tr = trend(i)
        read = ("more positive sentiment" if tr == "up" else
                "more negative sentiment" if tr == "down" else
                "non-monotonic (context-dependent)")
        rows.append((n, family_map[n], dir_txt[direction_map[n]],
                     f"{swing[i]:.3f}", f"{swing_std[i]:.3f}", f"{corr[n]:+.3f}",
                     tr, f"higher {n.replace('_',' ')} → {read}"))
    (out_dir / "concept_table.csv").write_text(
        "\n".join(",".join(r) for r in rows), encoding="utf-8")
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    (out_dir / "concept_table.md").write_text("\n".join(md), encoding="utf-8")

    payload = {
        "note_sentiment140_only": (
            "Concept scores are zero-shot cosine intensities in the encoder space; "
            "Sentiment140 has no concept labels, so supervised concept-extractor "
            "training and concept-level F1/AUROC/Brier/ECE/calibration are OUT OF "
            "SCOPE (not computed, not estimated)."),
        "sentiment_results": {"logreg_on_concepts": lr_res, "additive_kan": kan_res,
                              "kan_seed_val_accs": seed_accs},
        "fusion": fused,
        "concepts": names,
        "importance_ranked": [names[i] for i in order],
        "swing": {names[i]: float(swing[i]) for i in range(C)},
        "swing_std": {names[i]: float(swing_std[i]) for i in range(C)},
        "corr_with_label": corr,
        "trend": {names[i]: trend(i) for i in range(C)},
        "curves_mean": {names[i]: mean_c[i].tolist() for i in range(C)},
        "curves_ci_low": {names[i]: lo_c[i].tolist() for i in range(C)},
        "curves_ci_high": {names[i]: hi_c[i].tolist() for i in range(C)},
    }
    (out_dir / "concept_kan_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] report + figures written to {out_dir.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", default="data/embeddings_finetuned")
    ap.add_argument("--encoder", default="models/deberta_finetuned/best")
    ap.add_argument("--test-csv", default="data/splitsfull/test.csv")
    ap.add_argument("--output-dir", default="outputs/concept_kan_report")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-eval", type=int, default=5000)
    ap.add_argument("--grid-size", type=int, default=6)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--l-smooth", type=float, default=1e-3)
    ap.add_argument("--l-sparse", type=float, default=1e-4)
    ap.add_argument("--l-mono", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--export-fusion-probs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--smoke", action="store_true", help="Tiny fast self-check.")
    args = ap.parse_args()

    if args.smoke:
        args.max_train, args.max_eval, args.epochs, args.seeds = 2000, 800, 4, 2
        args.export_fusion_probs = False

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    names = list(CONCEPTS)
    direction_map = {n: CONCEPTS[n][1] for n in names}
    family_map = {n: CONCEPTS[n][2] for n in names}
    print(f"[INFO] Additive Concept KAN | {len(names)} concepts | device={device}")

    # 1. anchors -------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder, dtype=torch.float32).to(device)
    anchors = np.stack([embed_texts(enc, tok, CONCEPTS[n][0], device).mean(0) for n in names])
    anchors = (anchors / np.linalg.norm(anchors, axis=1, keepdims=True)).astype(np.float32)
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. concept intensities per split --------------------------------------
    raw = {s: load_split(Path(args.embeddings_dir), s) for s in SPLITS}
    sub = {}
    for s in SPLITS:
        cap = args.max_train if s == "train" else args.max_eval
        X, y = stratified_subsample(*raw[s], cap, args.seed)
        sub[s] = (concept_scores(X, anchors), y)   # cosine
    lo, hi = fit_intensity(sub["train"][0])
    data = {s: (to_intensity(c, lo, hi), y) for s, (c, y) in sub.items()}
    Xtr, ytr = data["train"]
    corr = {names[i]: float(np.corrcoef(Xtr[:, i], ytr)[0, 1]) for i in range(len(names))}
    densities = [np.histogram(Xtr[:, i], bins=20, range=(0, 1))[0].astype(float)
                 for i in range(len(names))]
    densities = [d / d.max() if d.max() > 0 else d for d in densities]

    # 3. linear baseline on concepts ----------------------------------------
    lr_res = run_logreg({s: (data[s][0], data[s][1]) for s in SPLITS}, args.seed)

    # 4. additive KAN over multiple seeds -----------------------------------
    direction = torch.tensor([direction_map[n] for n in names], dtype=torch.float32, device=device)
    per_seed_curves, seed_accs, models = [], [], []
    for k in range(args.seeds):
        set_seed(args.seed + k)
        m = AdditiveConceptKAN(len(names), args.grid_size, args.spline_order).to(device)
        va = train_one(m, Xtr, ytr, data["val"][0], data["val"][1], direction, device,
                       epochs=args.epochs, batch=args.batch_size, lr=args.lr,
                       wd=args.weight_decay, patience=args.patience, seed=args.seed + k,
                       l_smooth=args.l_smooth, l_sparse=args.l_sparse, l_mono=args.l_mono)
        seed_accs.append(va)
        per_seed_curves.append(curve_effect(m, len(names), device))
        models.append((va, m))
        print(f"  [seed {k}] val_acc={va:.4f}")
    per_seed_curves = np.stack(per_seed_curves)   # [S,C,G]
    best_model = max(models, key=lambda t: t[0])[1]
    kan_res = {s: scores(data[s][1],
               (forward_logits(best_model, data[s][0], device).numpy() > 0).astype(int))
               for s in SPLITS}
    acc_mean = float(np.mean([kan_res[s]["accuracy"] for s in ("val",)]))
    print(f"[INFO] additive KAN val acc mean={np.mean(seed_accs):.4f} std={np.std(seed_accs):.4f} "
          f"| best test acc={kan_res['test']['accuracy']:.4f}")

    # 5. per-prediction decomposition on a few test tweets ------------------
    decomposition = []
    try:
        import pandas as pd
        df = pd.read_csv(args.test_csv, usecols=["text", "label"])
        te_cos, te_y = concept_scores(raw["test"][0], anchors), raw["test"][1]
        te_x = to_intensity(te_cos, lo, hi)
        if np.array_equal(df["label"].to_numpy(), te_y):
            base_vec = np.full((1, len(names)), 0.5, np.float32)
            ref = float(forward_logits(best_model, base_vec, device).numpy()[0])
            # pick a confident pos, a confident neg, and a sarcasm-cued tweet
            logits_all = forward_logits(best_model, te_x, device).numpy()
            picks = [int(np.argmax(logits_all)), int(np.argmin(logits_all))]
            cue = df["text"].astype(str).str.lower().str.contains(
                r"yeah right|oh great|how wonderful|/s\b|so much fun", regex=True, na=False).to_numpy()
            if cue.any():
                picks.append(int(np.flatnonzero(cue)[0]))
            for idx in picks:
                xi = te_x[idx]
                contribs = []
                for i, nm in enumerate(names):
                    v = base_vec.copy(); v[0, i] = xi[i]
                    ci = float(forward_logits(best_model, v, device).numpy()[0]) - ref
                    contribs.append((nm, ci))
                decomposition.append({"text": str(df["text"].iloc[idx]),
                                      "label": int(te_y[idx]), "logit": float(logits_all[idx]),
                                      "reference": ref, "contributions": contribs})
    except Exception as e:
        print(f"[WARN] decomposition skipped: {e}")

    # 6. export FULL probs for fusion ---------------------------------------
    fused = None
    if args.export_fusion_probs:
        for s in SPLITS:
            Xe = np.load(Path(args.embeddings_dir) / f"{s}_embeddings.npy").astype(np.float32)
            cos = concept_scores(Xe, anchors)
            np.save(out / f"{s}_probs.npy", probs_full(best_model, cos, lo, hi, device))
        print(f"[INFO] exported full {'/'.join(SPLITS)} probs for fusion -> {out}")

    build_report(names, direction_map, family_map, per_seed_curves, corr, densities,
                 decomposition, seed_accs, out, kan_res, lr_res, fused)

    print("\n" + "=" * 60)
    print("SENTIMENT ACCURACY (Sentiment140)")
    print("=" * 60)
    print(f"  logreg on concepts   val={lr_res['val']['accuracy']:.4f} test={lr_res['test']['accuracy']:.4f}")
    print(f"  additive concept KAN val={kan_res['val']['accuracy']:.4f} test={kan_res['test']['accuracy']:.4f} "
          f"(seed val mean {np.mean(seed_accs):.4f} ± {np.std(seed_accs):.4f})")
    print(f"\n[DONE] {out.resolve()}")


if __name__ == "__main__":
    main()

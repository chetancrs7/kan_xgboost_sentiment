"""Diagnostic: is the KAN or the embedding the bottleneck?

Trains THREE classifiers on *exactly the same* precomputed embeddings and the
*same* train/val/test split:

    1. Logistic Regression  (linear probe -- how linearly separable the
       embeddings already are)
    2. Small MLP            (a plain feed-forward net of comparable capacity)
    3. KAN                  (the project's real ``KANSentimentClassifier``)

All three see the identical StandardScaler-normalised features, so "input
scaling" is held constant across models and cannot explain a gap on its own.

This script is deliberately *standalone and read-only*: it never re-runs any
embedding model, never touches the shared training code paths, only reads the
``.npy`` arrays already on disk, and writes its results to a fresh output dir.
So it is safe to run alongside another training job (it just adds CPU load --
keep --max-train modest if a real run is in progress).

Interpretation (printed again at the end):

  * All three poor            -> embeddings / preprocessing / labels / dataset
                                 are the bottleneck.
  * MLP good, KAN poor        -> KAN architecture / input scaling / spline
                                 config is the bottleneck.
  * KAN train high, val low   -> KAN is overfitting.
  * KAN train AND val low     -> KAN underfitting, or embeddings are not
                                 sentiment-discriminative.

Usage:
    python scripts/diagnose_kan_vs_embeddings.py
    python scripts/diagnose_kan_vs_embeddings.py --embeddings-dir data/embeddings_bertweet
    python scripts/diagnose_kan_vs_embeddings.py --max-train 0   # use ALL rows
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Make the project root importable so we can reuse the real KAN model.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kan.model import DEFAULT_HIDDEN_DIMS, KANSentimentClassifier  # noqa: E402

DEFAULT_EMBEDDINGS_DIR = "data/embeddings/deberta_v3_base"
DEFAULT_OUTPUT_DIR = "outputs/diagnostic_probe"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(embeddings_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    xp = embeddings_dir / f"{split}_embeddings.npy"
    yp = embeddings_dir / f"{split}_labels.npy"
    for p in (xp, yp):
        if not p.exists():
            sys.exit(f"[ERROR] Missing {p} -- generate embeddings first.")
    X = np.load(xp).astype(np.float32)
    y = np.load(yp).astype(np.int64)
    if len(X) != len(y):
        sys.exit(f"[ERROR] {split}: {len(X)} embeddings vs {len(y)} labels.")
    return X, y


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, cap: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cap a split to ``cap`` rows, keeping class proportions. cap<=0 -> all."""
    if cap <= 0 or len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = max(1, round(cap * len(cls_idx) / len(y)))
        take = min(take, len(cls_idx))
        idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def standardize(
    splits: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit StandardScaler on TRAIN only, apply to all splits (no leakage)."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(splits["train"][0])
    return {
        s: (scaler.transform(X).astype(np.float32), y) for s, (X, y) in splits.items()
    }


def reduce_pca(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], n_components: int, seed: int
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Fit PCA on TRAIN only (whitened), apply unchanged to val/test.

    Whitening gives unit-variance components, which land cleanly inside the
    KAN's spline grid range. Returns the reduced splits and the fraction of
    variance retained. No leakage: val/test never touch PCA fitting.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    pca.fit(splits["train"][0])
    retained = float(pca.explained_variance_ratio_.sum())
    reduced = {
        s: (pca.transform(X).astype(np.float32), y) for s, (X, y) in splits.items()
    }
    return reduced, retained


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


# --------------------------------------------------------------------------- #
# Model 1: Logistic Regression
# --------------------------------------------------------------------------- #
def run_logreg(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], seed: int
) -> dict[str, dict]:
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr = splits["train"]
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    out = {}
    for s in SPLITS:
        X, y = splits[s]
        out[s] = scores(y, clf.predict(X))
    return out


# --------------------------------------------------------------------------- #
# Model 2: Small MLP
# --------------------------------------------------------------------------- #
class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden=(256, 64), dropout=0.3):
        super().__init__()
        dims = [input_dim, *hidden]
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        blocks.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Shared torch training loop (used by BOTH the MLP and the KAN so the two are
# compared on an identical budget: same epochs, batch size, optimizer, early
# stopping). The only difference is the module itself.
# --------------------------------------------------------------------------- #
def _loader(X, y, batch_size, shuffle, seed):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None
    )


@torch.no_grad()
def _predict(model, X, device, batch_size=1024):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def _val_loss(model, loader, criterion, device):
    """Return (mean val loss, val accuracy) over ``loader``."""
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        tot += criterion(logits, yb).item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return tot / max(1, n), correct / max(1, n)


def train_torch(
    model: nn.Module,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    clip_norm: float = 0.0,
    grid_seed_fn=None,
    grid_update_freq: int = 0,
    label: str = "model",
) -> tuple[dict[str, dict], dict]:
    """Train ``model`` with early stopping on val loss; return metrics + curve.

    ``grid_seed_fn(model)`` (KAN only) seeds/refits spline grids from the
    training data; ``grid_update_freq`` re-seeds every N epochs (mirrors the
    real KAN trainer). Both are no-ops for the MLP.
    """
    Xtr, ytr = splits["train"]
    train_loader = _loader(Xtr, ytr, batch_size, True, seed)
    val_loader = _loader(*splits["val"], batch_size, False, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if grid_seed_fn is not None:
        grid_seed_fn(model)

    best_val, best_state, wait, stopped = float("inf"), copy.deepcopy(model.state_dict()), 0, epochs
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            bs = yb.size(0)
            tot_loss += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
            n += bs

        train_loss, train_acc = tot_loss / n, correct / n
        val_loss, val_acc = _val_loss(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if grid_seed_fn is not None and grid_update_freq and epoch % grid_update_freq == 0 and epoch < epochs:
            grid_seed_fn(model)

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
            marker = "  * best"
        else:
            wait += 1
        print(
            f"  [{label} ep {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}"
        )
        if wait >= patience:
            stopped = epoch
            print(f"  [{label}] early stop at epoch {epoch} (patience {patience}).")
            break

    model.load_state_dict(best_state)
    out = {}
    for s in SPLITS:
        X, y = splits[s]
        out[s] = scores(y, _predict(model, X, device))
    history["stopped_epoch"] = stopped
    return out, history


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def diagnose(results: dict[str, dict], histories: dict[str, dict] | None = None) -> str:
    """Turn the three models' accuracies into the user's interpretation table.

    Uses the training *history* (not just the final best-val checkpoint) to
    detect overfitting: a KAN can early-stop with a small final train/val gap
    yet have blown past it mid-training (peak train acc high, val loss
    diverging). That trajectory is the real overfitting signal.
    """
    histories = histories or {}
    lr_val = results["logreg"]["val"]["accuracy"]
    mlp_val = results["mlp"]["val"]["accuracy"]
    kan_tr = results["kan"]["train"]["accuracy"]
    kan_val = results["kan"]["val"]["accuracy"]

    # Peak training accuracy the KAN actually reached (memorisation ceiling),
    # and whether its val loss diverged from its best -- both from history.
    kh = histories.get("kan", {})
    kan_peak_tr = max(kh.get("train_acc", [kan_tr]), default=kan_tr)
    kan_vl = kh.get("val_loss", [])
    kan_vl_diverged = bool(kan_vl) and (max(kan_vl) - min(kan_vl) > 0.15)

    # "poor" = not much better than a coin flip on this (near-balanced) task.
    poor = 0.60
    good = 0.65
    overfit_gap = 0.10

    lines = []
    # Overfitting is checked FIRST and against the training trajectory, so an
    # early-stopped KAN whose final gap looks small is still flagged.
    kan_overfit = (kan_peak_tr - kan_val >= overfit_gap and kan_peak_tr >= good) or (
        kan_vl_diverged and kan_val < max(lr_val, mlp_val)
    )

    if lr_val < poor and mlp_val < poor and kan_val < poor:
        lines.append(
            "=> ALL THREE PERFORM POORLY. The bottleneck is upstream of the "
            "classifier: the embeddings, preprocessing, labels, or dataset. "
            "The KAN is not the problem -- these features are not "
            "sentiment-discriminative as fed."
        )
    elif mlp_val >= good and kan_val < poor:
        lines.append(
            "=> MLP GOOD, KAN POOR. The embeddings carry the signal (the MLP "
            "recovers it) but the KAN does not. Bottleneck is the KAN itself: "
            "architecture, input scaling into the spline grids, or spline "
            "configuration (grid_size / spline_order / grid_range)."
        )
    elif kan_overfit:
        lines.append(
            f"=> KAN OVERFITTING. Peak train acc {kan_peak_tr:.3f} vs val acc "
            f"{kan_val:.3f} (gap {kan_peak_tr - kan_val:.3f})"
            + (", and val loss diverged from its best" if kan_vl_diverged else "")
            + ". The embeddings are fine (logreg/MLP reach "
            f"{max(lr_val, mlp_val):.3f} val); the KAN memorises the train set "
            "and generalises worse than a linear probe. Regularise: more weight "
            "decay / dropout, smaller grid_size, fewer hidden dims, lower LR, or "
            "more training data."
        )
    elif kan_tr < good and kan_val < good:
        lines.append(
            f"=> KAN UNDERFITTING (train {kan_tr:.3f}, val {kan_val:.3f} both "
            "low). Either the KAN lacks capacity / is under-trained (raise "
            "epochs, lr, grid_size), or -- if the MLP is also weak -- the "
            "embeddings are not sentiment-discriminative."
        )
    else:
        lines.append(
            "=> MIXED / HEALTHY. No single clean pattern; compare the numbers "
            "in the table above against the four reference cases."
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR,
                    help="Same dir the KAN trains on (default: %(default)s).")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--max-train", type=int, default=20000,
                    help="Cap train rows (stratified) for speed. 0 = use all. Default 20000.")
    ap.add_argument("--max-eval", type=int, default=5000,
                    help="Cap val/test rows (stratified). 0 = use all. Default 5000.")
    ap.add_argument("--no-standardize", action="store_true",
                    help="Feed raw embeddings instead of StandardScaler-normalised.")
    ap.add_argument("--pca", type=int, default=0,
                    help="Reduce embeddings to N PCA components (fit on train only, "
                    "whitened) before the classifiers. 0 = disabled (full 768-dim).")
    # KAN / MLP shared budget -- matched to the real KAN trainer defaults.
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--hidden-dims", default=",".join(str(d) for d in DEFAULT_HIDDEN_DIMS),
                    help="KAN layer 1-3 widths, e.g. '256,128,64'.")
    ap.add_argument("--grid-update-freq", type=int, default=10)
    ap.add_argument("--kan-clip-norm", type=float, default=1.0,
                    help="Grad clip for the KAN (TPS layer stability). Default 1.0.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: auto).")
    ap.add_argument("--skip-kan", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    emb_dir = Path(args.embeddings_dir)
    print(f"[INFO] Embeddings: {emb_dir}  | device: {device}")

    # 1. Load + (optionally) subsample the SAME split for every model.
    raw = {s: load_split(emb_dir, s) for s in SPLITS}
    splits = {}
    for s in SPLITS:
        cap = args.max_train if s == "train" else args.max_eval
        X, y = stratified_subsample(*raw[s], cap, args.seed)
        splits[s] = (X, y)
        pos = int((y == 1).sum())
        print(f"[INFO] {s}: {len(X):,} rows (pos={pos}, neg={len(y) - pos})")

    input_dim = splits["train"][0].shape[1]
    n_classes = int(len(np.unique(np.concatenate([splits[s][1] for s in SPLITS]))))
    print(f"[INFO] input_dim={input_dim} n_classes={n_classes} "
          f"standardize={not args.no_standardize}")

    if not args.no_standardize:
        splits = standardize(splits)

    pca_retained = None
    if args.pca and args.pca > 0:
        if args.pca >= input_dim:
            sys.exit(f"[ERROR] --pca {args.pca} must be < input_dim {input_dim}.")
        splits, pca_retained = reduce_pca(splits, args.pca, args.seed)
        input_dim = args.pca
        print(f"[INFO] PCA -> {args.pca} dims (train-fit, whitened); "
              f"retained {pca_retained:.1%} of variance. New input_dim={input_dim}.")

    results: dict[str, dict] = {}
    histories: dict[str, dict] = {}

    # 2. Logistic Regression (linear probe).
    print("\n" + "=" * 64 + "\n[1/3] LOGISTIC REGRESSION\n" + "=" * 64)
    results["logreg"] = run_logreg(splits, args.seed)
    print(f"  val acc={results['logreg']['val']['accuracy']:.4f}")

    # 3. Small MLP.
    print("\n" + "=" * 64 + "\n[2/3] SMALL MLP\n" + "=" * 64)
    set_seed(args.seed)
    mlp = SmallMLP(input_dim, n_classes).to(device)
    print(f"  params={sum(p.numel() for p in mlp.parameters()):,}")
    results["mlp"], histories["mlp"] = train_torch(
        mlp, splits, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, seed=args.seed,
        label="MLP",
    )

    # 4. KAN (the real project model).
    if not args.skip_kan:
        print("\n" + "=" * 64 + "\n[3/3] KAN (KANSentimentClassifier)\n" + "=" * 64)
        set_seed(args.seed)
        hidden = tuple(int(d) for d in args.hidden_dims.split(","))
        kan = KANSentimentClassifier(
            input_dim=input_dim, hidden_dims=hidden, n_classes=n_classes,
            grid_size=args.grid_size, spline_order=args.spline_order,
        ).to(device)
        print(f"  params={sum(p.numel() for p in kan.parameters()):,}")

        Xtr = splits["train"][0]

        def seed_grids(model):
            batch = torch.from_numpy(Xtr[: min(len(Xtr), 512)]).to(device)
            model.update_grids(batch)

        results["kan"], histories["kan"] = train_torch(
            kan, splits, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience, seed=args.seed,
            clip_norm=args.kan_clip_norm, grid_seed_fn=seed_grids,
            grid_update_freq=args.grid_update_freq, label="KAN",
        )

    # 5. Report.
    print("\n" + "=" * 72)
    print("RESULTS  (accuracy / macro-F1)")
    print("=" * 72)
    header = f"{'model':<10}" + "".join(f"{s:>22}" for s in SPLITS)
    print(header)
    for name in ("logreg", "mlp", "kan"):
        if name not in results:
            continue
        row = f"{name:<10}"
        for s in SPLITS:
            m = results[name][s]
            row += f"{m['accuracy']:.4f}/{m['macro_f1']:.4f}".rjust(22)
        print(row)

    print("\n" + "=" * 72 + "\nDIAGNOSIS\n" + "=" * 72)
    if "kan" in results:
        print(diagnose(results, histories))
    else:
        print("KAN skipped (--skip-kan); no KAN diagnosis.")

    print(
        "\nReference cases:\n"
        "  * all three poor           -> embeddings/preprocessing/labels/dataset\n"
        "  * MLP good, KAN poor        -> KAN architecture / scaling / spline config\n"
        "  * KAN train high, val low   -> KAN overfitting\n"
        "  * KAN train & val both low  -> KAN underfitting or embeddings weak"
    )

    # 6. Persist.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings_dir": str(emb_dir),
        "config": {
            "max_train": args.max_train, "max_eval": args.max_eval,
            "standardize": not args.no_standardize,
            "pca": args.pca or None, "pca_variance_retained": pca_retained,
            "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "grid_size": args.grid_size,
            "spline_order": args.spline_order, "hidden_dims": args.hidden_dims,
        },
        "split_sizes": {s: len(splits[s][0]) for s in SPLITS},
        "results": results,
        "histories": histories,
        "diagnosis": diagnose(results, histories) if "kan" in results else None,
    }
    (out_dir / "diagnostic_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[INFO] Saved results to {(out_dir / 'diagnostic_results.json').resolve()}")


if __name__ == "__main__":
    main()

"""Residual Bottleneck KAN -- standalone experiment.

Tests the "Residual Bottleneck KAN" architecture the user proposed, WITHOUT
touching any existing code: it only imports the project's KAN spline layer as a
reusable block and reads precomputed embeddings.

Architecture (the proposed head, fed precomputed 768-d embeddings)::

    x  (embedding, 768)
      -> LayerNorm(768)
      -> Linear 768 -> 256 -> GELU -> Dropout(0.2) -> Linear 256 -> 64
      -> LayerNorm(64)                                   = z
      -> z' = z + gamma * KAN_64_64(z)   (gamma learnable, init 0.1)   [x n_blocks]
      -> LayerNorm(64)
      -> KAN_64_classes(z')  -> logits

Faithfulness note: the original diagram also concatenates a CLS token vector and
a masked-mean vector into 1536 dims and *unfreezes 2 BERT layers*. Both need
raw/token-level BERT, which we don't re-run here; instead we feed the 768-d
pooled embedding (the fine-tuned set already encodes an unfrozen BERT). So the
projection starts at 768, not 1536 -- everything downstream is exactly as drawn.

The residual design's point: with a small gamma the model starts as a stable
linear projection (~a linear probe) and the KAN only adds *nonlinear
corrections* as gamma/coefs grow -- so the val-accuracy curve should rise
gradually from the linear baseline rather than snapping to a plateau in epoch 1.

Usage:
    python scripts/residual_bottleneck_kan.py --embeddings-dir data/embeddings_finetuned
    python scripts/residual_bottleneck_kan.py --embeddings-dir data/embeddings/deberta_v3_base
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
# Reuse the diagnostic script's data + ceiling helpers (our own module).
from scripts.diagnose_kan_vs_embeddings import (  # noqa: E402
    SPLITS,
    load_split,
    run_logreg,
    scores,
    set_seed,
    standardize,
    stratified_subsample,
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResidualKANBlock(nn.Module):
    """z' = z + gamma * KAN(z), with a learnable scalar gamma (init small)."""

    def __init__(self, dim: int, grid_size: int, spline_order: int, gamma_init: float):
        super().__init__()
        self.kan = AdaptiveSplineKANLayer(dim, dim, grid_size, spline_order)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.gamma * self.kan(z)


class ResidualBottleneckKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        proj_hidden: int = 256,
        bottleneck: int = 64,
        grid_size: int = 3,
        spline_order: int = 3,
        dropout: float = 0.2,
        gamma_init: float = 0.1,
        n_blocks: int = 1,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, bottleneck),
        )
        self.post_proj_norm = nn.LayerNorm(bottleneck)
        self.blocks = nn.ModuleList(
            [ResidualKANBlock(bottleneck, grid_size, spline_order, gamma_init) for _ in range(n_blocks)]
        )
        self.block_norms = nn.ModuleList([nn.LayerNorm(bottleneck) for _ in range(n_blocks)])
        self.classifier = AdaptiveSplineKANLayer(bottleneck, n_classes, grid_size, spline_order)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_proj_norm(self.proj(self.in_norm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._project(x)
        for blk, norm in zip(self.blocks, self.block_norms):
            z = norm(blk(z))
        return self.classifier(z)

    @torch.no_grad()
    def seed_grids(self, x: torch.Tensor) -> None:
        """Seed every KAN layer's spline grid from real activations."""
        was_training = self.training
        self.eval()
        z = self._project(x)
        for blk, norm in zip(self.blocks, self.block_norms):
            blk.kan.adapt_grid(z)
            z = norm(blk(z))
        self.classifier.adapt_grid(z)
        if was_training:
            self.train()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _loader(X, y, batch_size, shuffle, seed):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


@torch.no_grad()
def _eval(model, X, y, device, batch_size=2048):
    model.eval()
    preds, loss_sum, n = [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    yt = torch.from_numpy(y)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        yb = yt[i : i + batch_size].to(device)
        logits = model(xb)
        loss_sum += crit(logits, yb).item()
        preds.append(logits.argmax(1).cpu().numpy())
        n += len(yb)
    pred = np.concatenate(preds)
    return loss_sum / n, float((pred == y).mean())


def train(model, splits, device, *, epochs, batch_size, lr, weight_decay, patience,
          seed, clip_norm, grid_update_freq, label="ResKAN"):
    Xtr, ytr = splits["train"]
    Xval, yval = splits["val"]
    loader = _loader(Xtr, ytr, batch_size, True, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    seed_batch = torch.from_numpy(Xtr[: min(len(Xtr), 512)]).to(device)
    model.seed_grids(seed_batch)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "gamma": []}
    best_val, best_state, wait, stopped = float("inf"), copy.deepcopy(model.state_dict()), 0, epochs

    for epoch in range(1, epochs + 1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            bs = yb.size(0)
            tot += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
            n += bs

        train_loss, train_acc = tot / n, correct / n
        val_loss, val_acc = _eval(model, Xval, yval, device)
        gamma = float(model.blocks[0].gamma.detach().cpu())
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["gamma"].append(gamma)

        if grid_update_freq and epoch % grid_update_freq == 0 and epoch < epochs:
            model.seed_grids(seed_batch)

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
            marker = "  * best"
        else:
            wait += 1
        print(f"  [{label} ep {epoch:03d}/{epochs}] train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} gamma={gamma:+.3f}{marker}")
        if wait >= patience:
            stopped = epoch
            print(f"  [{label}] early stop at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    out = {s: scores(splits[s][1], _eval_pred(model, splits[s][0], device)) for s in SPLITS}
    history["stopped_epoch"] = stopped
    return out, history


@torch.no_grad()
def _eval_pred(model, X, device, batch_size=2048):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", default="data/embeddings_finetuned")
    ap.add_argument("--output-dir", default="outputs/residual_bottleneck_kan")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-eval", type=int, default=5000)
    ap.add_argument("--proj-hidden", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--grid-size", type=int, default=3)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gamma-init", type=float, default=0.1)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--grid-update-freq", type=int, default=10)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = Path(args.embeddings_dir)
    print(f"[INFO] Residual Bottleneck KAN | embeddings={emb} | device={device}")

    splits = {}
    for s in SPLITS:
        cap = args.max_train if s == "train" else args.max_eval
        X, y = stratified_subsample(*load_split(emb, s), cap, args.seed)
        splits[s] = (X, y)
    input_dim = splits["train"][0].shape[1]
    n_classes = int(len(np.unique(np.concatenate([splits[s][1] for s in SPLITS]))))
    print(f"[INFO] input_dim={input_dim} n_classes={n_classes} "
          f"train={len(splits['train'][0])} val={len(splits['val'][0])}")

    splits = standardize(splits)

    # Linear-probe ceiling on the identical features, for the plateau question.
    print("\n[ceiling] logistic regression ...")
    lr_scores = run_logreg(splits, args.seed)
    print(f"  logreg val acc = {lr_scores['val']['accuracy']:.4f}")

    print("\n[Residual Bottleneck KAN] training ...")
    set_seed(args.seed)
    model = ResidualBottleneckKAN(
        input_dim, n_classes, args.proj_hidden, args.bottleneck,
        args.grid_size, args.spline_order, args.dropout, args.gamma_init, args.n_blocks,
    ).to(device)
    print(f"  params={sum(p.numel() for p in model.parameters()):,} "
          f"(bottleneck={args.bottleneck}, grid={args.grid_size}, blocks={args.n_blocks})")
    res, hist = train(
        model, splits, device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
        seed=args.seed, clip_norm=args.clip_norm, grid_update_freq=args.grid_update_freq,
    )

    va = hist["val_acc"]
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  logreg (linear ceiling) val = {lr_scores['val']['accuracy']:.4f}")
    for s in SPLITS:
        print(f"  ResKAN {s:<5} acc={res[s]['accuracy']:.4f} f1={res[s]['macro_f1']:.4f}")
    print(f"\n  val_acc first(ep1)={va[0]:.4f}  last(ep{len(va)})={va[-1]:.4f}  "
          f"max={max(va):.4f}  min={min(va):.4f}")
    print(f"  gamma  first={hist['gamma'][0]:+.3f} -> last={hist['gamma'][-1]:+.3f}")
    print(f"  per-epoch val_acc: {[round(v, 4) for v in va]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings_dir": str(emb), "config": vars(args),
        "logreg_val": lr_scores["val"]["accuracy"], "results": res, "history": hist,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[INFO] Saved to {(out_dir / 'results.json').resolve()}")


if __name__ == "__main__":
    main()

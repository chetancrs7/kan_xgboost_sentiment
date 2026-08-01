"""SHAP explanations for the XGBoost structural expert.

Uses :class:`shap.TreeExplainer`, which computes **exact** Shapley values for
gradient-boosted trees (no sampling, no surrogate model). For the binary
sentiment classifier the Shapley values are in the model's log-odds space:
positive value = pushes the prediction toward the positive class.

Outputs (written under ``--output-dir``):

* ``shap_summary_beeswarm.png``  per-feature value + impact distribution
* ``shap_global_importance.png`` mean |SHAP| bar chart (global ranking)
* ``shap_vs_gain_importance.png`` SHAP ranking vs XGBoost's built-in gain
* ``shap_waterfall_*.png``        local explanations for a few test tweets
* ``shap_global_importance.json`` machine-readable global ranking
* ``shap_feature_importance.csv`` mean |SHAP| per feature

Run::

    python -m src.explainability.shap_xgboost \
        --model-dir models/xgboost --split-path data/splits/test.csv --n-samples 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from src.explainability import require_shap  # noqa: E402
from src.models.xgboost.data_loader import load_split  # noqa: E402


def load_expert(model_dir: Path):
    """Load the trained XGBoost model and its fitted feature extractor."""
    model = joblib.load(model_dir / "xgboost_model.pkl")
    extractor = joblib.load(model_dir / "feature_extractor.pkl")
    return model, extractor


def compute_shap_values(model, X: np.ndarray):
    """Return ``(explainer, shap_values, expected_value)`` for a feature matrix.

    ``shap_values`` is ``[n_samples, n_features]`` in log-odds space; for the
    binary objective TreeExplainer returns the positive-class attribution.
    """
    shap = require_shap()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Some SHAP/XGBoost combinations return a per-class list for binary models;
    # keep the positive-class attributions so the sign is interpretable.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.ravel(expected)[-1])
    return explainer, np.asarray(shap_values), float(expected)


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Rank features by mean absolute SHAP value (global importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {"rank": r + 1, "feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for r, i in enumerate(order)
    ]


def plot_global_bar(ranking: list[dict], path: Path, top_k: int = 20) -> None:
    """Horizontal bar chart of the top-``k`` features by mean |SHAP|."""
    top = ranking[:top_k][::-1]
    names = [r["feature"] for r in top]
    vals = [r["mean_abs_shap"] for r in top]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(top))))
    ax.barh(names, vals, color="#4878d0", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("mean |SHAP value|  (impact on model output, log-odds)")
    ax.set_title(f"XGBoost expert — global feature importance (top {len(top)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_vs_gain(
    ranking: list[dict], model, feature_names: list[str], path: Path, top_k: int = 15
) -> None:
    """Compare SHAP importance against XGBoost's built-in gain importance."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    # XGBoost keys features as f0, f1, ...; map back to names by position.
    gain_by_name = {feature_names[int(k[1:])]: v for k, v in gain.items() if k[1:].isdigit()}

    top = ranking[:top_k]
    names = [r["feature"] for r in top]
    shap_imp = np.array([r["mean_abs_shap"] for r in top])
    gain_imp = np.array([gain_by_name.get(n, 0.0) for n in names])
    # Normalise each to [0, 1] so the two scales can share an axis.
    shap_n = shap_imp / (shap_imp.max() or 1.0)
    gain_n = gain_imp / (gain_imp.max() or 1.0)

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.4 * len(names))))
    ax.barh(y + 0.2, shap_n, height=0.4, label="SHAP (mean |value|)", color="#4878d0")
    ax.barh(y - 0.2, gain_n, height=0.4, label="XGBoost gain", color="#ee854a")
    ax.set_yticks(y, names)
    ax.set_xlabel("normalised importance")
    ax.set_title("SHAP vs. built-in gain importance (top SHAP features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_beeswarm(shap_values: np.ndarray, X: np.ndarray, feature_names, path: Path) -> None:
    """Beeswarm summary plot (per-feature value colour + SHAP spread)."""
    shap = require_shap()
    plt.figure()
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, show=False, max_display=20
    )
    plt.title("XGBoost expert — SHAP summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_waterfalls(
    explainer, shap_values, expected, X, feature_names, texts, out_dir: Path, n: int = 3
) -> None:
    """Local per-sample waterfall explanations for a few illustrative tweets."""
    shap = require_shap()
    # Pick the most confidently-pushed samples in each direction + one near zero.
    net = shap_values.sum(axis=1)
    picks = list({int(net.argmax()), int(net.argmin()), int(np.abs(net).argmin())})
    for rank, idx in enumerate(picks[:n]):
        try:
            expl = shap.Explanation(
                values=shap_values[idx],
                base_values=expected,
                data=X[idx],
                feature_names=feature_names,
            )
            plt.figure()
            shap.plots.waterfall(expl, show=False, max_display=14)
            if texts is not None:
                plt.title(str(texts[idx])[:80], fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"shap_waterfall_{rank}.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:  # waterfall API varies across shap versions
            print(f"[WARN] waterfall {rank} skipped ({type(exc).__name__}: {exc}).")


def explain_xgboost(
    model_dir: Path,
    split_path: Path,
    output_dir: Path,
    n_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Run the full SHAP report for the XGBoost expert; returns the ranking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model, extractor = load_expert(model_dir)
    feature_names = list(extractor.feature_names_)

    X_text, _y = load_split(str(split_path), "test")
    texts = X_text.tolist()
    X = extractor.transform(X_text)

    # Subsample for tractable, representative explanations.
    rng = np.random.default_rng(seed)
    if len(X) > n_samples:
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X, texts = X[idx], [texts[i] for i in idx]

    explainer, shap_values, expected = compute_shap_values(model, X)
    ranking = global_importance(shap_values, feature_names)

    plot_global_bar(ranking, output_dir / "shap_global_importance.png")
    plot_shap_vs_gain(ranking, model, feature_names, output_dir / "shap_vs_gain_importance.png")
    save_beeswarm(shap_values, X, feature_names, output_dir / "shap_summary_beeswarm.png")
    save_waterfalls(
        explainer, shap_values, expected, X, feature_names, texts, output_dir
    )

    (output_dir / "shap_global_importance.json").write_text(
        json.dumps({"expected_value": expected, "ranking": ranking}, indent=2),
        encoding="utf-8",
    )
    csv_lines = ["rank,feature,mean_abs_shap"]
    csv_lines += [f"{r['rank']},{r['feature']},{r['mean_abs_shap']:.6f}" for r in ranking]
    (output_dir / "shap_feature_importance.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    print(f"[INFO] XGBoost SHAP report written to {output_dir.resolve()}")
    print("Top 10 features by mean |SHAP|:")
    for r in ranking[:10]:
        print(f"  {r['rank']:>2}. {r['feature']:<24} {r['mean_abs_shap']:.4f}")
    return {"expected_value": expected, "ranking": ranking}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explanations for the XGBoost expert.")
    parser.add_argument("--model-dir", default="models/xgboost")
    parser.add_argument("--split-path", default="data/splits/test.csv")
    parser.add_argument("--output-dir", default="models/xgboost/shap")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explain_xgboost(
        Path(args.model_dir), Path(args.split_path), Path(args.output_dir),
        n_samples=args.n_samples, seed=args.seed,
    )


if __name__ == "__main__":
    main()

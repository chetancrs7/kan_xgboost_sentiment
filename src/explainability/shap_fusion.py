"""SHAP explanations for the Residual-Aware Adaptive Fusion gate.

The gate maps the meta-feature vector to a trust value ``alpha in [0, 1]``
(1 = rely on the KAN semantic expert, 0 = rely on the XGBoost structural
expert). We explain **alpha** because it is a clean function of the
meta-features alone -- exactly the quantity a reader wants attributed: *why did
the ensemble trust one expert over the other on this sample?*

``alpha`` is explained with :class:`shap.KernelExplainer` (model-agnostic
Shapley sampling) wrapped around the gate's forward pass. The fused probability
is deliberately **not** explained w.r.t. the meta-features: it also depends on
the two experts' per-sample probabilities, which are separate inputs KernelSHAP
would not perturb, so a feature attribution of the fused output would be
ill-posed.

Outputs (written under ``--output-dir``):

* ``fusion_shap_summary_beeswarm.png``  per-feature effect on alpha
* ``fusion_shap_global_importance.png`` mean |SHAP| bar chart
* ``fusion_shap_waterfall_trust_kan.png`` / ``_trust_xgb.png`` local cases
* ``fusion_shap_global_importance.json`` machine-readable ranking

Run::

    python -m src.explainability.shap_fusion \
        --checkpoint models/fusion/fusion_model.pt \
        --inputs-dir data/fusion_inputs --kan-dir models/kan_subset --n-explain 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.explainability import require_shap  # noqa: E402
from src.fusion.fusion_dataset import load_fusion_inputs  # noqa: E402
from src.fusion.predictor import FusionPredictor  # noqa: E402


def build_test_features(predictor: FusionPredictor, inputs_dir: Path, kan_dir: Path):
    """Reconstruct the scaled test meta-features + names from the checkpoint."""
    builder = predictor.builder_from_checkpoint()
    inputs = load_fusion_inputs(inputs_dir, kan_dir, splits=("test",))["test"]
    # Apply the checkpoint's temperature calibration so the meta-features match
    # exactly what the gate was trained on (no-op if trained without calibration).
    kan_probs, xgb_probs = predictor.calibrate_probs(inputs["kan_probs"], inputs["xgb_probs"])
    split = builder.build(
        "test", inputs["df"], inputs["embeddings"], kan_probs, xgb_probs
    )
    X_scaled = predictor.scaler.transform(split.features).astype(np.float64)
    return X_scaled, list(split.feature_names), split


def gate_alpha_fn(predictor: FusionPredictor):
    """Numpy callable mapping scaled meta-features -> gate alpha (trust in KAN)."""

    def f(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.atleast_2d(x_np), dtype=torch.float32, device=predictor.device)
            alpha = predictor.model(t).squeeze(-1).cpu().numpy()
        return alpha

    return f


def compute_shap_values(f, X_scaled: np.ndarray, n_explain: int, n_background: int, seed: int):
    """KernelSHAP values of alpha for a subsample of rows.

    Returns ``(shap_values[n_explain, n_features], expected_value, explain_idx)``.
    """
    shap = require_shap()
    rng = np.random.default_rng(seed)
    background = shap.kmeans(X_scaled, min(n_background, len(X_scaled)))
    explainer = shap.KernelExplainer(f, background)

    explain_idx = (
        rng.choice(len(X_scaled), size=n_explain, replace=False)
        if len(X_scaled) > n_explain
        else np.arange(len(X_scaled))
    )
    shap_values = explainer.shap_values(X_scaled[explain_idx], nsamples="auto", silent=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    expected = explainer.expected_value
    expected = float(np.ravel(expected)[0])
    return np.asarray(shap_values), expected, explain_idx


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> list[dict]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {"rank": r + 1, "feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for r, i in enumerate(order)
    ]


def plot_global_bar(ranking: list[dict], path: Path, top_k: int = 20) -> None:
    top = ranking[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(top))))
    ax.barh(
        [r["feature"] for r in top], [r["mean_abs_shap"] for r in top],
        color="#6acc64", edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel("mean |SHAP value|  (impact on gate trust α)")
    ax.set_title(f"Fusion gate — what drives trust in the KAN expert (top {len(top)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_beeswarm(shap_values, X, feature_names, path: Path) -> None:
    shap = require_shap()
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=20)
    plt.title("Fusion gate — SHAP summary for trust α (beeswarm)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_waterfalls(shap_values, expected, X, feature_names, alpha, out_dir: Path) -> None:
    """Local explanations for the strongest 'trust KAN' and 'trust XGB' cases."""
    shap = require_shap()
    cases = {"trust_kan": int(alpha.argmax()), "trust_xgb": int(alpha.argmin())}
    for label, idx in cases.items():
        try:
            expl = shap.Explanation(
                values=shap_values[idx], base_values=expected,
                data=X[idx], feature_names=feature_names,
            )
            plt.figure()
            shap.plots.waterfall(expl, show=False, max_display=14)
            plt.title(f"α = {alpha[idx]:.2f} ({label.replace('_', ' ')})", fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f"fusion_shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            print(f"[WARN] fusion waterfall '{label}' skipped ({type(exc).__name__}: {exc}).")


def explain_fusion(
    checkpoint: Path,
    inputs_dir: Path,
    kan_dir: Path,
    output_dir: Path,
    n_explain: int = 300,
    n_background: int = 25,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Run the full SHAP report for the fusion gate; returns the alpha ranking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = FusionPredictor.from_checkpoint(checkpoint, device=device)
    X_scaled, feature_names, _split = build_test_features(predictor, inputs_dir, kan_dir)

    f = gate_alpha_fn(predictor)
    shap_values, expected, explain_idx = compute_shap_values(
        f, X_scaled, n_explain, n_background, seed
    )
    X_explained = X_scaled[explain_idx]
    alpha = f(X_explained)
    ranking = global_importance(shap_values, feature_names)

    plot_global_bar(ranking, output_dir / "fusion_shap_global_importance.png")
    save_beeswarm(shap_values, X_explained, feature_names, output_dir / "fusion_shap_summary_beeswarm.png")
    save_waterfalls(shap_values, expected, X_explained, feature_names, alpha, output_dir)

    (output_dir / "fusion_shap_global_importance.json").write_text(
        json.dumps(
            {"expected_alpha": expected, "n_explained": int(len(explain_idx)), "ranking": ranking},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Fusion SHAP report written to {output_dir.resolve()}")
    print(f"Mean gate trust α over background: {expected:.3f}")
    print("Top 10 meta-features driving expert selection:")
    for r in ranking[:10]:
        print(f"  {r['rank']:>2}. {r['feature']:<26} {r['mean_abs_shap']:.4f}")
    return {"expected_alpha": expected, "ranking": ranking}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explanations for the fusion gate.")
    parser.add_argument("--checkpoint", default="models/fusion/fusion_model.pt")
    parser.add_argument("--inputs-dir", default="data/fusion_inputs")
    parser.add_argument("--kan-dir", default="models/kan_subset")
    parser.add_argument("--output-dir", default="models/fusion/shap")
    parser.add_argument("--n-explain", type=int, default=300, help="Rows to attribute (KernelSHAP is slow).")
    parser.add_argument("--n-background", type=int, default=25, help="k-means background summary size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explain_fusion(
        Path(args.checkpoint), Path(args.inputs_dir), Path(args.kan_dir), Path(args.output_dir),
        n_explain=args.n_explain, n_background=args.n_background, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()

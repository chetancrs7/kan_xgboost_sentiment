"""Run SHAP explanations for both the XGBoost expert and the fusion gate.

Thin orchestrator over :mod:`src.explainability.shap_xgboost` and
:mod:`src.explainability.shap_fusion`. Either half can be skipped so you can
regenerate one report without the other.

Run::

    python -m src.explainability.run_explainability            # both
    python -m src.explainability.run_explainability --only xgboost
    python -m src.explainability.run_explainability --only fusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explainability.shap_fusion import explain_fusion  # noqa: E402
from src.explainability.shap_xgboost import explain_xgboost  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explainability for the hybrid ensemble.")
    parser.add_argument("--only", choices=["xgboost", "fusion"], default=None,
                        help="Run only one explainer (default: both).")
    # XGBoost expert
    parser.add_argument("--xgb-model-dir", default="models/xgboost")
    parser.add_argument("--xgb-split-path", default="data/splits/test.csv")
    parser.add_argument("--xgb-output-dir", default="models/xgboost/shap")
    parser.add_argument("--xgb-n-samples", type=int, default=2000)
    # Fusion gate
    parser.add_argument("--fusion-checkpoint", default="models/fusion/fusion_model.pt")
    parser.add_argument("--fusion-inputs-dir", default="data/fusion_inputs")
    parser.add_argument("--fusion-kan-dir", default="models/kan_subset")
    parser.add_argument("--fusion-output-dir", default="models/fusion/shap")
    parser.add_argument("--fusion-n-explain", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.only in (None, "xgboost"):
        print("=" * 64 + "\nXGBoost expert — SHAP\n" + "=" * 64)
        explain_xgboost(
            Path(args.xgb_model_dir), Path(args.xgb_split_path), Path(args.xgb_output_dir),
            n_samples=args.xgb_n_samples, seed=args.seed,
        )
    if args.only in (None, "fusion"):
        print("=" * 64 + "\nFusion gate — SHAP\n" + "=" * 64)
        explain_fusion(
            Path(args.fusion_checkpoint), Path(args.fusion_inputs_dir), Path(args.fusion_kan_dir),
            Path(args.fusion_output_dir), n_explain=args.fusion_n_explain,
            seed=args.seed, device=args.device,
        )


if __name__ == "__main__":
    main()

"""Train the *reversed* XGBoost expert on dense DeBERTa embeddings.

Role reversal
-------------
The main architecture gives XGBoost the engineered structural features and the
KAN the semantic embeddings. This script does the opposite half of the swap: it
trains XGBoost on the precomputed DeBERTa embeddings
(``{split}_embeddings.npy`` + ``{split}_labels.npy``), reusing the *exact* same
Optuna search, final-fit, and evaluation code the production XGBoost expert uses
so the only thing that changes is the input matrix.

Nothing here modifies existing files — it imports the existing helpers:
``optimize_xgboost`` / ``train_final_model`` (``src/models/xgboost/model.py``)
and ``evaluate_split`` / ``print_classification_report`` /
``summarize_feature_importance`` (``src/models/xgboost/evaluate.py``).

The output layout mirrors ``models/xgboost/`` (model + ``{split}_probs.npy`` +
metrics + importance), so the reversed expert is a drop-in for any downstream
comparison or fusion step.

Usage::

    # Full dataset (reuses the 80k/10k/10k DeBERTa embeddings already on disk):
    python experiments/reversed_roles/train_xgboost_on_embeddings.py

    python experiments/reversed_roles/train_xgboost_on_embeddings.py \
        --embeddings-dir data/embeddings/deberta_v3_base \
        --output-dir models/reversed_roles/xgboost_on_embeddings \
        --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402

from src.models.kan.data_loader import load_splits  # noqa: E402  (reads emb+labels npy)
from src.models.xgboost.evaluate import (  # noqa: E402
    evaluate_split,
    print_classification_report,
    summarize_feature_importance,
)
from src.models.xgboost.model import (  # noqa: E402
    RANDOM_STATE,
    _scale_pos_weight,
    optimize_xgboost,
    train_final_model,
)

DEFAULT_EMBEDDINGS_DIR = "data/embeddings/deberta_v3_base"
DEFAULT_OUTPUT_DIR = "models/reversed_roles/xgboost_on_embeddings"
DEFAULT_N_TRIALS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reversed XGBoost expert: trained on DeBERTa embeddings."
    )
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Optuna trials. 768-dim inputs make each trial heavier than the "
        "structural expert; lower this to trade tuning depth for speed.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_artifacts(output_dir: Path, model, best_params, study, metrics, importance, probs):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "xgboost_model.json"))
    joblib.dump(model, output_dir / "xgboost_model.pkl")
    for split, array in probs.items():
        np.save(output_dir / f"{split}_probs.npy", array)
    (output_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_importance.json").write_text(
        json.dumps(importance, indent=2), encoding="utf-8"
    )
    if study is not None:
        study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)
    print(f"\n[INFO] All artifacts saved to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    # 1. Load precomputed embeddings + labels (same arrays the KAN normally eats).
    splits = load_splits(args.embeddings_dir)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    print(f"[INFO] Embedding dim: {X_train.shape[1]} | {n_classes} classes.")

    # 2-3. Reuse the production Optuna search + early-stopped final fit verbatim.
    best_params, study = optimize_xgboost(
        X_train, y_train, X_val, y_val, n_trials=args.n_trials, random_state=args.seed
    )
    model = train_final_model(
        X_train, y_train, X_val, y_val, best_params, random_state=args.seed
    )

    # 4. Evaluate with the shared evaluator so metrics are comparable 1:1.
    print("\n" + "=" * 64)
    print("EVALUATION")
    print("=" * 64)
    train_metrics, train_probs = evaluate_split(model, X_train, y_train, "train", n_classes)
    val_metrics, val_probs = evaluate_split(model, X_val, y_val, "validation", n_classes)
    test_metrics, test_probs = evaluate_split(model, X_test, y_test, "test", n_classes)
    print_classification_report(model, X_test, y_test, n_classes)

    feature_names = [f"emb_{i:03d}" for i in range(X_train.shape[1])]
    importance = summarize_feature_importance(model, feature_names)
    print("\nTop 10 embedding dims by importance:")
    for rec in importance[:10]:
        print(f"  {rec['rank']:>2}. {rec['feature']:<10} {rec['importance']:.4f}")

    metrics = {
        "model": "XGBoost (reversed: trained on DeBERTa embeddings)",
        "n_classes": n_classes,
        "n_features": int(X_train.shape[1]),
        "best_val_f1": float(study.best_value) if study is not None else None,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_artifacts(
        output_dir=Path(args.output_dir),
        model=model,
        best_params=best_params,
        study=study,
        metrics=metrics,
        importance=importance,
        probs={"train": train_probs, "val": val_probs, "test": test_probs},
    )
    print("\n[DONE] Reversed XGBoost-on-embeddings pipeline complete.")


if __name__ == "__main__":
    main()

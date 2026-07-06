"""End-to-end training pipeline for the XGBoost expert sentiment model.

Execution flow
--------------
1. Load the pre-split train/validation/test CSVs.
2. Fit a :class:`FeatureExtractor` on training text and transform every split.
3. Run Optuna to optimise XGBoost hyperparameters on validation macro F1.
4. Refit the final model with early stopping.
5. Evaluate on train / validation / test.
6. Export prediction probabilities as ``{train,val,test}_probs.npy``.
7. Persist the model, feature extractor, metrics, feature importance, and the
   Optuna trial log.

Usage:
    python src/models/xgboost/train.py
    python src/models/xgboost/train.py --n-trials 100
    python src/models/xgboost/train.py \
        --train-path data/splits/train.csv \
        --val-path data/splits/val.csv \
        --test-path data/splits/test.csv \
        --output-dir models/xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

# Make the project root importable so this file works as a script
# (`python src/models/xgboost/train.py`), as a module (`python -m
# src.models.xgboost.train`), and imported from the Hydra entry point
# (`src/train.py`). Package-qualified imports avoid clashing with the
# KAN expert's identically named data_loader/evaluate/model modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import joblib  # noqa: E402

from src.models.xgboost.data_loader import (  # noqa: E402
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    load_splits,
)
from src.models.xgboost.evaluate import (  # noqa: E402
    evaluate_split,
    print_classification_report,
    summarize_feature_importance,
)
from src.models.xgboost.features import (  # noqa: E402
    FeatureExtractor,
    ensure_nltk_resources,
)
from src.models.xgboost.model import (  # noqa: E402
    RANDOM_STATE,
    _scale_pos_weight,
    optimize_xgboost,
    train_final_model,
)

DEFAULT_OUTPUT_DIR = "models/xgboost"
DEFAULT_N_TRIALS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost expert sentiment classifier with engineered features."
    )
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Number of Optuna hyperparameter trials (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Global random seed (default: %(default)s).",
    )
    return parser.parse_args()


def save_artifacts(
    output_dir: Path,
    model,
    extractor: FeatureExtractor,
    best_params: dict,
    study,
    metrics: dict,
    importance: list[dict],
    probs: dict[str, np.ndarray],
) -> None:
    """Persist all model artifacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trained model: native JSON + joblib pickle for convenience.
    model.save_model(str(output_dir / "xgboost_model.json"))
    joblib.dump(model, output_dir / "xgboost_model.pkl")
    joblib.dump(extractor, output_dir / "feature_extractor.pkl")

    # Prediction probabilities for downstream ensembling.
    for split, array in probs.items():
        np.save(output_dir / f"{split}_probs.npy", array)

    # Hyperparameters.
    (output_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )

    # Metrics for every split.
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Feature importance: JSON + CSV.
    (output_dir / "feature_importance.json").write_text(
        json.dumps(importance, indent=2), encoding="utf-8"
    )
    csv_lines = ["rank,feature,importance"]
    csv_lines += [
        f"{rec['rank']},{rec['feature']},{rec['importance']:.6f}"
        for rec in importance
    ]
    (output_dir / "feature_importance.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    # Optuna trial log (absent when the search was skipped).
    if study is not None:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    print(f"\n[INFO] All artifacts saved to: {output_dir.resolve()}")


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    set_global_seed(args.seed)
    ensure_nltk_resources()

    # 1. Load splits ------------------------------------------------------
    X_train_text, y_train, X_val_text, y_val, X_test_text, y_test = load_splits(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )
    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()
    y_test_arr = y_test.to_numpy()
    n_classes = int(len(np.unique(np.concatenate([y_train_arr, y_val_arr, y_test_arr]))))
    print(f"[INFO] Detected {n_classes} target classes.")

    # 2. Feature engineering ---------------------------------------------
    print("[INFO] Extracting linguistic features...")
    extractor = FeatureExtractor()
    X_train = extractor.fit_transform(X_train_text)
    X_val = extractor.transform(X_val_text)
    X_test = extractor.transform(X_test_text)
    print(
        f"[INFO] Feature matrices | train: {X_train.shape} | "
        f"val: {X_val.shape} | test: {X_test.shape}"
    )

    # 3-4. Hyperparameter search + final fit ------------------------------
    # The Hydra entry point (src/train.py) may pass fixed hyperparameters
    # and disable the Optuna search; standalone CLI runs keep searching.
    use_optuna = getattr(args, "use_optuna", True)
    fixed_params = getattr(args, "fixed_params", None)
    if use_optuna or not fixed_params:
        best_params, study = optimize_xgboost(
            X_train, y_train_arr, X_val, y_val_arr,
            n_trials=args.n_trials, random_state=args.seed,
        )
    else:
        best_params = dict(fixed_params)
        if n_classes == 2:
            best_params.setdefault("scale_pos_weight", _scale_pos_weight(y_train_arr))
        study = None
        print("[INFO] Optuna disabled; using fixed hyperparameters:")
        for key, value in best_params.items():
            print(f"         {key}: {value}")
    model = train_final_model(
        X_train, y_train_arr, X_val, y_val_arr, best_params, random_state=args.seed,
    )

    # 5. Evaluate ---------------------------------------------------------
    print("\n" + "=" * 64)
    print("EVALUATION")
    print("=" * 64)
    train_metrics, train_probs = evaluate_split(
        model, X_train, y_train_arr, "train", n_classes
    )
    val_metrics, val_probs = evaluate_split(
        model, X_val, y_val_arr, "validation", n_classes
    )
    test_metrics, test_probs = evaluate_split(
        model, X_test, y_test_arr, "test", n_classes
    )
    print_classification_report(model, X_test, y_test_arr, n_classes)

    metrics = {
        "model": "XGBoost (engineered linguistic features)",
        "n_classes": n_classes,
        "n_features": extractor.n_features_,
        "best_val_f1": float(study.best_value) if study is not None else None,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    # 6. Feature importance ----------------------------------------------
    importance = summarize_feature_importance(model, extractor.feature_names_)
    print("\nTop 10 features by importance:")
    for rec in importance[:10]:
        print(f"  {rec['rank']:>2}. {rec['feature']:<22} {rec['importance']:.4f}")

    # 7. Persist ----------------------------------------------------------
    save_artifacts(
        output_dir=Path(args.output_dir),
        model=model,
        extractor=extractor,
        best_params=best_params,
        study=study,
        metrics=metrics,
        importance=importance,
        probs={"train": train_probs, "val": val_probs, "test": test_probs},
    )

    print("\n[DONE] XGBoost expert pipeline complete.")


if __name__ == "__main__":
    main()

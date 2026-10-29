"""Self-contained TF-IDF + XGBoost sentiment classifier (full dataset).

Why this exists
---------------
The original XGBoost expert (``src/models/xgboost``) feeds the model ~30
hand-crafted linguistic features (negation counts, VADER scores, POS ratios,
...). Those features never let the model see *which words* a tweet contains, so
on this dataset it performs at roughly chance level. This model instead
vectorises the raw text with TF-IDF word + character n-grams and trains on the
full training split, which is the standard, strong classical baseline for tweet
sentiment.

It is intentionally standalone. To remove it entirely:

    rm -r src/models/xgboost_tfidf models/xgboost_tfidf

Usage
-----
    python src/models/xgboost_tfidf/train.py
    python src/models/xgboost_tfidf/train.py --word-features 60000 --char-features 60000
    python src/models/xgboost_tfidf/train.py \
        --train-path data/splits/train.csv \
        --val-path data/splits/val.csv \
        --test-path data/splits/test.csv \
        --output-dir models/xgboost_tfidf

To also emit per-split probability arrays that the fusion pipeline can consume
(``scripts/prepare_fusion_inputs.py``), add ``--save-split-probs``. This writes
``{train,val,test}_probs.npy`` (shape ``[n_rows, 2]``) into ``--output-dir``,
row-aligned with the *raw* split CSVs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Defaults
DEFAULT_TRAIN_PATH = "data/splits/train.csv"
DEFAULT_VAL_PATH = "data/splits/val.csv"
DEFAULT_TEST_PATH = "data/splits/test.csv"
DEFAULT_OUTPUT_DIR = "models/xgboost_tfidf"

TEXT_COLUMN = "text"          # raw tweet text (casing/punctuation carry signal)
FALLBACK_TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "label"
RANDOM_STATE = 42


# Data loading (inlined so this package has no internal dependencies)
def load_split(path: str, name: str) -> tuple[pd.Series, np.ndarray]:
    """Return ``(text, label)`` for one split CSV."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[ERROR] Missing split file: {p.resolve()}")
    df = pd.read_csv(p)
    if TEXT_COLUMN in df.columns:
        col = TEXT_COLUMN
    elif FALLBACK_TEXT_COLUMN in df.columns:
        col = FALLBACK_TEXT_COLUMN
        print(f"[WARN] '{TEXT_COLUMN}' missing in {name}; using '{col}'.")
    else:
        raise SystemExit(
            f"[ERROR] {name} has no text column. Found: {list(df.columns)}"
        )
    if LABEL_COLUMN not in df.columns:
        raise SystemExit(f"[ERROR] {name} has no '{LABEL_COLUMN}' column.")

    df = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    text = df[col].fillna("").astype(str)
    labels = df[LABEL_COLUMN].astype(int).to_numpy()
    print(f"[INFO] {name}: {len(text):,} rows (text column: '{col}').")
    return text, labels


# Feature engineering: TF-IDF word + character n-grams
def build_vectorizers(
    word_features: int, char_features: int
) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """Word (1-2 gram) and character (2-5 gram) TF-IDF vectorizers.

    Word n-grams capture lexical sentiment ("not good", "love it"); character
    n-grams add robustness to the misspellings, elongations and hashtags that
    are pervasive in tweets ("sooo", "gr8", "#happy").
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b|[:;=][-^]?[)(\]\[dpDP/|]|<3",  # keep emoticons
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=char_features,
        sublinear_tf=True,
    )
    return word_vec, char_vec


def vectorize(
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    train_text: pd.Series,
    val_text: pd.Series,
    test_text: pd.Series,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Fit vectorizers on train text, transform every split, and concat."""
    print("[INFO] Fitting TF-IDF vectorizers on training text...")
    Xw_tr = word_vec.fit_transform(train_text)
    Xc_tr = char_vec.fit_transform(train_text)
    X_train = hstack([Xw_tr, Xc_tr]).tocsr()

    X_val = hstack([word_vec.transform(val_text), char_vec.transform(val_text)]).tocsr()
    X_test = hstack([word_vec.transform(test_text), char_vec.transform(test_text)]).tocsr()

    print(
        f"[INFO] Feature matrix: {X_train.shape[1]:,} features "
        f"({Xw_tr.shape[1]:,} word + {Xc_tr.shape[1]:,} char). "
        f"train={X_train.shape[0]:,} val={X_val.shape[0]:,} test={X_test.shape[0]:,}"
    )
    return X_train, X_val, X_test


# Model
def build_model(
    n_estimators: int,
    eval_metric: str | list[str] = "logloss",
    callbacks: list | None = None,
) -> XGBClassifier:
    """XGBoost tuned for wide, sparse TF-IDF input.

    Heavy column subsampling (``colsample_bytree``) is the key knob for
    high-dimensional sparse text: it both regularises and speeds up training.

    ``eval_metric`` may be a list; XGBoost keys early stopping on the **last**
    entry, so callers that add extra metrics (e.g. ``error`` for accuracy) must
    keep ``logloss`` last to preserve the early-stopping behaviour. ``callbacks``
    is forwarded to XGBoost's per-iteration callback hook (used for live
    TensorBoard logging).
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric=eval_metric,
        callbacks=callbacks,
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.4,
        min_child_weight=2,
        reg_lambda=2.0,
        reg_alpha=0.0,
        gamma=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=40,
    )


def save_split_probs(
    model: XGBClassifier,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    split_paths: dict[str, str],
    out: Path,
) -> None:
    """Persist full-split predicted probabilities for the fusion pipeline.

    For each split the model's ``predict_proba`` is run over **every** row of
    the raw CSV, in the original file order, and saved as ``{split}_probs.npy``
    with shape ``[n_rows, 2]``. This is exactly the layout
    ``scripts/prepare_fusion_inputs.py`` expects: it asserts
    ``len(probs) == len(raw_csv)`` and then slices the array by its subsample
    indices, so no rows may be dropped or reordered here (unlike
    :func:`load_split`, which drops NaN-label rows for training).

    Column order is ``[P(class 0 = negative), P(class 1 = positive)]`` because
    ``model.classes_ == [0, 1]``, matching the KAN expert's ordering.
    """
    for name, path in split_paths.items():
        df = pd.read_csv(path)
        if TEXT_COLUMN in df.columns:
            col = TEXT_COLUMN
        elif FALLBACK_TEXT_COLUMN in df.columns:
            col = FALLBACK_TEXT_COLUMN
        else:
            raise SystemExit(
                f"[ERROR] {path} has no text column for probability export."
            )
        text = df[col].fillna("").astype(str)
        X = hstack([word_vec.transform(text), char_vec.transform(text)]).tocsr()
        proba = model.predict_proba(X)
        dest = out / f"{name}_probs.npy"
        np.save(dest, proba)
        print(f"[INFO] Saved {name} probabilities {proba.shape} -> {dest}")


def evaluate(model: XGBClassifier, X, y, split: str) -> dict:
    """Compute metrics for one split and print a one-line summary."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "split": split,
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    print(
        f"  {split:<11} acc={metrics['accuracy']:.4f} "
        f"f1={metrics['f1']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"auc={metrics['auc']:.4f}"
    )
    return metrics


# Entry point
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    p.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    p.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--word-features", type=int, default=50000)
    p.add_argument("--char-features", type=int, default=50000)
    p.add_argument("--n-estimators", type=int, default=1200)
    p.add_argument(
        "--save-split-probs",
        action="store_true",
        help=(
            "Also write {train,val,test}_probs.npy ([n_rows, 2]) into --output-dir, "
            "row-aligned with the raw split CSVs, for scripts/prepare_fusion_inputs.py."
        ),
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help=(
            "Log live train/validation logloss and accuracy to TensorBoard, one "
            "point per boosting round. View with: tensorboard --logdir "
            "outputs/tensorboard/xgboost_tfidf. Adds a per-round evaluation on the "
            "training split, so it slightly slows training."
        ),
    )
    p.add_argument(
        "--tb-logdir",
        default="outputs/tensorboard/xgboost_tfidf",
        help="TensorBoard event-file directory (used only with --tensorboard).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # 1. Load ------------------------------------------------------------
    train_text, y_train = load_split(args.train_path, "train")
    val_text, y_val = load_split(args.val_path, "validation")
    test_text, y_test = load_split(args.test_path, "test")

    # 2. Vectorise -------------------------------------------------------
    word_vec, char_vec = build_vectorizers(args.word_features, args.char_features)
    X_train, X_val, X_test = vectorize(
        word_vec, char_vec, train_text, val_text, test_text
    )

    # 3. Train (early stopping on validation) ----------------------------
    print("[INFO] Training XGBoost (early stopping on validation)...")
    tb_callback = None
    eval_metric: str | list[str] = "logloss"
    # Default eval set: validation only (early stopping keys on its last metric).
    eval_set = [(X_val, y_val)]
    eval_names = {"validation_0": "validation"}
    if args.tensorboard:
        from src.models.xgboost_tfidf.tb_callback import TensorBoardCallback

        # Add 'error' (0.5-threshold classification error) to derive accuracy.
        # 'logloss' stays LAST so early stopping still keys on log-loss.
        eval_metric = ["error", "logloss"]
        # Put train first and validation last: XGBoost early-stops on the last
        # dataset, so validation must remain last to preserve behaviour.
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = {"validation_0": "train", "validation_1": "validation"}
        tb_callback = TensorBoardCallback(args.tb_logdir, eval_names)

    model = build_model(
        args.n_estimators,
        eval_metric=eval_metric,
        callbacks=[tb_callback] if tb_callback is not None else None,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    if tb_callback is not None:
        tb_callback.close()
    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        print(f"[INFO] Best iteration: {best_it} / {args.n_estimators}")

    # 4. Evaluate --------------------------------------------------------
    print("\n" + "=" * 60 + "\nEVALUATION\n" + "=" * 60)
    train_metrics = evaluate(model, X_train, y_train, "train")
    val_metrics = evaluate(model, X_val, y_val, "validation")
    test_metrics = evaluate(model, X_test, y_test, "test")
    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
            target_names=["negative", "positive"],
            digits=4,
        )
    )

    # 5. Persist ---------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "xgboost_tfidf_model.json"))
    joblib.dump(model, out / "xgboost_tfidf_model.pkl")
    joblib.dump(word_vec, out / "word_vectorizer.pkl")
    joblib.dump(char_vec, out / "char_vectorizer.pkl")

    if args.save_split_probs:
        print("\n[INFO] Exporting per-split probabilities for the fusion pipeline...")
        save_split_probs(
            model,
            word_vec,
            char_vec,
            {
                "train": args.train_path,
                "val": args.val_path,
                "test": args.test_path,
            },
            out,
        )

    metrics = {
        "model": "XGBoost (TF-IDF word+char n-grams, full dataset)",
        "n_features": int(X_train.shape[1]),
        "best_iteration": int(best_it) if best_it is not None else None,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[INFO] Artifacts saved to {out.resolve()}")
    print(f"[DONE] Total runtime: {metrics['runtime_seconds']}s")


if __name__ == "__main__":
    main()

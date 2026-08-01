"""Interpretable stacking fusion of two experts (KAN + XGBoost).

The most interpretable AND competitive fusion method: a logistic-regression
STACKER over the two experts' class probabilities. Its coefficients ARE the
fusion rule -- you can read exactly how much the ensemble trusts each expert,
and (unlike a neural gate) there is no black box in the combiner.

    p_kan  = KAN P(positive)            (semantic expert -- interpretable concept KAN)
    p_xgb  = XGBoost P(positive)        (structural expert -- interpretable features)
        -> logistic regression on [p_kan, p_xgb]  (fit on VAL, evaluated on TEST)
        -> fused sentiment

Standalone + read-only on the expert outputs; writes only to --output-dir.

The stacker is fit on the VALIDATION split's expert probabilities (never the
experts' own training split, where they are overconfident) and evaluated on the
untouched TEST split -- standard leak-free stacking.

Usage:
    python scripts/fusion_stack.py \
        --kan-dir outputs/concept_kan_full \
        --xgb-dir models/xgboost_full \
        --labels-dir data/embeddings_finetuned \
        --test-csv data/splitsfull/test.csv \
        --output-dir outputs/fusion_stack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_probs(d, split):
    p = Path(d) / f"{split}_probs.npy"
    if not p.exists():
        raise SystemExit(f"[ERROR] missing {p}. (Re-run the expert so it exports {split}_probs.npy.)")
    return np.load(p).astype(np.float64)


def acc(pred, y):
    return float((pred == y).mean())


def macro_f1(pred, y):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro"))


def sarcasm_slice(pred, y, test_csv):
    """Accuracy of the fused predictions on the sarcasm-cue test slice vs the rest."""
    import pandas as pd
    df = pd.read_csv(test_csv, usecols=["text", "label", "sarcasm_flag", "sarcasm_cue_count"])
    if not np.array_equal(df["label"].to_numpy(), y):
        print("[WARN] test CSV not aligned with labels; skipping sarcasm slice.")
        return None
    t = df["text"].astype(str).str.lower()
    marker = t.str.contains(r"#sarcas|#iron|#not\b|\bsarcas|yeah,? right|oh great|just great|/s\b",
                            regex=True, na=False).to_numpy()
    flag = (df["sarcasm_flag"] == 1).to_numpy()
    correct = (pred == y)
    out = {}
    for name, m in [("broad (flag|cue>=1|marker)", flag | (df["sarcasm_cue_count"] >= 1).to_numpy() | marker),
                    ("strict (flag|cue>=2|marker)", flag | (df["sarcasm_cue_count"] >= 2).to_numpy() | marker)]:
        out[name] = {"n": int(m.sum()), "correct": int(correct[m].sum()),
                     "acc_slice": float(correct[m].mean()) if m.any() else None,
                     "acc_rest": float(correct[~m].mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kan-dir", default="outputs/concept_kan_full", help="dir with {val,test}_probs.npy (KAN)")
    ap.add_argument("--xgb-dir", default="models/xgboost_full", help="dir with {val,test}_probs.npy (XGBoost)")
    ap.add_argument("--labels-dir", default="data/embeddings_finetuned", help="dir with {val,test}_labels.npy")
    ap.add_argument("--test-csv", default="data/splitsfull/test.csv")
    ap.add_argument("--output-dir", default="outputs/fusion_stack")
    ap.add_argument("--kan-name", default="KAN")
    ap.add_argument("--xgb-name", default="XGBoost")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load expert probabilities + labels (all row-aligned to the same splits).
    kan = {s: load_probs(args.kan_dir, s) for s in ("val", "test")}
    xgb = {s: load_probs(args.xgb_dir, s) for s in ("val", "test")}
    lab = {s: np.load(Path(args.labels_dir) / f"{s}_labels.npy").astype(np.int64) for s in ("val", "test")}
    for s in ("val", "test"):
        n = len(lab[s])
        if not (len(kan[s]) == len(xgb[s]) == n):
            raise SystemExit(f"[ERROR] {s} length mismatch: kan={len(kan[s])} xgb={len(xgb[s])} labels={n}")
    print(f"[INFO] aligned | val={len(lab['val']):,} test={len(lab['test']):,}")

    # Stand-alone expert accuracies (test).
    kan_acc = acc(kan["test"].argmax(1), lab["test"])
    xgb_acc = acc(xgb["test"].argmax(1), lab["test"])

    # --- Logistic-regression stacker (the interpretable fusion) --------------
    # Features = each expert's P(positive); 2 coefficients = readable trust weights.
    from sklearn.linear_model import LogisticRegression
    Xval = np.column_stack([kan["val"][:, 1], xgb["val"][:, 1]])
    Xtest = np.column_stack([kan["test"][:, 1], xgb["test"][:, 1]])
    clf = LogisticRegression(max_iter=1000, random_state=args.seed).fit(Xval, lab["val"])
    stack_pred = clf.predict(Xtest)
    w_kan, w_xgb = float(clf.coef_[0][0]), float(clf.coef_[0][1])
    denom = abs(w_kan) + abs(w_xgb) + 1e-12
    trust_kan, trust_xgb = abs(w_kan) / denom, abs(w_xgb) / denom

    # --- Simple average baseline (most-interpretable, non-adaptive) ----------
    avg_pred = (0.5 * kan["test"] + 0.5 * xgb["test"]).argmax(1)

    results = {
        "experts": {args.kan_name: {"test_acc": kan_acc}, args.xgb_name: {"test_acc": xgb_acc}},
        "fusion_average": {"test_acc": acc(avg_pred, lab["test"]),
                           "test_macro_f1": macro_f1(avg_pred, lab["test"])},
        "fusion_stack_logreg": {
            "test_acc": acc(stack_pred, lab["test"]),
            "test_macro_f1": macro_f1(stack_pred, lab["test"]),
            "coef": {args.kan_name: w_kan, args.xgb_name: w_xgb, "intercept": float(clf.intercept_[0])},
            "trust_share": {args.kan_name: trust_kan, args.xgb_name: trust_xgb},
        },
    }
    results["fusion_stack_logreg"]["sarcasm_slice"] = sarcasm_slice(stack_pred, lab["test"], args.test_csv)

    # Report.
    print("\n" + "=" * 66)
    print("STACKING FUSION  (logistic-regression meta-learner)")
    print("=" * 66)
    print(f"  {args.kan_name} alone       test acc = {kan_acc:.4f}")
    print(f"  {args.xgb_name} alone   test acc = {xgb_acc:.4f}")
    print(f"  average (0.5/0.5)     test acc = {results['fusion_average']['test_acc']:.4f}")
    print(f"  STACK (logreg)        test acc = {results['fusion_stack_logreg']['test_acc']:.4f}  "
          f"macro_f1 = {results['fusion_stack_logreg']['test_macro_f1']:.4f}")
    print(f"\n  interpretable fusion rule (coefficients on P(positive)):")
    print(f"    {args.kan_name}: {w_kan:+.3f}   {args.xgb_name}: {w_xgb:+.3f}   intercept: {clf.intercept_[0]:+.3f}")
    print(f"    trust share -> {args.kan_name} {trust_kan:.0%} | {args.xgb_name} {trust_xgb:.0%}")
    sl = results["fusion_stack_logreg"]["sarcasm_slice"]
    if sl:
        print("\n  fused accuracy on sarcasm-cue slice (heuristic markers):")
        for name, s in sl.items():
            a = "n/a" if s["acc_slice"] is None else f"{s['acc_slice']:.4f}"
            print(f"    {name}: n={s['n']:,} correct={s['correct']:,} acc={a} vs rest {s['acc_rest']:.4f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / "fusion_stack_results.json", "w"), indent=2)
    np.save(out / "fused_test_probs.npy", clf.predict_proba(Xtest))
    print(f"\n[INFO] Saved fusion results to {(out / 'fusion_stack_results.json').resolve()}")


if __name__ == "__main__":
    main()

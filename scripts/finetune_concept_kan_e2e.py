"""End-to-end Concept-Bottleneck KAN over an unfrozen DeBERTa encoder.

Combines the two earlier standalone pieces into the model the user asked for:
a KAN that is (a) genuinely nonlinear and deeper (2 KAN layers, so concepts can
INTERACT), (b) interpretable (its inputs are 30 NAMED semantic concepts), and
(c) trained end-to-end so the encoder keeps improving the representation -- which
is the only thing that raises the accuracy ceiling instead of plateauing at the
frozen-embedding one.

    Tweet tokens
      -> DeBERTa-v3 encoder (last N transformer layers UNFROZEN)
      -> masked-mean pooling (768)
      -> cosine-project onto 30 fixed NAMED concept anchors -> 30 concept scores
      -> BatchNorm(30)
      -> KAN 30 -> hidden  (spline)     )  a 2-layer KAN: concepts interact,
      -> LayerNorm(hidden)              )  so this is nonlinear + deeper,
      -> KAN hidden -> classes (spline) )  not a linear/GAM model
      -> logits

Interpretability is preserved via per-concept **partial-dependence** curves
(sweep one concept, hold the rest at their mean) -- still a readable
"concept -> sentiment" curve, now through an interacting 2-layer KAN.

Standalone: imports only the project's KAN layer + the concept definitions and
helpers from the two sibling scripts; writes to its own --output-dir. Nothing
existing is modified.

NaN-safety (same traps as scripts/finetune_residual_kan_e2e.py): encoder loaded
in fp32 (transformers v5 defaults to fp16 -> AdamW underflows to NaN), the KAN
head runs in fp32 with autocast disabled, linear LR warmup, gradient clipping.

Usage (full dataset):
    python scripts/finetune_concept_kan_e2e.py \
        --train-csv data/splitsfull/train.csv --val-csv data/splitsfull/val.csv \
        --test-csv data/splitsfull/test.csv --train-size 0 --val-size 8000 \
        --final-eval-size 0 --output-dir outputs/e2e_concept_kan_full
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kan.layers import AdaptiveSplineKANLayer  # noqa: E402
from scripts.concept_bottleneck_kan import CONCEPTS, save_plots  # noqa: E402
from scripts.finetune_residual_kan_e2e import (  # noqa: E402
    TweetDataset, make_collate, stratified_sample, evaluate,
)
from transformers import AutoModel, AutoTokenizer  # noqa: E402


class E2EConceptKAN(nn.Module):
    def __init__(self, encoder_name, concept_names, n_classes, *, unfrozen_layers=2,
                 hidden=16, grid_size=4, spline_order=3):
        super().__init__()
        # fp32 weights (transformers v5 -> fp16 default NaNs AdamW on step 1).
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        hdim = self.encoder.config.hidden_size
        for p in self.encoder.parameters():
            p.requires_grad = False
        layers = self.encoder.encoder.layer
        self.n_unfrozen = min(unfrozen_layers, len(layers))
        for layer in layers[-self.n_unfrozen:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.concept_names = list(concept_names)
        C = len(self.concept_names)
        self.register_buffer("anchors", torch.zeros(C, hdim))  # fixed per epoch, not trained
        self.concept_bn = nn.BatchNorm1d(C)
        self.kan1 = AdaptiveSplineKANLayer(C, hidden, grid_size, spline_order)
        self.mid_norm = nn.LayerNorm(hidden)
        self.kan2 = AdaptiveSplineKANLayer(hidden, n_classes, grid_size, spline_order)

    @staticmethod
    def _masked_mean(h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)

    def encode_pool(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self._masked_mean(out.last_hidden_state, mask)

    def concept_scores(self, pooled):
        pooled = pooled.float()
        p = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-8)
        return p @ self.anchors.t()  # cosine sim [B, C] (anchors are unit-norm)

    def head(self, scores):
        # KAN splines are fp32-only and not autocast-safe.
        with torch.autocast(device_type=scores.device.type, enabled=False):
            z = self.concept_bn(scores.float())
            z = self.mid_norm(self.kan1(z))
            return self.kan2(z)

    def forward(self, input_ids, attention_mask, **kw):
        return self.head(self.concept_scores(self.encode_pool(input_ids, attention_mask)))

    @torch.no_grad()
    def refresh_anchors(self, tok, device, max_len=16):
        """(Re)compute the fixed concept anchors from the CURRENT encoder, so the
        named directions track the encoder as its top layers fine-tune."""
        was = self.training
        self.encoder.eval()
        vecs = []
        for name in self.concept_names:
            e = tok(CONCEPTS[name], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(device)
            h = self.encoder(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            vecs.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).mean(0))
        A = torch.stack(vecs)
        self.anchors.copy_(A / (A.norm(dim=1, keepdim=True) + 1e-8))
        if was:
            self.train()

    @torch.no_grad()
    def seed_grids(self, ids, mask):
        was = self.training
        self.eval()
        scores = self.concept_scores(self.encode_pool(ids, mask))
        with torch.autocast(device_type=scores.device.type, enabled=False):
            z = self.concept_bn(scores.float())
            self.kan1.adapt_grid(z)
            z = self.mid_norm(self.kan1(z))
            self.kan2.adapt_grid(z)
        if was:
            self.train()


@torch.no_grad()
def compute_scores(model, loader, device, amp_dtype, limit=None):
    """Concept scores + labels for a loader (for the final interpretation)."""
    model.eval()
    S, Y = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            sc = model.concept_scores(model.encode_pool(ids, mask))
        S.append(sc.float().cpu().numpy()); Y.append(batch["labels"].numpy())
        if limit and sum(len(s) for s in S) >= limit:
            break
    return np.concatenate(S), np.concatenate(Y)


@torch.no_grad()
def concept_pd_curves(model, val_scores, device, names, n=100):
    """Per-concept partial-dependence curves: sweep one concept over its range,
    hold the others at their mean, read the pos-minus-neg logit."""
    model.eval()
    base = val_scores.mean(0)
    curves, importance = {}, {}
    for i, name in enumerate(names):
        lo, hi = np.percentile(val_scores[:, i], [2, 98])
        xs = np.linspace(lo, hi, n).astype(np.float32)
        inp = np.tile(base, (n, 1)).astype(np.float32)
        inp[:, i] = xs
        out = model.head(torch.from_numpy(inp).to(device)).cpu().numpy()
        eff = out[:, 1] - out[:, 0]
        eff = eff - eff.mean()
        curves[name] = {"x": xs.tolist(), "effect": eff.tolist()}
        importance[name] = float(eff.max() - eff.min())
    return curves, importance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--train-csv", default="data/splits/train.csv")
    ap.add_argument("--val-csv", default="data/splits/val.csv")
    ap.add_argument("--test-csv", default="data/splits/test.csv")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--output-dir", default="outputs/e2e_concept_kan")
    ap.add_argument("--train-size", type=int, default=40000)
    ap.add_argument("--val-size", type=int, default=4000)
    ap.add_argument("--final-eval-size", type=int, default=0, help="0 = full val/test")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--unfrozen-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=4)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    print(f"[INFO] E2E Concept-KAN | encoder={args.encoder} | device={device} | amp={amp}")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    tr = stratified_sample(pd.read_csv(args.train_csv, usecols=[args.text_column, args.label_column]).dropna(),
                           args.train_size, args.label_column, args.seed)
    va = stratified_sample(pd.read_csv(args.val_csv, usecols=[args.text_column, args.label_column]).dropna(),
                           args.val_size, args.label_column, args.seed)
    n_classes = int(tr[args.label_column].nunique())
    print(f"[INFO] train={len(tr)} val={len(va)} classes={n_classes} concepts={len(CONCEPTS)}")

    collate = make_collate(tok, args.max_length)
    train_loader = DataLoader(TweetDataset(tr[args.text_column], tr[args.label_column]),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(TweetDataset(va[args.text_column], va[args.label_column]),
                            batch_size=64, shuffle=False, collate_fn=collate)

    names = list(CONCEPTS)
    model = E2EConceptKAN(args.encoder, names, n_classes, unfrozen_layers=args.unfrozen_layers,
                          hidden=args.hidden, grid_size=args.grid_size, spline_order=args.spline_order).to(device)
    model.refresh_anchors(tok, device)
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    print(f"[INFO] trainable: encoder {sum(p.numel() for p in enc_p):,} (last {model.n_unfrozen}) + "
          f"head {sum(p.numel() for p in head_p):,} | 2-layer KAN 30->{args.hidden}->{n_classes}")

    opt = torch.optim.AdamW([{"params": enc_p, "lr": args.encoder_lr},
                             {"params": head_p, "lr": args.head_lr}], weight_decay=args.weight_decay)
    total = max(1, len(train_loader) * args.epochs)
    warm = int(total * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    crit = nn.CrossEntropyLoss()

    seed_b = next(iter(train_loader))
    model.seed_grids(seed_b["input_ids"].to(device), seed_b["attention_mask"].to(device))
    print("[INFO] Seeded KAN grids.")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    hist = {"step": [], "val_acc": [], "gamma_na": []}
    best, best_state, step, t0 = 0.0, copy.deepcopy(model.state_dict()), 0, time.time()
    vl, vacc = evaluate(model, val_loader, device, amp)
    hist["step"].append(0); hist["val_acc"].append(vacc)
    print(f"[step 000000] val_acc={vacc:.4f} (pre-training)")

    for epoch in range(1, args.epochs + 1):
        model.refresh_anchors(tok, device)  # keep concept directions aligned with the encoder
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                logits = model(ids, mask)
            loss = crit(logits.float(), y)
            loss.backward()
            if args.clip_norm > 0:
                nn.utils.clip_grad_norm_(enc_p + head_p, args.clip_norm)
            opt.step(); sched.step(); step += 1
            if step % args.eval_every == 0:
                vl, vacc = evaluate(model, val_loader, device, amp)
                hist["step"].append(step); hist["val_acc"].append(vacc)
                if vacc > best:
                    best, best_state = vacc, copy.deepcopy(model.state_dict())
                print(f"[ep {epoch} step {step:06d}] val_acc={vacc:.4f} ({(time.time()-t0)/60:.1f} min)")
                (out / "history.json").write_text(json.dumps({"args": vars(args), "best_val_acc": best, "history": hist}), encoding="utf-8")
                model.train()
        vl, vacc = evaluate(model, val_loader, device, amp)
        hist["step"].append(step); hist["val_acc"].append(vacc)
        if vacc > best:
            best, best_state = vacc, copy.deepcopy(model.state_dict())
        print(f"== epoch {epoch} done | val_acc={vacc:.4f} ==")

    model.load_state_dict(best_state)

    # Final full val/test.
    final = {}
    for nm, csv in (("val", args.val_csv), ("test", args.test_csv)):
        df = stratified_sample(pd.read_csv(csv, usecols=[args.text_column, args.label_column]).dropna(),
                               args.final_eval_size, args.label_column, args.seed)
        ld = DataLoader(TweetDataset(df[args.text_column], df[args.label_column]),
                        batch_size=64, shuffle=False, collate_fn=collate)
        fl, fa = evaluate(model, ld, device, amp)
        final[nm] = {"n": len(df), "acc": fa}
        print(f"[FINAL] full {nm}: n={len(df):,} acc={fa:.4f}")

    # Interpretation: per-concept partial-dependence curves + graphs.
    model.refresh_anchors(tok, device)
    val_scores, _ = compute_scores(model, val_loader, device, amp, limit=8000)
    curves, importance = concept_pd_curves(model, val_scores, device, names)
    ranked = sorted(importance, key=importance.get, reverse=True)
    corr = {}  # linear corr of each concept with... not available here; leave placeholder
    save_plots(names, importance, ranked, curves, out)

    print("\n" + "=" * 60 + "\nRESULTS\n" + "=" * 60)
    print(f"  val_acc: pre-train={hist['val_acc'][0]:.4f}  best={best:.4f}")
    print(f"  FINAL full val={final['val']['acc']:.4f}  test={final['test']['acc']:.4f}")
    print("  top concepts (partial-dependence swing):")
    for nm in ranked[:12]:
        tr_up = curves[nm]["effect"][-1] > curves[nm]["effect"][0]
        print(f"    {nm:<15} swing={importance[nm]:.3f}  trend~{'positive' if tr_up else 'negative'}")

    torch.save({"model_state_dict": best_state, "args": vars(args)}, out / "best_model.pt")
    json.dump({"args": vars(args), "best_val_acc": best, "final_full_eval": final,
               "importance": importance, "ranked": ranked, "curves": curves,
               "history": hist}, open(out / "concept_e2e_results.json", "w"), indent=2)
    print(f"\n[INFO] Saved results + graphs to {out.resolve()}")


if __name__ == "__main__":
    main()

# Mixed-spline KAN ablation (test split)

All experiments share widths 768->256->128->64->32->classes, a linear output head, and identical training settings; only the per-layer spline families differ.

| experiment | spline types (L1..L4) | accuracy | precision | recall | macro_f1 | weighted_f1 | auc | best_val_loss | epochs |
|---|---|---|---|---|---|---|---|---|---|
| all_bspline | bspline / bspline / bspline / bspline | 0.7870 | 0.7493 | 0.8631 | 0.7857 | 0.7857 | 0.8781 | 0.4839 | 20 |
| bspline_cubic | bspline / cubic_bspline / bspline / bspline | 0.7875 | 0.7509 | 0.8611 | 0.7863 | 0.7863 | 0.8778 | 0.4860 | 20 |
| bspline_cubic_rbf | bspline / cubic_bspline / rbf / rbf | 0.7970 | 0.8083 | 0.7792 | 0.7969 | 0.7969 | 0.8739 | 0.4754 | 19 |
| full_mixed | bspline / cubic_bspline / thin_plate / rbf | 0.7950 | 0.7982 | 0.7902 | 0.7950 | 0.7950 | 0.8705 | 0.4720 | 19 |

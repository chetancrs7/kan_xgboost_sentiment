"""Self-contained TF-IDF + XGBoost sentiment model.

This package is deliberately independent of the other XGBoost expert
(``src/models/xgboost``) so it can be deleted wholesale with no side effects:

    rm -r src/models/xgboost_tfidf models/xgboost_tfidf

Unlike the "expert" model, which feeds XGBoost ~30 hand-crafted linguistic
features, this model vectorises the *actual tweet text* with TF-IDF word and
character n-grams and trains on the full 80k-row training split. That is what
lets a gradient-boosted model learn real lexical sentiment signal.
"""

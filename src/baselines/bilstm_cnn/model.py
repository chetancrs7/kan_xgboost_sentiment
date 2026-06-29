"""Model setup, tuning, and training helpers for the BiLSTM-CNN baseline.

The model combines a bidirectional LSTM for contextual sequence encoding with a
1D convolution layer for local n-gram pattern extraction, then predicts
sentiment with a softmax classifier.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow.keras import Model  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.callbacks import (  # type: ignore[reportMissingModuleSource]
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import (  # type: ignore[reportMissingModuleSource]
        LSTM,
        BatchNormalization,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPooling1D,
        Input,
        SpatialDropout1D,
    )
    from tensorflow.keras.preprocessing.sequence import (  # type: ignore[reportMissingModuleSource]
        pad_sequences,
    )
    from tensorflow.keras.preprocessing.text import (  # type: ignore[reportMissingModuleSource]
        Tokenizer,
    )
except ImportError as exc:  # pragma: no cover - only reached when dependencies miss
    raise ImportError(
        "BiLSTM-CNN requires TensorFlow. Install dependencies with "
        "`pip install -r library_requirements.txt`."
    ) from exc

# Preprocessing defaults.
VOCAB_SIZE: int = 20_000
MAX_LENGTH: int = 200
OOV_TOKEN: str = "<OOV>"
PADDING: str = "post"
TRUNCATING: str = "post"
LOWERCASE: bool = True

# Training defaults.
BATCH_SIZE: int = 64
EPOCHS: int = 15
TUNING_EPOCHS: int = 3
LEARNING_RATE: float = 0.001
VALIDATION_SPLIT: float = 0.15
EARLY_STOPPING_PATIENCE: int = 2
VERBOSE: int = 1
RANDOM_STATE: int = 42
SCORING: str = "f1_macro"

# Fixed architecture hyperparameters.
EMBEDDING_DIM: int = 128
FILTERS: int = 128
KERNEL_SIZE: int = 3
DENSE_UNITS: int = 64

# Tuned architecture hyperparameters (grid searched on the validation split).
LSTM_UNITS_VALUES: tuple[int, ...] = (64, 128)
DROPOUT_RATES: tuple[float, ...] = (0.3, 0.5)

ArrayLike = np.ndarray


@dataclass(frozen=True)
class BiLSTMCNNParams:
    embedding_dim: int
    lstm_units: int
    filters: int
    kernel_size: int
    dense_units: int
    dropout_rate: float


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def param_candidates(
    lstm_units_values: tuple[int, ...] = LSTM_UNITS_VALUES,
    dropout_rates: tuple[float, ...] = DROPOUT_RATES,
) -> list[BiLSTMCNNParams]:
    """Enumerate the architecture configurations to evaluate during tuning."""
    return [
        BiLSTMCNNParams(
            embedding_dim=EMBEDDING_DIM,
            lstm_units=int(lstm_units),
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            dense_units=DENSE_UNITS,
            dropout_rate=float(dropout_rate),
        )
        for lstm_units in lstm_units_values
        for dropout_rate in dropout_rates
    ]


def fit_tokenizer(
    train_texts: pd.Series,
    vocab_size: int = VOCAB_SIZE,
) -> Tokenizer:
    """Fit a Keras tokenizer on the training texts only."""
    if train_texts.empty:
        raise ValueError("Cannot fit tokenizer on empty training data.")

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token=OOV_TOKEN,
        lower=LOWERCASE,
    )
    tokenizer.fit_on_texts(train_texts.tolist())
    return tokenizer


def vectorize_texts(
    tokenizer: Tokenizer,
    texts: pd.Series,
    max_length: int = MAX_LENGTH,
) -> ArrayLike:
    """Convert texts to padded integer sequences using a fitted tokenizer."""
    sequences = tokenizer.texts_to_sequences(texts.tolist())
    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding=PADDING,
        truncating=TRUNCATING,
    )


def build_model(
    params: BiLSTMCNNParams,
    num_classes: int,
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Construct and compile the BiLSTM-CNN classifier."""
    inputs = Input(shape=(max_length,), name="token_ids")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=params.embedding_dim,
        name="token_embedding",
    )(inputs)
    x = SpatialDropout1D(params.dropout_rate, name="embedding_dropout")(x)
    x = Bidirectional(
        LSTM(
            params.lstm_units,
            return_sequences=True,
            dropout=params.dropout_rate,
            recurrent_dropout=0.0,
        ),
        name="bilstm",
    )(x)
    x = Conv1D(
        filters=params.filters,
        kernel_size=params.kernel_size,
        activation="relu",
        padding="same",
        name="cnn_ngram_features",
    )(x)
    x = GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = BatchNormalization(name="feature_normalization")(x)
    x = Dense(params.dense_units, activation="relu", name="dense_features")(x)
    x = Dropout(params.dropout_rate, name="classifier_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="sentiment")(x)

    model = Model(inputs=inputs, outputs=outputs, name="bilstm_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(
    patience: int = EARLY_STOPPING_PATIENCE,
) -> list[tf.keras.callbacks.Callback]:
    """Early stopping and learning-rate scheduling on validation loss."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-6,
        ),
    ]


def class_weights(labels: ArrayLike) -> dict[int, float]:
    """Compute balanced class weights to offset label imbalance."""
    classes = np.sort(np.unique(labels))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def _fit(
    model: Model,
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    epochs: int,
) -> Model:
    """Fit a model with early stopping against an explicit validation split."""
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=build_callbacks(),
        class_weight=class_weights(y_train),
        verbose=VERBOSE,
    )
    return model


def tune_model(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    num_classes: int,
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    candidates: list[BiLSTMCNNParams] | None = None,
) -> BiLSTMCNNParams:
    """Grid-search architecture params on the validation split (macro-F1)."""
    candidates = candidates if candidates is not None else param_candidates()
    if len(candidates) == 1:
        print(f"[INFO] Single candidate; skipping tuning: {candidates[0]}")
        return candidates[0]

    print(
        f"[INFO] Tuning BiLSTM-CNN on validation split "
        f"({len(candidates)} candidates, tuning_epochs={TUNING_EPOCHS}, "
        f"scoring={SCORING}) ..."
    )

    best_score = -np.inf
    best_params = candidates[0]

    for params in candidates:
        model = build_model(
            params,
            num_classes=num_classes,
            vocab_size=vocab_size,
            max_length=max_length,
        )
        _fit(model, x_train, y_train, x_val, y_val, epochs=TUNING_EPOCHS)
        y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
        score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"[INFO]   {params} -> val macro-F1={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
        tf.keras.backend.clear_session()

    print(f"[INFO] Best validation {SCORING}: {best_score:.4f}")
    print(f"[INFO] Best parameters: {best_params}")
    return best_params


def train_final_model(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    best_params: BiLSTMCNNParams,
    num_classes: int,
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    epochs: int = EPOCHS,
) -> Model:
    """Train the selected configuration to convergence on the training split."""
    print(
        f"[INFO] Fitting final model on {len(x_train):,} samples "
        f"(validation={len(x_val):,}, epochs={epochs}) ..."
    )
    model = build_model(
        best_params,
        num_classes=num_classes,
        vocab_size=vocab_size,
        max_length=max_length,
    )
    _fit(model, x_train, y_train, x_val, y_val, epochs=epochs)
    print("[INFO] Training complete.")
    return model

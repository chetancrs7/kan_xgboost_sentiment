# TODO: Remove this file once testing is done for the split model/train/evaluate files
"""BiLSTM-CNN baseline for sentiment classification.

The model combines a bidirectional LSTM for contextual sequence encoding with a
1D convolution layer for local n-gram pattern extraction, then predicts
sentiment with a softmax classifier.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow.keras import Model  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore[reportMissingModuleSource]
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
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore[reportMissingModuleSource]
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore[reportMissingModuleSource]
except ImportError as exc:  # pragma: no cover - only reached when dependencies miss
    raise ImportError(
        "BiLSTM-CNN requires TensorFlow. Install dependencies with "
        "`pip install -r library_requirements.txt`."
    ) from exc


ArrayLike = np.ndarray


@dataclass(frozen=True)
class BiLSTMCNNParams:
    embedding_dim: int
    lstm_units: int
    filters: int
    kernel_size: int
    dense_units: int
    dropout_rate: float


class BiLSTMCNNTrainer:
    """Train, tune, evaluate, and save a BiLSTM-CNN sentiment baseline."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_config = config.get("dataset", {})
        self.preprocessing_config = config.get("preprocessing", {})
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})
        self.output_config = config.get("output", {})
        self.random_state = int(config.get("random_state", 42))
        self.num_classes = int(config.get("num_classes", 0))

        self._set_reproducibility(self.random_state)

        self.tokenizer: Tokenizer | None = None
        self.model: Model | None = None
        self.best_params: BiLSTMCNNParams | None = None

    def run(self) -> dict[str, Any]:
        texts, labels = self.load_dataset()
        train_texts, test_texts, train_labels, test_labels = self.split_dataset(
            texts,
            labels,
        )

        self.tokenizer = self.fit_tokenizer(train_texts)
        x_train = self.vectorize_texts(train_texts)
        x_test = self.vectorize_texts(test_texts)

        self.best_params = self.select_model_params(x_train, train_labels)
        self.model = self.train_model(
            x_train,
            train_labels,
            self.best_params,
            epochs=int(self.training_config.get("epochs", 15)),
        )

        probabilities = self.model.predict(
            x_test,
            batch_size=int(self.training_config.get("batch_size", 64)),
            verbose=0,
        )
        predictions = np.argmax(probabilities, axis=1)
        metrics = self.evaluate(test_labels, predictions)

        result = {
            "model_name": self.config.get("model_name", "bilstm_cnn"),
            "best_params": asdict(self.best_params),
            "metrics": metrics,
        }

        self.save_artifacts(result)
        print(json.dumps(result, indent=2))
        return result

    def load_dataset(self) -> tuple[pd.Series, pd.Series]:
        dataset_path = Path(self.dataset_config.get("path", ""))
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        has_header = bool(self.dataset_config.get("header", False))
        dataframe = pd.read_csv(
            dataset_path,
            encoding=self.dataset_config.get("encoding", "latin-1"),
            header=0 if has_header else None,
        )

        if not has_header and dataframe.shape[1] >= 6:
            dataframe = dataframe.iloc[:, :6]
            dataframe.columns = ["target", "id", "date", "query", "user", "text"]

        text_column = self.dataset_config.get("text_column", "text")
        label_column = self.dataset_config.get("label_column", "target")
        if text_column not in dataframe.columns or label_column not in dataframe.columns:
            raise ValueError(
                f"Expected columns '{text_column}' and '{label_column}'. "
                f"Available columns: {list(dataframe.columns)}"
            )

        dataframe = dataframe[[text_column, label_column]].dropna()
        dataframe[text_column] = dataframe[text_column].astype(str).map(self.clean_text)
        dataframe = dataframe[dataframe[text_column].str.len() > 0]
        if dataframe.empty:
            raise ValueError("Dataset contains no non-empty text rows after cleaning.")

        labels = self.normalize_labels(dataframe[label_column])
        dataframe = dataframe.assign(label=labels).dropna(subset=["label"])
        dataframe["label"] = dataframe["label"].astype(int)

        sample_size = self.dataset_config.get("sample_size")
        if sample_size:
            dataframe = self.stratified_sample(dataframe, int(sample_size))

        self.validate_labels(dataframe["label"])
        self.num_classes = max(self.num_classes, int(dataframe["label"].nunique()))
        return dataframe[text_column].reset_index(drop=True), dataframe["label"].reset_index(
            drop=True
        )

    def split_dataset(
        self,
        texts: pd.Series,
        labels: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        if len(texts) != len(labels):
            raise ValueError("Text and label counts do not match.")

        return train_test_split(
            texts,
            labels,
            test_size=float(self.training_config.get("test_size", 0.2)),
            random_state=self.random_state,
            stratify=labels,
        )

    def fit_tokenizer(self, train_texts: pd.Series) -> Tokenizer:
        if train_texts.empty:
            raise ValueError("Cannot fit tokenizer on empty training data.")

        tokenizer = Tokenizer(
            num_words=int(self.preprocessing_config.get("vocab_size", 20000)),
            oov_token=self.preprocessing_config.get("oov_token", "<OOV>"),
            lower=bool(self.preprocessing_config.get("lowercase", True)),
        )
        tokenizer.fit_on_texts(train_texts.tolist())
        return tokenizer

    def vectorize_texts(self, texts: pd.Series) -> ArrayLike:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been fitted.")

        sequences = self.tokenizer.texts_to_sequences(texts.tolist())
        return pad_sequences(
            sequences,
            maxlen=int(self.preprocessing_config.get("max_length", 200)),
            padding=self.preprocessing_config.get("padding", "post"),
            truncating=self.preprocessing_config.get("truncating", "post"),
        )

    def select_model_params(self, x_train: ArrayLike, y_train: pd.Series) -> BiLSTMCNNParams:
        candidates = self.model_param_candidates()
        if len(candidates) == 1:
            return candidates[0]

        train_x, val_x, train_y, val_y = train_test_split(
            x_train,
            y_train,
            test_size=float(self.training_config.get("validation_split", 0.15)),
            random_state=self.random_state,
            stratify=y_train,
        )

        best_score = -np.inf
        best_params = candidates[0]
        tuning_epochs = int(self.training_config.get("tuning_epochs", 3))

        for params in candidates:
            model = self.train_model(train_x, train_y, params, epochs=tuning_epochs)
            predictions = np.argmax(model.predict(val_x, verbose=0), axis=1)
            _, _, macro_f1, _ = precision_recall_fscore_support(
                val_y,
                predictions,
                average="macro",
                zero_division=0,
            )
            if macro_f1 > best_score:
                best_score = macro_f1
                best_params = params
            tf.keras.backend.clear_session()

        return best_params

    def train_model(
        self,
        x_train: ArrayLike,
        y_train: pd.Series,
        params: BiLSTMCNNParams,
        epochs: int,
    ) -> Model:
        if x_train.shape[0] != len(y_train):
            raise ValueError("Training features and labels have mismatched lengths.")

        model = self.build_model(params)
        model.fit(
            x_train,
            y_train.to_numpy(),
            batch_size=int(self.training_config.get("batch_size", 64)),
            epochs=epochs,
            validation_split=float(self.training_config.get("validation_split", 0.15)),
            callbacks=self.callbacks(),
            class_weight=self.class_weights(y_train),
            verbose=int(self.training_config.get("verbose", 1)),
        )
        return model

    def build_model(self, params: BiLSTMCNNParams) -> Model:
        vocab_size = int(self.preprocessing_config.get("vocab_size", 20000))
        max_length = int(self.preprocessing_config.get("max_length", 200))

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
        outputs = Dense(
            self.num_classes or 2,
            activation="softmax",
            name="sentiment",
        )(x)

        model = Model(inputs=inputs, outputs=outputs, name="bilstm_cnn")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=float(self.training_config.get("learning_rate", 0.001))
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def callbacks(self) -> list[tf.keras.callbacks.Callback]:
        patience = int(self.training_config.get("early_stopping_patience", 2))
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

    def class_weights(self, labels: pd.Series) -> dict[int, float]:
        classes = np.sort(labels.unique())
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=labels.to_numpy(),
        )
        return {int(label): float(weight) for label, weight in zip(classes, weights)}

    def evaluate(self, y_true: pd.Series, y_pred: ArrayLike) -> dict[str, Any]:
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(weighted_precision),
            "recall_weighted": float(weighted_recall),
            "f1_weighted": float(weighted_f1),
            "precision_macro": float(macro_precision),
            "recall_macro": float(macro_recall),
            "f1_macro": float(macro_f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
                output_dict=True,
            ),
        }

    def save_artifacts(self, result: dict[str, Any]) -> None:
        save_dir = Path(self.output_config.get("save_dir", "outputs/bilstm_cnn"))
        save_dir.mkdir(parents=True, exist_ok=True)

        with (save_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        if self.tokenizer is not None:
            joblib.dump(self.tokenizer, save_dir / "tokenizer.joblib")
        if self.model is not None:
            self.model.save(save_dir / "bilstm_cnn.keras")

    def model_param_candidates(self) -> list[BiLSTMCNNParams]:
        return [
            BiLSTMCNNParams(
                embedding_dim=int(embedding_dim),
                lstm_units=int(lstm_units),
                filters=int(filters),
                kernel_size=int(kernel_size),
                dense_units=int(dense_units),
                dropout_rate=float(dropout_rate),
            )
            for embedding_dim in self.as_list(self.model_config.get("embedding_dim", 128))
            for lstm_units in self.as_list(self.model_config.get("lstm_units", 128))
            for filters in self.as_list(self.model_config.get("filters", 128))
            for kernel_size in self.as_list(self.model_config.get("kernel_size", 3))
            for dense_units in self.as_list(self.model_config.get("dense_units", 64))
            for dropout_rate in self.as_list(self.model_config.get("dropout_rate", 0.5))
        ]

    def stratified_sample(self, dataframe: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        sample_size = min(sample_size, len(dataframe))
        # Sample each class group explicitly instead of groupby().apply();
        # pandas 3.0 excludes the grouping column from apply, which would
        # silently drop 'label' from the result.
        parts = [
            group.sample(
                n=max(1, round(sample_size * len(group) / len(dataframe))),
                random_state=self.random_state,
            )
            for _, group in dataframe.groupby("label")
        ]
        sampled = pd.concat(parts).sample(frac=1.0, random_state=self.random_state)
        return sampled.head(sample_size).reset_index(drop=True)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_labels(labels: pd.Series) -> pd.Series:
        numeric_labels = pd.to_numeric(labels, errors="coerce")
        unique_values = set(numeric_labels.dropna().astype(int).unique())
        if unique_values.issubset({0, 4}):
            return numeric_labels.map({0: 0, 4: 1})
        if unique_values.issubset({0, 1}):
            return numeric_labels
        category_codes, _ = pd.factorize(labels)
        return pd.Series(category_codes, index=labels.index)

    @staticmethod
    def validate_labels(labels: pd.Series) -> None:
        class_counts = labels.value_counts()
        if len(class_counts) < 2:
            raise ValueError("Sentiment classification requires at least two classes.")
        if class_counts.min() < 2:
            raise ValueError("Each class needs at least two rows.")

    @staticmethod
    def as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _set_reproducibility(seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

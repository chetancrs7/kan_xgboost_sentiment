from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import contractions
from urlextract import URLExtract
from emot.emo_unicode import EMOTICONS_EMO
import nltk
from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon')

extractor = URLExtract()

RAW_COLUMNS = ["target", "tweet_id", "date", "query", "user", "text"]
LABEL_MAP = {0: 0, 4: 1}

# TODO: Use library to check URL?
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
TOKEN_RE = re.compile(r"[a-z0-9_<>']+|[!?]+")
EMOTICON_RE = re.compile(r"(?::|;|=|x)(?:-)?(?:\)|\(|d|p|/|\\)", re.IGNORECASE)

POSITIVE_WORDS = set(opinion_lexicon.positive())
NEGATIVE_WORDS = set(opinion_lexicon.negative())

NEGATORS = {"no", "not", "never", "none", "cannot", "can't", "dont", "don't", "wont", "won't"}
INTENSIFIERS = {"so", "very", "really", "too", "super", "extremely", "totally", "absolutely"}


@dataclass(frozen=True)
class PreprocessConfig:
    raw_path: str
    output_dir: str
    sample_size: int | None
    random_state: int
    test_size: float
    val_size: float
    kan_max_features: int
    xgb_word_max_features: int
    xgb_char_max_features: int
    min_df: int


def normalize_text(text: str) -> str:
    text = str(text).replace("&amp;", " and ").replace("&lt;", " < ").replace("&gt;", " > ")
    text = text.lower()

    urls = extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, " <url> ")

    text = MENTION_RE.sub(" <user> ", text)
    emoticons = extract_emoticons(text)

    for emo in emoticons:
        text = text.replace(emo, " <emoticon> ")

    text = HASHTAG_RE.sub(r" <hashtag> \1 ", text)
    text = expand_contractions(text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)

    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def extract_hashtags(text: str) -> str:
    return " ".join(match.group(1).lower() for match in HASHTAG_RE.finditer(str(text)))


def tokenize(clean_text: str) -> list[str]:
    return clean_text.split()


def count_matches(tokens: Iterable[str], lexicon: set[str]) -> int:
    return sum(token in lexicon for token in tokens)

def extract_emoticons(text: str) -> list[str]:
    text = str(text)
    return [emo for emo in EMOTICONS_EMO if emo in text]

def engineer_dense_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_text, clean_text in zip(df["text"], df["clean_text"]):
        raw = str(raw_text)
        tokens = tokenize(clean_text)
        token_count = len(tokens)
        char_count = len(raw)
        pos_count = count_matches(tokens, POSITIVE_WORDS)
        neg_count = count_matches(tokens, NEGATIVE_WORDS)

        rows.append(
            {
                "char_count": char_count,
                "token_count": token_count,
                "avg_token_length": np.mean([len(token) for token in tokens]) if tokens else 0.0,
                "url_count": len(extractor.find_urls(raw)),
                "mention_count": len(MENTION_RE.findall(raw)),
                "hashtag_count": len(HASHTAG_RE.findall(raw)),
                "emoticon_count": len(extract_emoticons(raw)),
                "exclamation_count": raw.count("!"),
                "question_count": raw.count("?"),
                "uppercase_ratio": sum(ch.isupper() for ch in raw) / max(sum(ch.isalpha() for ch in raw), 1),
                "elongated_count": len(REPEAT_CHAR_RE.findall(raw.lower())),
                "positive_lexicon_count": pos_count,
                "negative_lexicon_count": neg_count,
                "lexicon_polarity": (pos_count - neg_count) / max(pos_count + neg_count, 1),
                "negator_count": count_matches(tokens, NEGATORS),
                "intensifier_count": count_matches(tokens, INTENSIFIERS),
                "has_user_token": int("<user>" in tokens),
                "has_url_token": int("<url>" in tokens),
                "has_hashtag_token": int("<hashtag>" in tokens),
            }
        )
    return pd.DataFrame(rows, index=df.index)


def split_indices(y: np.ndarray, test_size: float, val_size: float, random_state: int) -> dict[str, np.ndarray]:
    all_idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y[train_val_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_sparse_splits(matrix: sparse.spmatrix, split_map: dict[str, np.ndarray], output_dir: Path, prefix: str) -> None:
    for split_name, idx in split_map.items():
        sparse.save_npz(output_dir / f"{prefix}_{split_name}.npz", matrix[idx])


def save_dense_splits(matrix: np.ndarray, split_map: dict[str, np.ndarray], output_dir: Path, prefix: str) -> None:
    for split_name, idx in split_map.items():
        np.save(output_dir / f"{prefix}_{split_name}.npy", matrix[idx].astype(np.float32))


def run_preprocessing(config: PreprocessConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    df = pd.read_csv(
        config.raw_path,
        encoding="latin-1",
        names=RAW_COLUMNS,
        usecols=["target", "tweet_id", "date", "user", "text"],
    )
    df["label"] = df["target"].map(LABEL_MAP).astype(np.int8)

    if config.sample_size is not None and config.sample_size < len(df):
        df = (
            df.groupby("label", group_keys=False)
            .sample(n=config.sample_size // 2, random_state=config.random_state)
            .sample(frac=1.0, random_state=config.random_state)
            .reset_index(drop=True)
        )

    df["clean_text"] = df["text"].map(normalize_text)
    df["hashtag_text"] = df["text"].map(extract_hashtags)
    dense_features = engineer_dense_features(df)
    y = df["label"].to_numpy()
    split_map = split_indices(y, config.test_size, config.val_size, config.random_state)

    kan_text_vectorizer = TfidfVectorizer(
        max_features=config.kan_max_features,
        min_df=config.min_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    xgb_word_vectorizer = TfidfVectorizer(
        max_features=config.xgb_word_max_features,
        min_df=config.min_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        binary=True,
        norm=None,
    )
    xgb_char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        max_features=config.xgb_char_max_features,
        min_df=config.min_df,
        ngram_range=(3, 5),
        sublinear_tf=True,
        norm=None,
    )

    train_text = df.loc[split_map["train"], "clean_text"]
    kan_text_vectorizer.fit(train_text)
    xgb_word_vectorizer.fit(train_text)
    xgb_char_vectorizer.fit(train_text)

    kan_text = kan_text_vectorizer.transform(df["clean_text"]).astype(np.float32)
    xgb_word = xgb_word_vectorizer.transform(df["clean_text"]).astype(np.float32)
    xgb_char = xgb_char_vectorizer.transform(df["clean_text"]).astype(np.float32)

    dense_scaler_for_kan = StandardScaler()
    dense_scaler_for_xgb = MinMaxScaler()
    dense_scaler_for_kan.fit(dense_features.iloc[split_map["train"]])
    dense_scaler_for_xgb.fit(dense_features.iloc[split_map["train"]])

    kan_dense = dense_scaler_for_kan.transform(dense_features).astype(np.float32)
    xgb_dense = sparse.csr_matrix(dense_scaler_for_xgb.transform(dense_features).astype(np.float32))

    kan_matrix = sparse.hstack([kan_text, sparse.csr_matrix(kan_dense)], format="csr")
    xgb_matrix = sparse.hstack([xgb_word, xgb_char, xgb_dense], format="csr")

    save_sparse_splits(kan_matrix, split_map, output_dir, "kan_features")
    save_sparse_splits(xgb_matrix, split_map, output_dir, "xgb_features")
    save_dense_splits(y, split_map, output_dir, "labels")

    audit_columns = ["tweet_id", "date", "user", "text", "clean_text", "hashtag_text", "label"]
    df[audit_columns].to_csv(output_dir / "clean_tweets.csv.gz", index=False, compression="gzip")
    dense_features.to_csv(output_dir / "engineered_dense_features.csv.gz", index=False, compression="gzip")

    for split_name, idx in split_map.items():
        np.save(output_dir / f"{split_name}_indices.npy", idx.astype(np.int64))

    joblib.dump(kan_text_vectorizer, output_dir / "models" / "kan_text_vectorizer.joblib")
    joblib.dump(xgb_word_vectorizer, output_dir / "models" / "xgb_word_vectorizer.joblib")
    joblib.dump(xgb_char_vectorizer, output_dir / "models" / "xgb_char_vectorizer.joblib")
    joblib.dump(dense_scaler_for_kan, output_dir / "models" / "kan_dense_scaler.joblib")
    joblib.dump(dense_scaler_for_xgb, output_dir / "models" / "xgb_dense_scaler.joblib")

    metadata = {
        "config": asdict(config),
        "label_map": {"negative": 0, "positive": 1},
        "rows": int(len(df)),
        "splits": {name: int(len(idx)) for name, idx in split_map.items()},
        "class_distribution": df["label"].value_counts().sort_index().astype(int).to_dict(),
        "kan_feature_count": int(kan_matrix.shape[1]),
        "xgb_feature_count": int(xgb_matrix.shape[1]),
        "dense_feature_names": dense_features.columns.tolist(),
        "fusion_contract": {
            "row_alignment": "All split feature files and label files use the same row order within each split.",
            "kan_inputs": "Continuous sparse TF-IDF plus standardized dense sentiment/style features.",
            "xgb_inputs": "Binary/sublinear word TF-IDF, character n-grams, and min-max dense indicators.",
            "calibration_note": "Fit probability calibration on validation predictions only, then train residual-aware fusion on validation out-of-fold style outputs.",
        },
    }
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess Sentiment140 for KAN + XGBoost hybrid sentiment modeling.")
    parser.add_argument("--raw-path", default="data/raw/sentiment140_dataset.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--kan-max-features", type=int, default=20000)
    parser.add_argument("--xgb-word-max-features", type=int, default=50000)
    parser.add_argument("--xgb-char-max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=5)
    args = parser.parse_args()
    return PreprocessConfig(**vars(args))


if __name__ == "__main__":
    run_preprocessing(parse_args())

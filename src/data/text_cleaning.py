from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import contractions
import pandas as pd
from emot.emo_unicode import EMOTICONS_EMO
from urlextract import URLExtract

from feature_extraction import EMOTICON_RE, add_all_features

# Repo root (src/data/ -> parents[2]) so default paths resolve to the project's
# data/ dir regardless of the launch directory (needed for cloud/HPC runs).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_COLUMNS = ["target", "tweet_id", "date", "query", "user", "text"]
LABEL_MAP = {0: 0, 4: 1}

MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
TOKEN_RE = re.compile(r"[a-z0-9_<>']+|[!?]+")

url_extractor = URLExtract()


@dataclass(frozen=True)
class PreprocessConfig:
    raw_path: str
    output_path: str
    sample_size: int | None
    random_state: int
    with_features: bool = False
    use_spacy: bool = True


def extract_hashtags(text: str) -> str:
    return " ".join(match.group(1).lower() for match in HASHTAG_RE.finditer(str(text)))


def clean_text(text: str) -> tuple[str, str]:

    text = str(text)

    text = text.replace("&amp;", " and ")
    text = text.replace("&lt;", " < ")
    text = text.replace("&gt;", " > ")

    # URLs first: a URL contains emoticon-like substrings (e.g. "://" holds
    # ":/"), so strip them before emoticon detection runs.
    for url in url_extractor.find_urls(text):
        text = text.replace(url, " <url> ")

    # Emoticons MUST be detected on the original-case text: most are
    # case-sensitive (":D", ":P", "XD", "D:") and would never match a
    # lower-cased string. EMOTICON_RE is longest-first, so ":))" wins over
    # ":)"; dict.fromkeys dedups the meanings while preserving order.
    emoticon_meanings = [
        EMOTICONS_EMO[e].replace(":", "").replace(",", "").replace("_", " ").lower()
        for e in dict.fromkeys(EMOTICON_RE.findall(text))
    ]
    text = EMOTICON_RE.sub(" <emoticon> ", text)

    text = text.lower()

    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r" <hashtag> \1 ", text)

    text = contractions.fix(text)

    text = REPEAT_CHAR_RE.sub(r"\1\1", text)

    tokens = TOKEN_RE.findall(text)

    clean_text_result = " ".join(tokens)
    emoticon_text_result = " ".join(emoticon_meanings)

    return clean_text_result, emoticon_text_result


def load_raw_tweets(raw_path: str) -> pd.DataFrame:
    return pd.read_csv(
        raw_path,
        encoding="latin-1",
        names=RAW_COLUMNS,
        usecols=["target", "text"],
    )


def preprocess_tweets(config: PreprocessConfig) -> pd.DataFrame:
    df = load_raw_tweets(config.raw_path)
    df["label"] = df["target"].map(LABEL_MAP).astype("int8")

    if config.sample_size is not None and config.sample_size < len(df):
        df = df.sample(
            n=config.sample_size, random_state=config.random_state
        ).reset_index(drop=True)

    # --- Feature extraction FIRST: every feature is read from the raw `text`
    # column, because clean_text() lower-cases and strips punctuation/emoji and
    # would otherwise destroy the source signal. ---
    if config.with_features:
        df = add_all_features(df, text_col="text", use_spacy=config.use_spacy)

    # --- Destructive cleaning second: produces the model text column. ---
    processed = df["text"].apply(clean_text)

    df["clean_text"] = processed.apply(lambda x: x[0])

    if config.with_features:
        # The feature stage already owns `emoticon_text` (from emoticon_features)
        # and `hashtag_text` (from hashtag_features); don't recompute/collide.
        base_cols = ["text", "clean_text", "hashtag_text", "emoticon_text", "label"]
        feature_cols = [
            c for c in df.columns
            if c not in base_cols and c != "target"
        ]
        return df[base_cols + feature_cols]

    df["emoticon_text"] = processed.apply(lambda x: x[1])
    df["hashtag_text"] = df["text"].map(extract_hashtags)
    return df[["text", "clean_text", "hashtag_text", "emoticon_text", "label"]]


def save_processed_tweets(df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def run_preprocessing(config: PreprocessConfig) -> None:
    processed_df = preprocess_tweets(config)
    save_processed_tweets(processed_df, config.output_path)


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Clean raw Sentiment140 tweets and save a processed CSV file."
    )
    parser.add_argument("--raw-path", default=str(PROJECT_ROOT / "data/raw/sentiment140_dataset.csv"))
    parser.add_argument("--output-path", default=str(PROJECT_ROOT / "data/processed/cleaned_tweets.csv"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=22)
    parser.add_argument("--with-features", action="store_true",
                        help="Extract the 10 sentiment-feature categories into columns.")
    parser.add_argument("--no-spacy", dest="use_spacy", action="store_false",
                        help="Skip spaCy; use NLTK for POS and zero-fill dependency feats.")
    args = parser.parse_args()
    return PreprocessConfig(**vars(args))


if __name__ == "__main__":
    run_preprocessing(parse_args())

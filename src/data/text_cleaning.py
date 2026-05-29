from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import contractions
import pandas as pd
from emot.emo_unicode import EMOTICONS_EMO
from urlextract import URLExtract

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


def extract_emoticon_meanings(text: str) -> list[str]:
    text = str(text)

    meanings = []

    for emoticon, meaning in EMOTICONS_EMO.items():

        if emoticon in text:

            clean_meaning = (
                meaning.replace(":", "").replace(",", "").replace("_", " ").lower()
            )

            meanings.append(clean_meaning)

    return meanings


def extract_hashtags(text: str) -> str:
    return " ".join(match.group(1).lower() for match in HASHTAG_RE.finditer(str(text)))


def clean_text(text: str) -> tuple[str, str]:

    text = str(text)

    text = text.replace("&amp;", " and ")
    text = text.replace("&lt;", " < ")
    text = text.replace("&gt;", " > ")

    text = text.lower()

    emoticon_meanings = []

    for url in url_extractor.find_urls(text):
        text = text.replace(url, " <url> ")

    text = MENTION_RE.sub(" <user> ", text)

    for emoticon, meaning in EMOTICONS_EMO.items():

        if emoticon in text:

            clean_meaning = (
                meaning.replace(":", "").replace(",", "").replace("_", " ").lower()
            )

            emoticon_meanings.append(clean_meaning)

            text = text.replace(emoticon, " <emoticon> ")

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
        usecols=["target", "tweet_id", "date", "user", "text"],
    )


def preprocess_tweets(config: PreprocessConfig) -> pd.DataFrame:
    df = load_raw_tweets(config.raw_path)
    df["label"] = df["target"].map(LABEL_MAP).astype("int8")

    if config.sample_size is not None and config.sample_size < len(df):
        df = df.sample(
            n=config.sample_size, random_state=config.random_state
        ).reset_index(drop=True)

    processed = df["text"].apply(clean_text)

    df["clean_text"] = processed.apply(lambda x: x[0])

    df["emoticon_text"] = processed.apply(lambda x: x[1])

    df["hashtag_text"] = df["text"].map(extract_hashtags)

    return df[
        [
            "tweet_id",
            "date",
            "user",
            "text",
            "clean_text",
            "hashtag_text",
            "emoticon_text",
            "label",
        ]
    ]


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
    parser.add_argument("--raw-path", default="data/raw/sentiment140_dataset.csv")
    parser.add_argument("--output-path", default="data/processed/cleaned_tweets.csv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=22)
    args = parser.parse_args()
    return PreprocessConfig(**vars(args))


if __name__ == "__main__":
    run_preprocessing(parse_args())

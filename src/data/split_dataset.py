from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = "data/processed/cleaned_tweets.csv"
DEFAULT_OUTPUT_DIR = "data/splits"
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 0.80
DEFAULT_VAL_SIZE = 0.10
DEFAULT_TEST_SIZE = 0.10
DEFAULT_TEXT_COLUMN = "clean_text"
DEFAULT_LABEL_COLUMN = "label"
ROW_ID_COLUMN = "__split_row_id"


@dataclass(frozen=True)
class SplitConfig:
    input_path: str = DEFAULT_INPUT_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    train_size: float = DEFAULT_TRAIN_SIZE
    val_size: float = DEFAULT_VAL_SIZE
    test_size: float = DEFAULT_TEST_SIZE
    seed: int = DEFAULT_SEED
    text_column: str = DEFAULT_TEXT_COLUMN
    label_column: str = DEFAULT_LABEL_COLUMN
    id_column: str | None = "tweet_id"


def validate_ratios(config: SplitConfig) -> None:
    ratios = {
        "train_size": config.train_size,
        "val_size": config.val_size,
        "test_size": config.test_size,
    }

    invalid = {name: value for name, value in ratios.items() if value <= 0 or value >= 1}
    if invalid:
        raise ValueError(f"Split ratios must be between 0 and 1: {invalid}")

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0. Current sum: {total:.6f}")


def load_dataset(config: SplitConfig) -> pd.DataFrame:
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}. Run preprocessing first, for example: "
            "python src/data/text_cleaning.py"
        )

    dataframe = pd.read_csv(input_path)
    missing = {config.text_column, config.label_column} - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    before = len(dataframe)
    dataframe = dataframe.dropna(subset=[config.text_column, config.label_column]).copy()
    dataframe[config.text_column] = dataframe[config.text_column].astype(str).str.strip()
    dataframe = dataframe[dataframe[config.text_column] != ""].copy()
    dropped_empty = before - len(dataframe)

    if config.id_column and config.id_column in dataframe.columns:
        before_dedupe = len(dataframe)
        dataframe = dataframe.drop_duplicates(subset=[config.id_column], keep="first").copy()
        dropped_duplicates = before_dedupe - len(dataframe)
    else:
        dropped_duplicates = 0

    dataframe[config.label_column] = pd.to_numeric(
        dataframe[config.label_column],
        errors="raise",
    ).astype(int)
    dataframe = dataframe.reset_index(drop=True)
    dataframe[ROW_ID_COLUMN] = range(len(dataframe))

    if dataframe.empty:
        raise ValueError("No usable rows remain after dropping empty text/label values.")

    class_counts = dataframe[config.label_column].value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError("At least two label classes are required for stratified splits.")
    if class_counts.min() < 3:
        raise ValueError(
            "Each class needs at least three rows for train/val/test stratification. "
            f"Class counts: {class_counts.to_dict()}"
        )

    if dropped_empty:
        print(f"[INFO] Dropped {dropped_empty:,} rows with empty text or label.")
    if dropped_duplicates:
        print(f"[INFO] Dropped {dropped_duplicates:,} duplicate rows by '{config.id_column}'.")
    print(f"[INFO] Loaded {len(dataframe):,} usable rows from {input_path}.")
    return dataframe


def stratified_train_val_test_split(
    dataframe: pd.DataFrame,
    config: SplitConfig,
) -> dict[str, pd.DataFrame]:
    split_parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    split_names = ("train", "val", "test")
    ratios = (config.train_size, config.val_size, config.test_size)

    for label, group in dataframe.groupby(config.label_column, sort=True):
        shuffled_group = group.sample(frac=1.0, random_state=config.seed + int(label))
        counts = stratified_counts(len(shuffled_group), ratios)

        start = 0
        for split_name, count in zip(split_names, counts):
            end = start + count
            split_parts[split_name].append(shuffled_group.iloc[start:end])
            start = end

    rng = np.random.default_rng(config.seed)
    splits = {}
    for split_name in split_names:
        split_df = pd.concat(split_parts[split_name], axis=0)
        split_df = split_df.iloc[rng.permutation(len(split_df))].reset_index(drop=True)
        splits[split_name] = split_df

    validate_split_integrity(splits, config)
    return splits


def stratified_counts(n_rows: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_rows < 3:
        raise ValueError("Each class needs at least three rows for train/val/test splits.")

    raw_counts = [n_rows * ratio for ratio in ratios]
    counts = [int(np.floor(value)) for value in raw_counts]
    leftover = n_rows - sum(counts)
    fractional_order = sorted(
        range(len(ratios)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in fractional_order[:leftover]:
        counts[index] += 1

    for index, count in enumerate(counts):
        if count > 0:
            continue
        donor_index = max(range(len(counts)), key=lambda donor: counts[donor])
        if counts[donor_index] <= 1:
            raise ValueError(f"Could not allocate {n_rows} rows into split counts: {counts}")
        counts[donor_index] -= 1
        counts[index] = 1

    split_counts = tuple(counts)
    if sum(split_counts) != n_rows:
        raise ValueError(f"Could not allocate {n_rows} rows into split counts: {split_counts}")
    return split_counts


def validate_split_integrity(splits: dict[str, pd.DataFrame], config: SplitConfig) -> None:
    seen_row_ids: set[int] = set()
    for split_name, split_df in splits.items():
        row_ids = set(split_df[ROW_ID_COLUMN].astype(int).tolist())
        overlap = seen_row_ids & row_ids
        if overlap:
            raise ValueError(f"Row overlap detected in '{split_name}' split.")
        seen_row_ids.update(row_ids)

        class_counts = split_df[config.label_column].value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"Split '{split_name}' does not contain every label class.")


def save_splits(
    splits: dict[str, pd.DataFrame],
    config: SplitConfig,
) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for split_name, split_df in splits.items():
        output_path = output_dir / f"{split_name}.csv"
        split_df.drop(columns=[ROW_ID_COLUMN]).to_csv(output_path, index=False)
        output_paths[split_name] = output_path
        print(f"[INFO] Wrote {split_name}: {len(split_df):,} rows -> {output_path}")

    return output_paths


def run_split_generation(config: SplitConfig) -> dict[str, Path]:
    validate_ratios(config)
    dataframe = load_dataset(config)
    splits = stratified_train_val_test_split(dataframe, config)
    return save_splits(splits, config)


def parse_args(
    argv: Iterable[str] | None = None,
    default_selected_split: str | None = None,
) -> tuple[SplitConfig, str | None]:
    parser = argparse.ArgumentParser(
        description="Generate deterministic shared train/val/test CSV splits."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=float, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--id-column",
        default="tweet_id",
        help="Optional stable identifier used for duplicate removal when present.",
    )
    parser.add_argument(
        "--selected-split",
        choices=("train", "val", "test"),
        default=default_selected_split,
        help="Print the path for one split after all split files are generated.",
    )
    args = parser.parse_args(argv)

    config = SplitConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        text_column=args.text_column,
        label_column=args.label_column,
        id_column=args.id_column or None,
    )
    return config, args.selected_split


def main(
    argv: Iterable[str] | None = None,
    default_selected_split: str | None = None,
) -> None:
    try:
        config, selected_split = parse_args(argv, default_selected_split)
        output_paths = run_split_generation(config)
    except Exception as exc:
        sys.exit(f"[ERROR] {exc}")

    if selected_split:
        print(f"[INFO] {selected_split} split path: {output_paths[selected_split]}")


if __name__ == "__main__":
    main()

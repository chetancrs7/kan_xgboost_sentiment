# Data preprocessing — `text_cleaning.py`

Cleans raw Sentiment140 tweets and (optionally) extracts linguistic / sentiment
features into separate columns, ready for XGBoost.

This directory contains:

| File | Role |
|---|---|
| `text_cleaning.py` | cleaning pipeline + CLI entry point |
| `feature_extraction.py` | the 10 sentiment-feature categories (one callable per category) |

---

## Design: clean second, extract first

`clean_text()` is **destructive** — it lower-cases, replaces URLs/mentions,
strips punctuation and emoji, and squeezes repeated characters. That removes the
exact signal several features depend on (capitalization, punctuation, emoji).

So the pipeline order is:

1. **Extract features from the raw `text` column** (`feature_extraction.add_all_features`)
2. **Then clean** the text into the `clean_text` column

Run with `--with-features` to enable step 1; without it you get the legacy
cleaned-text-only output.

---

## Usage

Run from **this directory** so the local `import feature_extraction` resolves:

```bash
cd src/data

# cleaned text only (legacy behaviour, default)
python text_cleaning.py

# full feature extraction (needs spacy + textblob for all 10 categories)
python text_cleaning.py --with-features

# run with only the base libraries (NLTK POS; lexicon subjectivity;
# dependency columns zero-filled)
python text_cleaning.py --with-features --no-spacy
```

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--raw-path` | `data/raw/sentiment140_dataset.csv` | input CSV |
| `--output-path` | `data/processed/cleaned_tweets.csv` | output CSV |
| `--sample-size` | `None` (all 1.6M rows) | random subsample for quick runs |
| `--random-state` | `22` | sampling seed |
| `--with-features` | off | add the 10 feature categories as columns |
| `--no-spacy` | spaCy on | skip spaCy (NLTK POS, no dependency features) |

> Paths resolve relative to where you launch the command. From `src/data`,
> point at the repo-root data folders, e.g.
> `--raw-path ../../data/raw/sentiment140_dataset.csv`
> `--output-path ../../data/processed/cleaned_tweets.csv`.

**Quick smoke test** (300 rows, base libraries only):

```bash
python text_cleaning.py \
  --raw-path ../../data/raw/sentiment140_dataset.csv \
  --output-path ../../data/processed/features_sample.csv \
  --sample-size 300 --with-features --no-spacy
```

---

## Dependencies

Base libraries (in repo `library_requirements.txt`) are enough to run with
`--no-spacy`. Two optional installs unlock the remaining features:

```bash
pip install textblob                       # real subjectivity score
pip install spacy && python -m spacy download en_core_web_sm   # dependency features + better POS
```

Without them the pipeline still completes: `subjectivity` falls back to the
fraction of tokens found in NLTK VADER's sentiment lexicon, and the `dep_*`
columns are zero-filled (both logged to stdout). The negation and intensifier
word lists are also sourced from VADER (`NEGATE` / `BOOSTER_DICT`), so they need
no manual upkeep.

---

## Output columns (`--with-features`)

**Text columns**

| Column | Description |
|---|---|
| `text` | original raw tweet |
| `clean_text` | normalized text for the model |
| `hashtag_text` | hashtag words, camel-split (`#FollowFriday` → `follow friday`) |
| `emoticon_text` | meanings of text emoticons (`:)` → `happy face or smiley`) |
| `label` | 0 = negative, 1 = positive |

**Feature columns**

| Category | Columns |
|---|---|
| Negation | `neg_count`, `neg_ratio` |
| Intensifier | `intensifier_count`, `intensifier_ratio`, `elongated_word_count` |
| Emoticon | `emoticon_count` |
| Sarcasm | `sarcasm_cue_count`, `sarcasm_flag` |
| Subjectivity | `subjectivity` |
| POS stats | `pos_adj_ratio`, `pos_adv_ratio`, `pos_verb_ratio`, `pos_noun_ratio`, `pos_pron_ratio`, `pos_intj_ratio` |
| Dependency | `dep_neg_count`, `dep_amod_count`, `dep_advmod_count`, `dep_max_depth`, `dep_mean_depth` |
| Punctuation | `punct_count`, `punct_density`, `exclaim_count`, `question_count`, `ellipsis_count`, `interrobang_count` |
| Capitalization | `caps_ratio`, `allcaps_word_count`, `allcaps_word_ratio` |
| Hashtag sentiment | `hashtag_count`, `hashtag_sentiment`, `hashtag_pos_count`, `hashtag_neg_count` |

Without `--with-features` the output is just:
`text`, `clean_text`, `hashtag_text`, `emoticon_text`, `label`.

---

## Using `feature_extraction.py` directly

Each category is an independently callable function; orchestrators add columns to
a DataFrame:

```python
import feature_extraction as fe

fe.extract_lexical_features(text)     # dict of all lexicon/regex features (one tweet)
fe.add_lexical_features(df, "text")   # vectorized: adds lexical columns
fe.add_syntactic_features(df, "text") # batched POS + dependency pass
fe.add_all_features(df, "text")       # everything
```

The spaCy POS/dependency work runs in a single batched `nlp.pipe` pass, so it
scales to the full 1.6M-row dataset; lexical features are pure-Python and fast.

> **Sentiment140 note:** this latin-1 2009 corpus has text emoticons (`:)`,
> `:D`) but essentially no unicode emoji, so dedicated emoji columns are
> intentionally omitted. Hashtags are also rare here, so `hashtag_sentiment` is
> near-zero. The hashtag features are built and tested, and activate on modern
> tweet data.

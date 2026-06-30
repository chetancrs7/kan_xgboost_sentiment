"""Sentiment-feature extraction for XGBoost.

This module is intentionally separate from ``text_cleaning``: cleaning is
*destructive* (lower-casing, punctuation/emoji stripping, repeat squeezing),
so every feature here is computed from the **raw** tweet text *before*
``clean_text`` runs. Each feature category is a small, independently callable
function returning a flat ``dict[str, float|str]`` so columns can be added,
dropped, or unit-tested in isolation.

Two cost tiers:
  * Lexical/regex features  -> pure Python, fast, always available.
  * Syntactic features (POS / dependency) -> NLTK (POS) and, when installed,
    spaCy (POS + dependency). spaCy is run in one batched ``nlp.pipe`` pass so
    it scales to the full 1.6M-row Sentiment140 dataset.

Public entry points:
  * ``extract_lexical_features(text) -> dict``      (row-wise, all lexicon/regex feats)
  * ``add_lexical_features(df, text_col) -> df``    (vectorised wrapper)
  * ``add_syntactic_features(df, text_col) -> df``  (batched POS + dependency)
  * ``add_all_features(df, text_col) -> df``        (orchestrator)
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from emot.emo_unicode import EMOTICONS_EMO
from nltk.sentiment.vader import VaderConstants

# --------------------------------------------------------------------------- #
# Lexicons sourced from NLTK VADER (curated + maintained upstream) rather than
# hand-rolled. VaderConstants holds plain Python constants, so importing it
# needs no `vader_lexicon` download; that resource is only fetched lazily when
# the sentiment lexicon / analyzer is actually used (see `_vader`).
# --------------------------------------------------------------------------- #

# VADER's NEGATE ships both apostrophe forms for some words ("don't") but not
# all; add stripped variants so a single membership test covers "dont" too
# (our WORD_RE keeps apostrophes, so "don't" stays intact and also matches).
_NEGATE = set(VaderConstants.NEGATE)
NEGATION_WORDS = _NEGATE | {w.replace("'", "") for w in _NEGATE}

# BOOSTER_DICT mixes intensifiers (value > 0) with downtoners like "barely"/
# "hardly" (value < 0); keep only true intensifiers. Multi-word boosters
# ("kind of") can't match our single-token regex, so drop them.
INTENSIFIER_WORDS = {
    w for w, v in VaderConstants.BOOSTER_DICT.items() if v > 0 and " " not in w
}

# Hashtags / tokens that explicitly flag irony.
SARCASM_MARKERS = {"sarcasm", "sarcastic", "irony", "ironic", "notreally", "yeahright"}
SARCASM_PHRASES = (
    "yeah right", "oh great", "oh wonderful", "how nice", "just great",
    "love that", "well done", "good job", "/s",
)

# Coarse universal POS tags we report ratios for (sentiment-bearing first).
POS_KEYS = ("ADJ", "ADV", "VERB", "NOUN", "PRON", "INTJ")

# --------------------------------------------------------------------------- #
# Pre-compiled regex
# --------------------------------------------------------------------------- #

WORD_RE = re.compile(r"[a-z']+")
HASHTAG_RE = re.compile(r"#(\w+)")
# Strip URLs before emoticon detection: "://" in a link holds ":/", which emot
# would otherwise count as a skeptical emoticon.
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EXCLAIM_RE = re.compile(r"!")
QUESTION_RE = re.compile(r"\?")
ELLIPSIS_RE = re.compile(r"\.{2,}|…")
PUNCT_RE = re.compile(r"[^\w\s]")
ALLCAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
ALPHA_RE = re.compile(r"[A-Za-z]")
UPPER_RE = re.compile(r"[A-Z]")
CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_]+")

# Build a single alternation for emoticon detection (longest first to avoid
# partial matches like ":" inside ":)").
_EMOTICON_KEYS = sorted(EMOTICONS_EMO.keys(), key=len, reverse=True)
EMOTICON_RE = re.compile("|".join(re.escape(k) for k in _EMOTICON_KEYS))


# --------------------------------------------------------------------------- #
# 1. Negation  &  2. Intensifier  (lexicon membership over lowercased words)
# --------------------------------------------------------------------------- #

def negation_features(text: str) -> dict:
    words = WORD_RE.findall(str(text).lower())
    n = len(words) or 1
    count = sum(1 for w in words if w in NEGATION_WORDS)
    return {"neg_count": count, "neg_ratio": count / n}


def intensifier_features(text: str) -> dict:
    words = WORD_RE.findall(str(text).lower())
    n = len(words) or 1
    count = sum(1 for w in words if w in INTENSIFIER_WORDS)
    # elongation as an intensity signal: "soooo", "amazinggg"
    elongated = sum(1 for w in words if re.search(r"(.)\1{2,}", w))
    return {
        "intensifier_count": count,
        "intensifier_ratio": count / n,
        "elongated_word_count": elongated,
    }


# --------------------------------------------------------------------------- #
# 3. Emoticons (text emoticons via emot)  -- MUST run before clean_text
# --------------------------------------------------------------------------- #

# Sentiment140 is a latin-1 2009 corpus: it carries text emoticons (":)", ":D")
# but essentially no unicode emoji, so dedicated emoji columns are omitted.
# These polarity sets stay because sarcasm_features uses them for the
# emoji/word incongruity cue, which activates on modern emoji-bearing data.
_EMOJI_POS = set("😀😁😂🤣😃😄😆😊😍🥰😘👍❤️💕🔥✨🎉😎🙌👏💯😻🤗🥳")
_EMOJI_NEG = set("😞😔😢😭😡🤬👎💔😠😤😨😩😱🙄😒😣😖😫💀🤮😷")


def emoticon_features(text: str) -> dict:
    # Drop URLs first so ":/" inside "http://..." isn't counted as an emoticon.
    text = URL_RE.sub(" ", str(text))
    # text emoticons via emot's table; longest-first EMOTICON_RE avoids partial
    # matches (":)" inside ":))"). dict.fromkeys dedups, preserving order.
    emoticons = EMOTICON_RE.findall(text)
    emoticon_meanings = " ".join(
        EMOTICONS_EMO[e].replace(":", "").replace(",", "").replace("_", " ").lower()
        for e in dict.fromkeys(emoticons) if e in EMOTICONS_EMO
    )
    return {
        "emoticon_count": len(emoticons),
        "emoticon_text": emoticon_meanings,
    }


# --------------------------------------------------------------------------- #
# 8. Punctuation density  &  9. Capitalization ratio  -- MUST run before clean_text
# --------------------------------------------------------------------------- #

def punctuation_features(text: str) -> dict:
    text = str(text)
    total = len(text) or 1
    exclaim = len(EXCLAIM_RE.findall(text))
    question = len(QUESTION_RE.findall(text))
    ellipsis = len(ELLIPSIS_RE.findall(text))
    punct = len(PUNCT_RE.findall(text))
    return {
        "punct_count": punct,
        "punct_density": punct / total,
        "exclaim_count": exclaim,
        "question_count": question,
        "ellipsis_count": ellipsis,
        # combined "!?" / "?!" interrobang is a strong affect cue
        "interrobang_count": len(re.findall(r"[!?]{2,}", text)),
    }


def capitalization_features(text: str) -> dict:
    text = str(text)
    alpha = len(ALPHA_RE.findall(text)) or 1
    upper = len(UPPER_RE.findall(text))
    words = text.split()
    allcaps_words = len(ALLCAPS_WORD_RE.findall(text))
    return {
        "caps_ratio": upper / alpha,
        "allcaps_word_count": allcaps_words,
        "allcaps_word_ratio": allcaps_words / (len(words) or 1),
    }


# --------------------------------------------------------------------------- #
# 4. Sarcasm (heuristic cue aggregation)
# --------------------------------------------------------------------------- #

def sarcasm_features(text: str) -> dict:
    raw = str(text)
    low = raw.lower()
    cues = 0

    hashtags = {h.lower() for h in HASHTAG_RE.findall(raw)}
    if hashtags & SARCASM_MARKERS:
        cues += 2  # explicit author signal -> weight higher
    cues += sum(1 for p in SARCASM_PHRASES if p in low)

    has_pos_emoji = any(e in _EMOJI_POS for e in raw)
    has_neg_emoji = any(e in _EMOJI_NEG for e in raw)
    # positive words next to a negative emoji (or vice-versa) -> incongruity
    has_pos_word = any(w in low for w in ("love", "great", "wonderful", "perfect", "amazing"))
    has_neg_word = any(w in low for w in ("hate", "awful", "terrible", "worst", "ugh"))
    if (has_pos_word and has_neg_emoji) or (has_neg_word and has_pos_emoji):
        cues += 1
    # ellipsis after praise / quoted scare-words
    if ("..." in raw or "…" in raw) and has_pos_word:
        cues += 1
    if re.search(r'"[^"]+"', raw):
        cues += 1

    return {"sarcasm_cue_count": cues, "sarcasm_flag": int(cues >= 2)}


# --------------------------------------------------------------------------- #
# 5. Subjectivity  (TextBlob if available, else lexicon ratio fallback)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _textblob():
    try:
        from textblob import TextBlob
        return TextBlob
    except Exception:
        return None


def subjectivity_features(text: str) -> dict:
    text = str(text)
    tb = _textblob()
    if tb is not None:
        return {"subjectivity": float(tb(text).sentiment.subjectivity)}
    # fallback: fraction of tokens that carry sentiment in the VADER lexicon
    # (a word having any non-zero valence is treated as subjective).
    lexicon = _vader().lexicon
    words = WORD_RE.findall(text.lower())
    n = len(words) or 1
    return {"subjectivity": sum(1 for w in words if w in lexicon) / n}


# --------------------------------------------------------------------------- #
# 10. Hashtag sentiment  (VADER over camel-split hashtag tokens)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _vader():
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _split_hashtag(tag: str) -> str:
    return CAMEL_SPLIT_RE.sub(" ", tag).lower().strip()


def hashtag_features(text: str) -> dict:
    raw = str(text)
    tags = HASHTAG_RE.findall(raw)
    if not tags:
        return {
            "hashtag_count": 0, "hashtag_sentiment": 0.0,
            "hashtag_pos_count": 0, "hashtag_neg_count": 0, "hashtag_text": "",
        }
    sia = _vader()
    expanded = [_split_hashtag(t) for t in tags]
    scores = [sia.polarity_scores(t)["compound"] for t in expanded]
    return {
        "hashtag_count": len(tags),
        "hashtag_sentiment": sum(scores) / len(scores),
        "hashtag_pos_count": sum(1 for s in scores if s > 0.05),
        "hashtag_neg_count": sum(1 for s in scores if s < -0.05),
        "hashtag_text": " ".join(expanded),
    }


# --------------------------------------------------------------------------- #
# Orchestrators
# --------------------------------------------------------------------------- #

LEXICAL_EXTRACTORS = (
    negation_features,
    intensifier_features,
    emoticon_features,
    punctuation_features,
    capitalization_features,
    sarcasm_features,
    subjectivity_features,
    hashtag_features,
)


def extract_lexical_features(text: str) -> dict:
    """All lexicon/regex features for one raw tweet (no POS/dependency)."""
    out: dict = {}
    for fn in LEXICAL_EXTRACTORS:
        out.update(fn(text))
    return out


def add_lexical_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    feats = df[text_col].apply(extract_lexical_features).apply(pd.Series)
    return pd.concat([df, feats], axis=1)


# ---- 6. POS statistics  &  7. Dependency features (batched) --------------- #

def _spacy_nlp():
    import spacy
    # parser kept (needed for dependency feats); NER disabled for speed.
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def add_syntactic_features(
    df: pd.DataFrame,
    text_col: str = "text",
    use_spacy: bool = True,
    batch_size: int = 1000,
    n_process: int = 1,
) -> pd.DataFrame:
    """POS ratios (always) + dependency features (spaCy only), one batched pass.

    Falls back to NLTK's averaged-perceptron tagger for POS when spaCy is not
    installed; dependency columns are then filled with 0.
    """
    texts = df[text_col].astype(str).tolist()

    if use_spacy:
        try:
            nlp = _spacy_nlp()
            rows = [_spacy_row(doc) for doc in nlp.pipe(
                texts, batch_size=batch_size, n_process=n_process)]
            return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
        except OSError:
            print("[feature_extraction] spaCy model 'en_core_web_sm' not found; "
                  "run `python -m spacy download en_core_web_sm`. "
                  "Falling back to NLTK POS, dependency features = 0.")
        except ImportError:
            print("[feature_extraction] spaCy not installed; "
                  "falling back to NLTK POS, dependency features = 0.")

    rows = [_nltk_pos_row(t) for t in texts]
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


def _spacy_row(doc) -> dict:
    n = len(doc) or 1
    counts = {k: 0 for k in POS_KEYS}
    dep_neg = dep_amod = dep_advmod = 0
    depths = []
    for tok in doc:
        if tok.pos_ in counts:
            counts[tok.pos_] += 1
        if tok.dep_ == "neg":
            dep_neg += 1
        elif tok.dep_ == "amod":
            dep_amod += 1
        elif tok.dep_ == "advmod":
            dep_advmod += 1
        depths.append(_dep_depth(tok))
    row = {f"pos_{k.lower()}_ratio": counts[k] / n for k in POS_KEYS}
    row.update({
        "dep_neg_count": dep_neg,
        "dep_amod_count": dep_amod,       # adjectives modifying nouns
        "dep_advmod_count": dep_advmod,   # adverbs/intensifiers modifying heads
        "dep_max_depth": max(depths) if depths else 0,
        "dep_mean_depth": (sum(depths) / len(depths)) if depths else 0.0,
    })
    return row


def _dep_depth(tok) -> int:
    depth, cur = 0, tok
    while cur.head != cur and depth < 64:
        depth += 1
        cur = cur.head
    return depth


@lru_cache(maxsize=1)
def _nltk_tagger_ready() -> bool:
    import nltk
    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "punkt"):
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    return True


# Penn Treebank tag prefix -> coarse universal tag
_PTB2UNI = {"JJ": "ADJ", "RB": "ADV", "VB": "VERB", "NN": "NOUN", "PRP": "PRON", "UH": "INTJ"}


def _nltk_pos_row(text: str) -> dict:
    _nltk_tagger_ready()
    from nltk import pos_tag, word_tokenize
    tokens = word_tokenize(str(text))
    tags = pos_tag(tokens) if tokens else []
    n = len(tags) or 1
    counts = {k: 0 for k in POS_KEYS}
    for _, tag in tags:
        uni = _PTB2UNI.get(tag[:2])
        if uni in counts:
            counts[uni] += 1
    row = {f"pos_{k.lower()}_ratio": counts[k] / n for k in POS_KEYS}
    # dependency parser unavailable in NLTK default install -> zero-filled
    row.update({
        "dep_neg_count": 0, "dep_amod_count": 0, "dep_advmod_count": 0,
        "dep_max_depth": 0, "dep_mean_depth": 0.0,
    })
    return row


def add_all_features(
    df: pd.DataFrame,
    text_col: str = "text",
    use_spacy: bool = True,
) -> pd.DataFrame:
    """Full feature stage: lexical (row-wise) + syntactic (batched)."""
    df = add_lexical_features(df, text_col)
    df = add_syntactic_features(df, text_col, use_spacy=use_spacy)
    return df

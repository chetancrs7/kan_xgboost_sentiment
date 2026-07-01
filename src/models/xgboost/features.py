"""Linguistic feature engineering for the XGBoost expert sentiment model.

The :class:`FeatureExtractor` turns raw tweet text into a fixed-width numeric
matrix of interpretable sentiment indicators. It follows the scikit-learn
transformer protocol (``fit`` / ``transform`` / ``fit_transform``) so a single
instance can be fitted once on the training split and reused to transform the
validation and test splits with identical column ordering.

Feature groups
--------------
* Negation        - count and token-normalised ratio of negation cues.
* Intensifiers    - weighted degree-adverb / emphasis score and count.
* Emoji           - polarity sum, count, and density relative to text length.
* Sarcasm         - structural/contextual heuristics (mixed polarity, quotes...).
* Punctuation     - exclamation / question / ellipsis counts and density.
* Capitalisation  - ALL-CAPS word frequency and ratio.
* POS distribution- proportion of each universal part-of-speech category.
* Subjectivity    - opinion-heavy vs. factual language ratio.

All features are stateless lexicon/heuristic computations; ``fit`` only records
the (constant) feature schema so the transformer integrates cleanly into
scikit-learn pipelines and persists predictably.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # NLTK is a hard dependency, but degrade gracefully if data is missing.
    import nltk
    from nltk import pos_tag, word_tokenize

    _NLTK_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without NLTK installed
    _NLTK_AVAILABLE = False

try:
    import emot

    _EMOT = emot.core.emot()
except Exception:  # pragma: no cover - emot optional at runtime
    _EMOT = None


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "no", "not", "none", "never", "neither", "nor", "nobody", "nothing",
        "nowhere", "cannot", "cant", "wont", "without", "lack", "lacks",
        "hardly", "barely", "scarcely", "rarely", "seldom", "aint",
        "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
        "havent", "hasnt", "hadnt", "shouldnt", "wouldnt", "couldnt",
    }
)

# Intensifier -> multiplicative weight (>1 amplifies, <1 downtones).
INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "really": 1.5, "so": 1.4, "extremely": 2.0, "absolutely": 1.9,
    "completely": 1.7, "totally": 1.6, "incredibly": 1.9, "super": 1.6,
    "highly": 1.5, "especially": 1.4, "particularly": 1.4, "too": 1.3,
    "such": 1.3, "quite": 1.2, "most": 1.4, "more": 1.2, "much": 1.3,
    "utterly": 1.9, "deeply": 1.6, "freaking": 1.8, "damn": 1.7,
    "slightly": 0.6, "somewhat": 0.7, "barely": 0.5, "kinda": 0.7,
    "sorta": 0.7, "fairly": 0.8, "almost": 0.8, "nearly": 0.8,
}

# Emoji -> sentiment polarity in [-1, 1].
EMOJI_SENTIMENT: dict[str, float] = {
    "😀": 0.8, "😁": 0.8, "😂": 0.7, "🤣": 0.7, "😃": 0.8, "😄": 0.9,
    "😅": 0.5, "😆": 0.7, "😊": 0.9, "😇": 0.7, "🙂": 0.5, "😍": 1.0,
    "🥰": 1.0, "😘": 0.9, "😗": 0.6, "😙": 0.6, "😚": 0.7, "😋": 0.7,
    "😎": 0.7, "🤗": 0.8, "👍": 0.8, "👌": 0.6, "✌": 0.5, "🙌": 0.8,
    "👏": 0.7, "💪": 0.6, "❤": 1.0, "🧡": 0.9, "💛": 0.9, "💚": 0.9,
    "💙": 0.9, "💜": 0.9, "🖤": 0.3, "💖": 1.0, "💕": 0.9, "😻": 0.9,
    "🔥": 0.6, "✨": 0.6, "🎉": 0.8, "🥳": 0.9, "😌": 0.5, "😢": -0.8,
    "😭": -0.8, "😞": -0.7, "😔": -0.7, "😟": -0.6, "😕": -0.5, "🙁": -0.6,
    "☹": -0.7, "😣": -0.6, "😖": -0.7, "😫": -0.7, "😩": -0.7, "😤": -0.5,
    "😠": -0.8, "😡": -1.0, "🤬": -1.0, "😨": -0.6, "😰": -0.7, "😱": -0.6,
    "😳": -0.3, "😬": -0.3, "🙄": -0.5, "😒": -0.6, "😏": -0.2, "💔": -1.0,
    "👎": -0.8, "😪": -0.4, "😴": -0.2, "🤢": -0.8, "🤮": -0.9, "😷": -0.3,
    "💀": -0.4, "👿": -0.8,
}

# Emoticon -> sentiment polarity, complementing the emoji map for ASCII faces.
EMOTICON_SENTIMENT: dict[str, float] = {
    ":)": 0.7, ":-)": 0.7, ":d": 0.9, ":-d": 0.9, "=)": 0.7, ":]": 0.6,
    ";)": 0.5, ";-)": 0.5, ":p": 0.4, ":-p": 0.4, "<3": 0.9, "^^": 0.6,
    ":(": -0.7, ":-(": -0.7, ":'(": -0.9, "=(": -0.7, ":[": -0.6,
    ":/": -0.4, ":-/": -0.4, ":|": -0.2, ">:(": -0.9, "d:": -0.6,
}

# Words that signal opinion/evaluation rather than fact (subjectivity cues).
SUBJECTIVITY_WORDS: frozenset[str] = frozenset(
    {
        "love", "hate", "like", "dislike", "adore", "enjoy", "want", "wish",
        "hope", "feel", "think", "believe", "guess", "suppose", "seem",
        "amazing", "awesome", "awful", "terrible", "horrible", "wonderful",
        "great", "good", "bad", "best", "worst", "beautiful", "ugly",
        "happy", "sad", "angry", "excited", "bored", "fantastic", "lovely",
        "disgusting", "annoying", "boring", "fun", "cool", "nice", "perfect",
        "favorite", "favourite", "gorgeous", "stunning", "pathetic", "lame",
        "brilliant", "ridiculous", "stupid", "smart", "crazy", "insane",
        "incredible", "delicious", "yummy", "gross", "sucks", "rocks",
    }
)

# Positive/negative seed words used by the sarcasm heuristic.
POSITIVE_SEEDS: frozenset[str] = frozenset(
    {
        "love", "great", "good", "awesome", "amazing", "wonderful", "perfect",
        "happy", "nice", "best", "fantastic", "excellent", "brilliant",
        "thanks", "thank", "yay", "lovely", "fun", "cool", "enjoy", "glad",
    }
)
NEGATIVE_SEEDS: frozenset[str] = frozenset(
    {
        "hate", "bad", "awful", "terrible", "horrible", "worst", "sad",
        "angry", "annoying", "boring", "sucks", "stupid", "ugly", "pathetic",
        "disappointed", "broken", "fail", "failed", "ruined", "waste",
    }
)
SARCASM_CUES: tuple[str, ...] = (
    "yeah right", "oh great", "oh wonderful", "just great", "how nice",
    "well done", "thanks a lot", "good job", "nice going", "as if",
    "wow just wow", "sure thing", "love that", "great just great",
)

# Universal POS tagset categories (stable, language-independent column schema).
POS_CATEGORIES: tuple[str, ...] = (
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP",
    "DET", "CONJ", "NUM", "PRT", "X", ".",
)

# Regexes (compiled once at import time).
_WORD_RE = re.compile(r"[A-Za-z']+")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_EMOJI_RANGE_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoji extensions
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators
    "\U00002B00-\U00002BFF"  # arrows / stars
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U00002190-\U000021FF"  # arrows
    "]",
    flags=re.UNICODE,
)
_EMOTICON_RE = re.compile(
    r"(?:[:;=8xX][\-^]?[\)\]\(\[dDpP/|]|<3|\^\^|[\)\(]:|D:)"
)


class FeatureExtractor:
    """Transform raw text into interpretable sentiment feature vectors.

    The extractor is stateless with respect to the data (all signals come from
    fixed lexicons/heuristics), so it produces identical columns for any split.
    It still implements ``fit`` to honour the scikit-learn transformer contract
    and to expose :attr:`feature_names_` after fitting.

    Example:
        >>> fx = FeatureExtractor()
        >>> X_train = fx.fit_transform(train_texts)
        >>> X_val = fx.transform(val_texts)        # identical column order
    """

    def __init__(self, use_emot: bool = True) -> None:
        """Args:
        use_emot: Use the ``emot`` library for emoji/emoticon detection when
            available. Falls back to internal regex scanning otherwise.
        """
        self.use_emot = use_emot and _EMOT is not None
        self.feature_names_: list[str] = self._build_feature_names()
        self.n_features_: int = len(self.feature_names_)
        self._fitted = False

    # ------------------------------------------------------------------ API
    def fit(
        self, X: Iterable[str], y: object | None = None
    ) -> "FeatureExtractor":
        """Record the feature schema. No data-dependent state is learned.

        Args:
            X: Iterable of raw text strings (unused beyond validation).
            y: Ignored; present for scikit-learn API compatibility.

        Returns:
            ``self``.
        """
        _ = list(X)  # materialise/validate the iterable without storing it
        self._fitted = True
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Convert texts into an ``[n_samples, n_features]`` float matrix.

        Args:
            X: Iterable of raw text strings. ``None``/NaN entries are treated
                as empty strings.

        Returns:
            A 2-D ``np.float64`` array with columns ordered as
            :attr:`feature_names_`.
        """
        texts = self._coerce_texts(X)
        if not texts:
            return np.empty((0, self.n_features_), dtype=np.float64)
        rows = [self._extract_one(text) for text in texts]
        return np.asarray(rows, dtype=np.float64)

    def fit_transform(
        self, X: Iterable[str], y: object | None = None
    ) -> np.ndarray:
        """Fit then transform in a single call."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self, input_features: object | None = None
    ) -> np.ndarray:
        """Return feature names as a numpy array (scikit-learn convention)."""
        return np.asarray(self.feature_names_, dtype=object)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _coerce_texts(X: Iterable[str]) -> list[str]:
        """Normalise an arbitrary text iterable to a list of clean strings."""
        if isinstance(X, pd.Series):
            return X.fillna("").astype(str).tolist()
        coerced: list[str] = []
        for item in X:
            if item is None or (isinstance(item, float) and np.isnan(item)):
                coerced.append("")
            else:
                coerced.append(str(item))
        return coerced

    def _build_feature_names(self) -> list[str]:
        names = [
            "negation_count",
            "negation_ratio",
            "intensifier_count",
            "intensifier_score",
            "emoji_sentiment",
            "emoji_count",
            "emoji_density",
            "sarcasm_score",
            "exclamation_count",
            "question_count",
            "ellipsis_count",
            "punctuation_density",
            "caps_word_count",
            "caps_ratio",
            "char_count",
            "token_count",
            "avg_word_length",
            "subjectivity_score",
        ]
        names += [f"pos_{cat.lower().strip('.') or 'punct'}_ratio" for cat in POS_CATEGORIES]
        return names

    # ----------------------------------------------------------- emoji/emoticon
    def _detect_emojis(self, text: str) -> list[str]:
        """Return the list of emoji characters present in ``text``."""
        if self.use_emot:
            try:
                result = _EMOT.emoji(text)
                values = result.get("value", []) if isinstance(result, dict) else []
                if values:
                    return list(values)
            except Exception:
                pass  # fall through to regex
        return _EMOJI_RANGE_RE.findall(text)

    def _detect_emoticons(self, text: str) -> list[str]:
        """Return the list of ASCII emoticons present in ``text``."""
        if self.use_emot:
            try:
                result = _EMOT.emoticons(text)
                values = result.get("value", []) if isinstance(result, dict) else []
                if values:
                    return list(values)
            except Exception:
                pass
        return _EMOTICON_RE.findall(text)

    # --------------------------------------------------------------- POS tags
    @staticmethod
    def _pos_distribution(tokens: Sequence[str]) -> dict[str, float]:
        """Proportion of tokens in each universal POS category."""
        dist = {cat: 0.0 for cat in POS_CATEGORIES}
        if not tokens or not _NLTK_AVAILABLE:
            return dist
        try:
            tagged = pos_tag(list(tokens), tagset="universal")
        except Exception:
            return dist
        total = len(tagged)
        if total == 0:
            return dist
        for _, tag in tagged:
            key = tag if tag in dist else "X"
            dist[key] += 1.0
        return {cat: count / total for cat, count in dist.items()}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Word-tokenise text, with a regex fallback if NLTK data is absent."""
        if _NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        return _WORD_RE.findall(text)

    # ------------------------------------------------------------ core extract
    def _extract_one(self, text: str) -> list[float]:
        """Compute the full feature vector for a single text."""
        text = text or ""
        lowered = text.lower()
        char_count = len(text)

        tokens = self._tokenize(text)
        word_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
        lower_words = [w.lower() for w in word_tokens]
        token_count = len(word_tokens)
        denom_tokens = max(token_count, 1)
        denom_chars = max(char_count, 1)

        # --- Negation -----------------------------------------------------
        neg_count = sum(1 for w in lower_words if w.strip("'") in NEGATION_WORDS)
        neg_ratio = neg_count / denom_tokens

        # --- Intensifiers -------------------------------------------------
        intensifier_hits = [INTENSIFIERS[w] for w in lower_words if w in INTENSIFIERS]
        intensifier_count = float(len(intensifier_hits))
        # Net amplification relative to neutral (1.0) per hit; 0 when none.
        intensifier_score = float(sum(w - 1.0 for w in intensifier_hits))

        # --- Emoji / emoticon sentiment ----------------------------------
        emojis = self._detect_emojis(text)
        emoticons = self._detect_emoticons(text)
        emoji_count = float(len(emojis) + len(emoticons))
        emoji_sentiment = float(
            sum(EMOJI_SENTIMENT.get(e, 0.0) for e in emojis)
            + sum(EMOTICON_SENTIMENT.get(e.lower(), 0.0) for e in emoticons)
        )
        emoji_density = emoji_count / denom_chars

        # --- Punctuation --------------------------------------------------
        exclamation_count = float(text.count("!"))
        question_count = float(text.count("?"))
        ellipsis_count = float(len(re.findall(r"\.{2,}|…", text)))
        punct_chars = sum(1 for c in text if c in "!?.,;:-\"'()[]{}")
        punctuation_density = punct_chars / denom_chars

        # --- Capitalisation ----------------------------------------------
        caps_words = _ALL_CAPS_RE.findall(text)
        caps_word_count = float(len(caps_words))
        caps_ratio = caps_word_count / denom_tokens

        # --- Lengths ------------------------------------------------------
        avg_word_length = (
            sum(len(w) for w in word_tokens) / denom_tokens if token_count else 0.0
        )

        # --- Subjectivity -------------------------------------------------
        subj_hits = sum(1 for w in lower_words if w in SUBJECTIVITY_WORDS)
        subjectivity_score = subj_hits / denom_tokens

        # --- Sarcasm heuristic -------------------------------------------
        sarcasm_score = self._sarcasm_score(
            lowered=lowered,
            lower_words=lower_words,
            emoji_sentiment=emoji_sentiment,
            exclamation_count=exclamation_count,
            ellipsis_count=ellipsis_count,
        )

        # --- POS distribution --------------------------------------------
        pos_dist = self._pos_distribution(word_tokens)

        vector = [
            float(neg_count),
            neg_ratio,
            intensifier_count,
            intensifier_score,
            emoji_sentiment,
            emoji_count,
            emoji_density,
            sarcasm_score,
            exclamation_count,
            question_count,
            ellipsis_count,
            punctuation_density,
            caps_word_count,
            caps_ratio,
            float(char_count),
            float(token_count),
            avg_word_length,
            subjectivity_score,
        ]
        vector += [pos_dist[cat] for cat in POS_CATEGORIES]
        return vector

    @staticmethod
    def _sarcasm_score(
        lowered: str,
        lower_words: Sequence[str],
        emoji_sentiment: float,
        exclamation_count: float,
        ellipsis_count: float,
    ) -> float:
        """Heuristic sarcasm signal in roughly [0, ~4].

        Combines several weak structural cues that correlate with sarcasm:
        explicit sarcasm phrases, positive-text + negative-emoji mismatch,
        co-occurrence of positive and negative seed words, and trailing
        ellipses after positive sentiment.
        """
        score = 0.0
        word_set = set(lower_words)

        # Explicit lexicalised sarcasm phrases.
        for cue in SARCASM_CUES:
            if cue in lowered:
                score += 1.0

        pos_hits = len(word_set & POSITIVE_SEEDS)
        neg_hits = len(word_set & NEGATIVE_SEEDS)

        # Polarity mismatch between words and emoji (e.g. "great" + 😡).
        if pos_hits and emoji_sentiment < -0.2:
            score += 1.0
        if neg_hits and emoji_sentiment > 0.2:
            score += 0.5

        # Both positive and negative seeds present -> contrast.
        if pos_hits and neg_hits:
            score += 0.5

        # Positive words with heavy emphasis or trailing ellipsis.
        if pos_hits and (exclamation_count >= 2 or ellipsis_count >= 1):
            score += 0.5

        return score


def ensure_nltk_resources() -> None:
    """Download the NLTK resources required for tokenisation and POS tagging.

    Safe to call repeatedly; downloads are skipped when already present. Call
    once before training in fresh environments.
    """
    if not _NLTK_AVAILABLE:
        print("[WARN] NLTK not installed; using regex tokeniser and zero POS features.")
        return
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/universal_tagset", "universal_tagset"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[INFO] Downloading NLTK resource: {package}")
            nltk.download(package, quiet=True)

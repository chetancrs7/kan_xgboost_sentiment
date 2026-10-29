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
* Lexicon polarity- VADER sentence scores (compound/pos/neg/neu) and
                    token-level lexicon hit ratios; the only feature group
                    that sees *which* words are used, giving the model
                    lexical sentiment signal the structural cues lack.
                    Lookups are elongation-aware ("goooood" resolves to
                    "good") and complemented by explicit negation-scope
                    counts ("not good" -> negated positive).
* Elongation      - stretched-word counts ("sooooo"), a Twitter-specific
                    emphasis cue invisible to lexicons and POS tags.
* Hashtag         - count and VADER polarity of camel-split hashtag tokens
                    ("#FollowFriday" -> "follow friday"); pos/neg tallies.
* Dependency      - spaCy dependency-parse statistics (negation/amod/advmod
                    relation counts, max & mean tree depth). Optional: the
                    columns are zero-filled when spaCy or its English model is
                    not installed, so the feature width never changes.

Most features are stateless lexicon/heuristic computations; ``fit`` only records
the (constant) feature schema so the transformer integrates cleanly into
scikit-learn pipelines and persists predictably. The dependency group is the
one exception, running a batched spaCy pass inside ``transform``.
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

try:  # VADER ships with NLTK; requires the ``vader_lexicon`` data package.
    from nltk.sentiment import SentimentIntensityAnalyzer

    _VADER_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without NLTK installed
    _VADER_AVAILABLE = False

try:
    import emot

    _EMOT = emot.core.emot()
except Exception:  # pragma: no cover - emot optional at runtime
    _EMOT = None


# Lexicons
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
# A character repeated 3+ times marks a deliberately stretched word.
_ELONGATED_RE = re.compile(r"(\w)\1{2,}")
_ANY_REPEAT_RE = re.compile(r"(\w)\1+")
# Scare-quoted spans (straight or curly quotes) — putting a word in quotes is a
# common irony/sarcasm cue: scare quotes around a word.
_QUOTED_SPAN_RE = re.compile(
    "\"[^\"]{1,40}\"|'[^']{1,40}'|‘[^’]{1,40}’|“[^”]{1,40}”"
)

# Contrast / discourse connectives. A sentiment flip across one of these
# (“great ... but ... terrible”) is a construction the semantic branch tends to
# average away — diagnostics show it is 1.75x more common where XGBoost is right
# and the KAN is wrong, so it is the highest-value complementary structural cue.
_CONTRAST_RE = re.compile(
    r'\b(?:but|however|although|though|yet|whereas|nonetheless|nevertheless|'
    r'still|instead|conversely|otherwise)\b'
)
# Hashtags and camel-case splitting for hashtag-sentiment scoring
# ("#FollowFriday" -> "follow friday", "#SUPER_good" -> "super good").
_HASHTAG_RE = re.compile(r"#(\w+)")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_]+")

#: spaCy dependency-relation labels tracked as engineered features.
_DEP_FEATURE_NAMES: tuple[str, ...] = (
    "dep_neg_count",     # explicit negation relations ("not" -> head)
    "dep_amod_count",    # adjectival modifiers of nouns
    "dep_advmod_count",  # adverbial modifiers (often intensifiers)
    "dep_max_depth",     # deepest token in the dependency tree
    "dep_mean_depth",    # mean token depth (syntactic complexity)
)
#: Hashtag-sentiment feature names (VADER over camel-split hashtag tokens).
_HASHTAG_FEATURE_NAMES: tuple[str, ...] = (
    "hashtag_count",
    "hashtag_sentiment",
    "hashtag_pos_count",
    "hashtag_neg_count",
)
#: Contrast / discourse structure feature names. These capture *sentiment shift
#: across a contrast connective* — a construction the semantic branch averages
#: away — to make the structural expert complementary on the KAN's errors.
_DISCOURSE_FEATURE_NAMES: tuple[str, ...] = (
    "contrast_count",            # number of contrast connectives
    "has_contrast",              # any contrast connective present
    "post_contrast_polarity",    # VADER compound of the clause AFTER the last one
    "contrast_polarity_delta",   # post-clause minus pre-clause compound (the shift)
    "contrast_polarity_flip",    # 1 if the clauses have opposite-sign polarity
    "quoted_polarity",           # VADER compound of scare-quoted spans (sarcasm)
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

    def __init__(
        self,
        use_emot: bool = True,
        use_vader: bool = True,
        use_hashtag: bool = True,
        use_spacy_dep: bool = True,
    ) -> None:
        """Args:
        use_emot: Use the ``emot`` library for emoji/emoticon detection when
            available. Falls back to internal regex scanning otherwise.
        use_vader: Compute VADER lexicon-polarity features when the
            ``vader_lexicon`` NLTK data is available. When unavailable the
            lexicon columns are zero-filled so the schema stays stable.
        use_hashtag: Compute hashtag-sentiment features (VADER over
            camel-split hashtag tokens). Requires ``use_vader``; the columns
            are always present in the schema and zero-filled when disabled or
            when VADER data is missing.
        use_spacy_dep: Compute dependency-parse features via spaCy's
            ``en_core_web_sm`` model. The columns are always present in the
            schema and zero-filled when spaCy (or its model) is unavailable,
            so the feature width is stable across environments.
        """
        self.use_emot = use_emot and _EMOT is not None
        self.use_vader = use_vader
        self.use_hashtag = use_hashtag
        self.use_spacy_dep = use_spacy_dep
        self._vader = self._build_vader() if use_vader else None
        # spaCy pipe is loaded lazily on first transform: None = not tried yet,
        # False = tried and unavailable (warned once), object = loaded model.
        self._spacy: object | None = None
        self.feature_names_: list[str] = self._build_feature_names()
        self.n_features_: int = len(self.feature_names_)
        self._fitted = False

    @staticmethod
    def _build_vader() -> "SentimentIntensityAnalyzer | None":
        """Instantiate VADER, or ``None`` when its lexicon data is missing."""
        if not _VADER_AVAILABLE:
            return None
        try:
            return SentimentIntensityAnalyzer()
        except Exception:
            return None

    # ------------------------------------------------------- (de)serialisation
    def __getstate__(self) -> dict:
        """Drop the VADER analyzer and spaCy pipe for pickling; rebuilt on load."""
        state = self.__dict__.copy()
        state["_vader"] = None
        state["_spacy"] = None  # unpicklable; reloaded lazily after unpickle
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Back-compat: instances pickled before these flags existed.
        self.use_hashtag = getattr(self, "use_hashtag", True)
        self.use_spacy_dep = getattr(self, "use_spacy_dep", True)
        self._spacy = None
        if self.use_vader and self._vader is None:
            self._vader = self._build_vader()

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
        # Row-wise lexical/hashtag features, then a single batched spaCy pass
        # for the dependency features (columns concatenated in schema order).
        base = np.asarray([self._extract_one(text) for text in texts], dtype=np.float64)
        dep = self._dependency_matrix(texts)
        return np.hstack([base, dep])

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
            "vader_compound",
            "vader_pos",
            "vader_neg",
            "vader_neu",
            "lexicon_pos_ratio",
            "lexicon_neg_ratio",
            "lexicon_net_polarity",
            "negated_positive_count",
            "negated_negative_count",
            "elongated_word_count",
            "elongated_word_ratio",
        ]
        names += [f"pos_{cat.lower().strip('.') or 'punct'}_ratio" for cat in POS_CATEGORIES]
        # Hashtag + discourse columns are produced row-wise by ``_extract_one``;
        # dependency columns are appended by the batched pass in ``transform``.
        # All groups are always in the schema (zero-filled when unavailable).
        names += list(_HASHTAG_FEATURE_NAMES)
        names += list(_DISCOURSE_FEATURE_NAMES)
        names += list(_DEP_FEATURE_NAMES)
        return names

    # ------------------------------------------------------------ lexicon utils
    def _lexicon_valence(self, word: str) -> float:
        """VADER valence of a lowercase word, resolving elongated spellings.

        Tries the word as written, then with character runs collapsed to two
        ("goood" -> "good"), then to one ("sooo" -> "soo" -> "so"), so
        stretched words still hit the lexicon.
        """
        lexicon = self._vader.lexicon
        if word in lexicon:
            return lexicon[word]
        if not _ELONGATED_RE.search(word):
            return 0.0
        collapsed_two = _ELONGATED_RE.sub(r"\1\1", word)
        if collapsed_two in lexicon:
            return lexicon[collapsed_two]
        collapsed_one = _ANY_REPEAT_RE.sub(r"\1", word)
        return lexicon.get(collapsed_one, 0.0)

    def _normalize_elongation(self, text: str) -> str:
        """Rewrite stretched words to their lexicon spelling for VADER.

        Only whitespace tokens containing a 3+ character run are touched, and
        case is preserved so VADER's ALL-CAPS emphasis handling still fires.
        Tokens that resolve to nothing in the lexicon keep the two-character
        collapse, which denoises without inventing words ("ahhhh" -> "ahh").
        """
        if not _ELONGATED_RE.search(text):
            return text
        lexicon = self._vader.lexicon
        parts = text.split(" ")
        for i, part in enumerate(parts):
            if not _ELONGATED_RE.search(part):
                continue
            collapsed_two = _ELONGATED_RE.sub(r"\1\1", part)
            if collapsed_two.lower() in lexicon:
                parts[i] = collapsed_two
                continue
            collapsed_one = _ANY_REPEAT_RE.sub(r"\1", part)
            parts[i] = collapsed_one if collapsed_one.lower() in lexicon else collapsed_two
        return " ".join(parts)

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

        # --- Lexicon polarity (VADER) --------------------------------------
        # Sentence-level scores capture negation/intensity handling; token
        # ratios separate "neutral" from "mixed positive+negative" texts,
        # which the compound score alone conflates.
        vader_compound = vader_pos = vader_neg = vader_neu = 0.0
        lexicon_pos_ratio = lexicon_neg_ratio = lexicon_net_polarity = 0.0
        negated_positive_count = negated_negative_count = 0.0
        if self._vader is not None:
            scores = self._vader.polarity_scores(self._normalize_elongation(text))
            vader_compound = scores["compound"]
            vader_pos = scores["pos"]
            vader_neg = scores["neg"]
            vader_neu = scores["neu"]
            valences = [self._lexicon_valence(w) for w in lower_words]
            lexicon_pos_ratio = sum(1 for v in valences if v > 0) / denom_tokens
            lexicon_neg_ratio = sum(1 for v in valences if v < 0) / denom_tokens
            # VADER valences live in [-4, 4]; rescale the mean to ~[-1, 1].
            lexicon_net_polarity = sum(valences) / (4.0 * denom_tokens)

            # Explicit negation-scope counts: a polar word within two tokens
            # of a negation cue ("not good", "never works"). VADER folds this
            # into its compound score; the raw counts give the trees a
            # direct, unsquashed view of the flip.
            for j, w in enumerate(lower_words):
                if w.strip("'") in NEGATION_WORDS:
                    for v in valences[j + 1 : j + 3]:
                        if v > 0:
                            negated_positive_count += 1.0
                        elif v < 0:
                            negated_negative_count += 1.0

        # --- Elongation ----------------------------------------------------
        elongated_word_count = float(
            sum(1 for w in word_tokens if _ELONGATED_RE.search(w))
        )
        elongated_word_ratio = elongated_word_count / denom_tokens

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
            vader_compound,
            vader_pos,
            vader_neg,
            vader_neu,
            lexicon_pos_ratio,
            lexicon_neg_ratio,
            lexicon_net_polarity,
            negated_positive_count,
            negated_negative_count,
            elongated_word_count,
            elongated_word_ratio,
        ]
        vector += [pos_dist[cat] for cat in POS_CATEGORIES]

        # --- Hashtag sentiment (VADER over camel-split hashtag tokens) -----
        vector += self._hashtag_features(text)

        # --- Contrast / discourse structure (complementary to the KAN) -----
        vector += self._discourse_features(text, lowered)
        return vector

    # ---------------------------------------------------- contrast / discourse
    def _discourse_features(self, text: str, lowered: str) -> list[float]:
        """Contrast-structure features: sentiment shift across a connective.

        Splits at the *last* contrast connective ("great ... but ... terrible")
        and compares the VADER polarity of the clause before and after it — the
        after-clause usually carries the sentiment a reader labels on, and a
        sign flip is the classic contrast the semantic branch smooths over. Also
        scores scare-quoted spans. Zero-filled without VADER so the six columns
        are always present.
        """
        matches = list(_CONTRAST_RE.finditer(lowered))
        count = float(len(matches))
        has = 1.0 if matches else 0.0
        post_pol = delta = flip = 0.0
        quoted_pol = 0.0
        if self._vader is not None:
            if matches:
                cut_start, cut_end = matches[-1].start(), matches[-1].end()
                pre, post = text[:cut_start], text[cut_end:]
                pre_c = self._vader.polarity_scores(pre)["compound"] if pre.strip() else 0.0
                post_c = self._vader.polarity_scores(post)["compound"] if post.strip() else 0.0
                post_pol = float(post_c)
                delta = float(post_c - pre_c)
                flip = 1.0 if pre_c * post_c < 0 else 0.0
            quoted = _QUOTED_SPAN_RE.findall(text)
            if quoted:
                quoted_pol = float(
                    np.mean([self._vader.polarity_scores(q)["compound"] for q in quoted])
                )
        return [count, has, post_pol, delta, flip, quoted_pol]

    # ------------------------------------------------------------ hashtags
    def _hashtag_features(self, text: str) -> list[float]:
        """Count and VADER-score hashtags, camel-split to their word tokens.

        Returns ``[count, mean_compound, pos_count, neg_count]``. Zero-filled
        when hashtags are disabled, absent, or VADER data is unavailable, so
        the four columns always exist.
        """
        if not self.use_hashtag:
            return [0.0, 0.0, 0.0, 0.0]
        tags = _HASHTAG_RE.findall(text)
        if not tags:
            return [0.0, 0.0, 0.0, 0.0]
        if self._vader is None:
            return [float(len(tags)), 0.0, 0.0, 0.0]
        scores = [
            self._vader.polarity_scores(
                _CAMEL_SPLIT_RE.sub(" ", tag).lower().strip()
            )["compound"]
            for tag in tags
        ]
        return [
            float(len(tags)),
            float(sum(scores) / len(scores)),
            float(sum(1 for s in scores if s > 0.05)),
            float(sum(1 for s in scores if s < -0.05)),
        ]

    # ---------------------------------------------------- dependency (spaCy)
    def _get_spacy(self):
        """Lazily load spaCy's ``en_core_web_sm``; cache ``False`` on failure."""
        if self._spacy is not None:
            return self._spacy or None  # None when previously found unavailable
        try:
            import spacy

            # Parser is required for dependency labels/depths; drop the rest.
            self._spacy = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception as exc:  # not installed, or model not downloaded
            print(
                "[WARN] spaCy dependency features unavailable "
                f"({type(exc).__name__}); dependency columns zero-filled. "
                "Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
            self._spacy = False
            return None
        return self._spacy

    def _dependency_matrix(self, texts: list[str]) -> np.ndarray:
        """Batched dependency features ``[n, 5]`` (zeros when spaCy is absent)."""
        n = len(texts)
        zeros = np.zeros((n, len(_DEP_FEATURE_NAMES)), dtype=np.float64)
        if not self.use_spacy_dep:
            return zeros
        nlp = self._get_spacy()
        if nlp is None:
            return zeros
        rows = [self._dep_row(doc) for doc in nlp.pipe(texts, batch_size=1000)]
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _dep_row(doc) -> list[float]:
        """Dependency-relation counts and tree-depth stats for one spaCy doc."""
        dep_neg = dep_amod = dep_advmod = 0
        depths: list[int] = []
        for tok in doc:
            if tok.dep_ == "neg":
                dep_neg += 1
            elif tok.dep_ == "amod":
                dep_amod += 1
            elif tok.dep_ == "advmod":
                dep_advmod += 1
            depth, cur = 0, tok
            while cur.head != cur and depth < 64:
                depth += 1
                cur = cur.head
            depths.append(depth)
        return [
            float(dep_neg),
            float(dep_amod),
            float(dep_advmod),
            float(max(depths) if depths else 0),
            float(sum(depths) / len(depths)) if depths else 0.0,
        ]

    @staticmethod
    def _sarcasm_score(
        lowered: str,
        lower_words: Sequence[str],
        emoji_sentiment: float,
        exclamation_count: float,
        ellipsis_count: float,
    ) -> float:
        """Heuristic sarcasm signal in roughly [0, ~5].

        Combines several weak structural cues that correlate with sarcasm:
        explicit sarcasm phrases, positive-text + negative-emoji mismatch,
        co-occurrence of positive and negative seed words, trailing ellipses
        after positive sentiment, and scare-quote usage.
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

        # Scare quotes: quoting a word/phrase signals irony ("she 'loved' it").
        # Weight higher when the quoted text sits next to positive sentiment.
        if _QUOTED_SPAN_RE.search(lowered):
            score += 1.0 if pos_hits else 0.5

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
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[INFO] Downloading NLTK resource: {package}")
            nltk.download(package, quiet=True)

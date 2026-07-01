"""Reusable, dependency-free helper functions shared across the Redrob
ranking system (data_loader, feature_extractor, scorer, reasoner, rank.py).

Performance notes:
    - find_matches() uses zero regex. Keyword matching is plain substring
      search over space-padded, pre-normalized text (both text and
      keywords are normalized once, then compared with the C-implemented
      `in` operator), which is both correct (still whole-word / whole-
      phrase bounded) and much faster than compiling/searching a regex per
      keyword per candidate.
    - normalize_text() does a single str.translate() pass (cached
      translation table, built once at import time) followed by a single
      split/join to collapse whitespace - no regex, no repeated lower()
      or strip() calls beyond what's structurally necessary.
    - Keyword normalization is memoized via a module-level dict cache
      (_normalize_keyword) so a fixed keyword list (the common case - the
      same TECH_KEYWORDS/CAREER_EVIDENCE_KEYWORDS/etc. lists are reused
      across every candidate) is only lowercased/translated once ever,
      not once per candidate.
    - No external libraries. No global *mutable-by-callers* state: the
      translation table, date-format tuple, and keyword-normalization
      cache are internal implementation details, not part of the public
      surface, and every public function remains pure from the caller's
      point of view.
"""

import string
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PUNCTUATION_TABLE = str.maketrans({c: " " for c in string.punctuation})

_DATE_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
)

# Memoizes keyword -> normalized-keyword. Keyword lists are small, fixed,
# module-level constants reused across every candidate in the dataset, so
# caching this avoids re-lowercasing/re-translating the same keyword
# thousands of times over a run.
_KEYWORD_NORM_CACHE: Dict[str, str] = {}

_TRUE_TOKENS = frozenset(("true", "yes", "1", "y", "t"))
_FALSE_TOKENS = frozenset(("false", "no", "0", "n", "f"))


def safe_get(obj: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """Safely read `key` from `obj`. Never raises, even if obj is None,
    not a dict, or the stored value is explicitly None."""
    if type(obj) is not dict:
        if not isinstance(obj, dict):
            return default
    value = obj.get(key, default)
    return default if value is None else value


def as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion of `value` to float, falling back to `default`
    on any failure (None, empty string, non-numeric text, etc.)."""
    if value is None:
        return default
    t = type(value)
    if t is float:
        return value
    if t is bool:
        return 1.0 if value else 0.0
    if t is int:
        return float(value)
    if t is str:
        stripped = value.strip()
        if not stripped:
            return default
        if "," in stripped:
            stripped = stripped.replace(",", "")
        try:
            return float(stripped)
        except ValueError:
            return default
    return default


def as_int(value: Any, default: int = 0) -> int:
    """Best-effort conversion of `value` to int, falling back to `default`
    on any failure."""
    if value is None:
        return default
    t = type(value)
    if t is int:
        return value
    if t is bool:
        return 1 if value else 0
    if t is float:
        return int(value)
    if t is str:
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(float(stripped))
        except ValueError:
            return default
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    """Best-effort conversion of `value` to bool.

    Supports: True/False, "true"/"false", "yes"/"no", 1/0, "1"/"0", None.
    """
    if value is None:
        return default
    t = type(value)
    if t is bool:
        return value
    if t is int or t is float:
        return value != 0
    if t is str:
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        return default
    return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
    *,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> float:
    """Clamp `value` into [minimum, maximum].

    `lo`/`hi` are accepted as keyword aliases for `minimum`/`maximum` for
    backward compatibility with existing call sites.
    """
    low = lo if lo is not None else minimum
    high = hi if hi is not None else maximum
    if low > high:
        low, high = high, low
    if value < low:
        return low
    if value > high:
        return high
    return value


def normalize_text(*parts: Optional[str]) -> str:
    """Join text fragments into a single lowercase string with punctuation
    stripped and whitespace collapsed. Safe for None entries.

    Single translate() pass (cached table) + one split/join keeps this
    O(n) with a low constant factor and no regex engine involvement.
    """
    joined = " ".join(str(p) for p in parts if p)
    if not joined:
        return ""
    return " ".join(joined.lower().translate(_PUNCTUATION_TABLE).split())


def _normalize_keyword(keyword: str) -> str:
    """Apply the same normalization used on scanned text to a single
    keyword, memoized since keyword lists are small, fixed, and reused
    across every candidate in a run."""
    cached = _KEYWORD_NORM_CACHE.get(keyword)
    if cached is not None:
        return cached
    normalized = " ".join(keyword.strip().lower().translate(_PUNCTUATION_TABLE).split())
    _KEYWORD_NORM_CACHE[keyword] = normalized
    return normalized


def find_matches(text: str, keywords: Iterable[str]) -> List[str]:
    """Return the unique subset of `keywords` found in `text`.

    Case-insensitive, whole-word / whole-phrase matching, implemented with
    plain substring search (no regex, no per-call compilation). `text` is
    assumed to already be normalize_text()-style (lowercase, punctuation
    stripped, single-spaced) - callers in this codebase always pass
    normalized text. Word/phrase boundaries are enforced by padding both
    sides with a single space and checking for " keyword " as a substring
    via the C-implemented `in` operator.
    """
    if not text:
        return []

    padded_text = " " + text + " "
    seen = set()
    matches: List[str] = []
    seen_add = seen.add
    matches_append = matches.append

    for kw in keywords:
        if kw is None:
            continue
        kw_norm = _normalize_keyword(kw) if type(kw) is str else _normalize_keyword(str(kw))
        if not kw_norm or kw_norm in seen:
            continue
        if (" " + kw_norm + " ") in padded_text:
            matches_append(kw)
            seen_add(kw_norm)

    return matches


def parse_date(value: Any) -> Optional[datetime]:
    """Best-effort parse of a date/timestamp value into a timezone-aware
    UTC datetime. Supports YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY, and common
    ISO8601 variants. Returns None if parsing fails or value is empty."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def days_since(dt: Optional[datetime], now: Optional[datetime] = None) -> Optional[int]:
    """Whole number of days between `dt` and `now` (defaults to current UTC
    time). Returns None if `dt` is None."""
    if dt is None:
        return None
    reference = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    delta = reference - dt
    return int(delta.total_seconds() // 86400)


def sort_key_for_ranking(candidate_id: str, score: float) -> Tuple[float, str]:
    """Sort key for building the final ranking: higher score first,
    candidate_id ascending as the tie-breaker. Usable directly as the
    `key=` argument (or a pre-computed value) inside sorted()."""
    return (-float(score), str(candidate_id))


def unique_preserve_order(sequence: Sequence[Any]) -> List[Any]:
    """Return the items of `sequence` with duplicates removed, preserving
    first-seen order. Items must be hashable."""
    seen = set()
    seen_add = seen.add
    return [x for x in sequence if not (x in seen or seen_add(x))]


def normalize_score(value: Any) -> float:
    """Convert `value` to float and clamp it into [0, 1]."""
    return clamp(as_float(value, default=0.0), minimum=0.0, maximum=1.0)


def weighted_average(score_weight_pairs: Iterable[Tuple[Optional[float], float]]) -> float:
    """Compute a normalized weighted average over (score, weight) pairs.

    Pairs whose score is None are ignored entirely (neither the score nor
    its weight contributes). Returns 0.0 if there is no usable weight.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for score, weight in score_weight_pairs:
        if score is None:
            continue
        w = as_float(weight, default=0.0)
        if w <= 0:
            continue
        weighted_sum += as_float(score, default=0.0) * w
        total_weight += w
    return safe_divide(weighted_sum, total_weight)


def percentage(value: Any) -> float:
    """Normalize a percentage-like value into a [0, 1] float.

    Accepts:
        95        -> 0.95
        0.95      -> 0.95
        "95"      -> 0.95
        "95%"     -> 0.95
        1         -> 1.0 (ambiguous; treated as already-normalized fraction)
    """
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        numeric = as_float(text, default=0.0)
    else:
        numeric = as_float(value, default=0.0)

    if numeric > 1.0:
        numeric = numeric / 100.0
    return clamp(numeric, minimum=0.0, maximum=1.0)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide a / b, returning `default` instead of raising when b is 0."""
    a_f = as_float(a, default=0.0)
    b_f = as_float(b, default=0.0)
    if b_f == 0:
        return default
    return a_f / b_f
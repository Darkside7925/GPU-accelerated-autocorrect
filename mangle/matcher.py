"""Layer 2: keyboard-aware and phonetic matcher (deterministic, no LLM).

For first-time mangles not yet in Layer 1. It does NOT use plain edit distance
(which caps out and fails on heavy mash). Instead it:

  1. Generates candidates cheaply from three angles, union'd:
       - phonetic buckets: words sharing the mangle's metaphone / nysiis code
       - SymSpell lookup at edit distance up to 3
       - a length + first/last letter bucket (mash-typing tends to preserve
         word length and the anchor letters)
  2. Scores each candidate with a keyboard-weighted Damerau-Levenshtein
     (substitutions between physically close QWERTY keys are cheap;
     transpositions are cheap because fast typists swap adjacent keys),
     combined with word frequency, length similarity, and a phonetic-match
     bonus.
  3. Returns the best candidate and a confidence in [0, 1]. The engine only
     auto-applies above a threshold; everything else defers to Layer 3.

Candidate scoring runs on a handful of words, so the pure-Python weighted
Damerau is plenty fast and avoids the fragile from-source build of the
weighted-levenshtein wheel. rapidfuzz gives a fast uniform-Damerau prefilter.
"""

from __future__ import annotations

import importlib.resources
import math

import jellyfish
from rapidfuzz.distance import DamerauLevenshtein
from symspellpy import SymSpell, Verbosity

# ------------------------------------------------------- QWERTY geometry

_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
# staggered offsets so key columns line up roughly like a real keyboard
_ROW_OFFSET = [0.0, 0.5, 0.75, 1.25]

_KEYPOS: dict[str, tuple[float, float]] = {}
for _r, _row in enumerate(_ROWS):
    for _c, _ch in enumerate(_row):
        _KEYPOS[_ch] = (_r * 1.0, _c + _ROW_OFFSET[_r])


def _key_dist(a: str, b: str) -> float:
    """Physical distance between two keys, ~0 adjacent to ~1 far. Unknown
    characters fall back to a full-cost substitution."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0.0
    pa, pb = _KEYPOS.get(a), _KEYPOS.get(b)
    if pa is None or pb is None:
        return 1.0
    d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    return min(1.0, d / 4.0)


def _sub_cost(a: str, b: str) -> float:
    """Substitution cost: adjacent-key slips are cheap, distant swaps near full."""
    if a == b:
        return 0.0
    return 0.4 + 0.6 * _key_dist(a, b)


def keyboard_damerau(s1: str, s2: str) -> float:
    """Damerau-Levenshtein with QWERTY-weighted substitutions and a discounted
    transposition cost (fast typists transpose adjacent keys constantly)."""
    s1, s2 = s1.lower(), s2.lower()
    n, m = len(s1), len(s2)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)
    INS = DEL = 1.0
    TRANSPOSE = 0.6
    prev2 = None
    prev = [j * INS for j in range(m + 1)]   # row for empty s1 prefix (j inserts)
    for i in range(1, n + 1):
        cur = [i * DEL] + [0.0] * m
        for j in range(1, m + 1):
            cost_sub = prev[j - 1] + _sub_cost(s1[i - 1], s2[j - 1])
            cost_del = prev[j] + DEL
            cost_ins = cur[j - 1] + INS
            best = min(cost_sub, cost_del, cost_ins)
            if (i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]):
                best = min(best, prev2[j - 2] + TRANSPOSE)
            cur[j] = best
        prev2, prev = prev, cur
    return prev[m]


class Layer2Matcher:
    def __init__(self, personal=None, max_candidates: int = 60):
        """personal: optional object with contains(word) -> bool, used to avoid
        recommending against the never-touch whitelist."""
        self._personal = personal
        self._max_candidates = max_candidates
        self._sym = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        dict_path = (importlib.resources.files("symspellpy")
                     / "frequency_dictionary_en_82_765.txt")
        self._sym.load_dictionary(str(dict_path), term_index=0, count_index=1)

        # freq + phonetic indexes over the same 80k dictionary
        self._freq: dict[str, int] = {}
        self._by_metaphone: dict[str, list[str]] = {}
        self._by_nysiis: dict[str, list[str]] = {}
        self._by_anchor: dict[tuple, list[str]] = {}
        with open(str(dict_path), encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                word, freq = parts[0], int(parts[1])
                self._freq[word] = freq
                self._bucket(self._by_metaphone, _safe(jellyfish.metaphone, word), word)
                self._bucket(self._by_nysiis, _safe(jellyfish.nysiis, word), word)
                if len(word) >= 3:
                    self._by_anchor.setdefault((word[0], word[-1], len(word)), []).append(word)
        self._max_freq = max(self._freq.values()) if self._freq else 1

    @staticmethod
    def _bucket(index: dict, code: str, word: str) -> None:
        if code:
            index.setdefault(code, []).append(word)

    def is_dictionary_word(self, word: str) -> bool:
        """O(1): is this a known English word? Used to decide passthrough."""
        return word.lower() in self._freq

    # ------------------------------------------------------ candidate set

    def _candidates(self, mangle: str) -> set[str]:
        cands: set[str] = set()
        code_m = _safe(jellyfish.metaphone, mangle)
        code_n = _safe(jellyfish.nysiis, mangle)
        if code_m:
            cands.update(self._by_metaphone.get(code_m, ()))
        if code_n:
            cands.update(self._by_nysiis.get(code_n, ()))
        if len(mangle) >= 3:
            for dl in (0, -1, 1):  # length can drift by a char under mash
                cands.update(self._by_anchor.get((mangle[0], mangle[-1], len(mangle) + dl), ()))
        for sug in self._sym.lookup(mangle, Verbosity.CLOSEST, max_edit_distance=3,
                                    include_unknown=False):
            cands.add(sug.term)
        return cands

    # ------------------------------------------------------------ scoring

    def _scored(self, low: str):
        """Return (word, score) for the plausible candidates, best first."""
        cands = self._candidates(low)
        if not cands:
            return []
        # cheap uniform-Damerau prefilter to the closest N before the weighted pass
        cands = sorted(cands, key=lambda w: DamerauLevenshtein.distance(low, w.lower()))
        cands = cands[: self._max_candidates]
        code_m = _safe(jellyfish.metaphone, low)
        scored = []
        for w in cands:
            wl = w.lower()
            if self._personal is not None and self._personal.contains(wl):
                continue
            dist = keyboard_damerau(low, wl)
            norm = max(len(low), len(wl)) or 1
            dist_score = 1.0 - min(1.0, dist / norm)                  # 1 = identical shape
            len_pen = abs(len(low) - len(wl)) / norm                  # 0 = same length
            freq_score = math.log1p(self._freq.get(w, 1)) / math.log1p(self._max_freq)
            phon_bonus = 0.15 if code_m and _safe(jellyfish.metaphone, wl) == code_m else 0.0
            scored.append((w, 0.60 * dist_score + 0.20 * freq_score
                           - 0.15 * len_pen + phon_bonus))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def match(self, mangle: str, min_confidence: float = 0.0):
        """Return (best_word, confidence) or (None, 0.0). Confidence in [0, 1]."""
        mangle = mangle.strip()
        if len(mangle) < 3:
            return None, 0.0
        scored = self._scored(mangle.lower())
        if not scored:
            return None, 0.0
        best_word, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else -1.0
        # Ambiguity gate: when the top two candidates are near-ties (e.g. teh ->
        # "the" vs "tee"), collapse confidence so the token defers to Layer 3,
        # where sentence context disambiguates. A clear winner keeps its score.
        margin = best_score - second_score if second_score >= 0 else best_score
        amb = min(1.0, max(0.0, margin) / 0.12)
        confidence = max(0.0, min(1.0, best_score * (0.55 + 0.45 * amb)))
        if confidence < min_confidence:
            return None, 0.0
        return _match_case(mangle, best_word), confidence

    def top_candidates(self, mangle: str, n: int = 5) -> list[str]:
        """The most keyboard-and-phonetically plausible dictionary words for a
        mangle, best first. Used as context hints for Layer 3 so it picks the
        word that fits the sentence rather than guessing blind."""
        if len(mangle.strip()) < 3:
            return []
        return [w for w, _ in self._scored(mangle.strip().lower())[:n]]


def _safe(fn, s: str) -> str:
    try:
        return fn(s) or ""
    except Exception:
        return ""


def _match_case(src: str, target: str) -> str:
    if src.isupper() and len(src) > 1:
        return target.upper()
    if src[:1].isupper():
        return target[:1].upper() + target[1:]
    return target

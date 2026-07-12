"""Curated homophone confusion groups for the guarded context pass.

A valid word from one of these groups may be sent to Layer 3 with its sentence,
and the model's answer is accepted ONLY if it is another member of the same
group. The worst possible outcome is a same-group swap (to <-> too), never a
rewrite, so passthrough safety is preserved by construction.
"""

from __future__ import annotations

GROUPS: list[frozenset[str]] = [
    frozenset({"to", "too", "two"}),
    frozenset({"their", "there", "they're"}),
    frozenset({"your", "you're"}),
    frozenset({"its", "it's"}),
    frozenset({"then", "than"}),
    frozenset({"were", "we're", "where"}),
    frozenset({"affect", "effect"}),
    frozenset({"loose", "lose"}),
    frozenset({"weather", "whether"}),
    frozenset({"whose", "who's"}),
    frozenset({"hear", "here"}),
    frozenset({"accept", "except"}),
    frozenset({"advice", "advise"}),
    frozenset({"brake", "break"}),
    frozenset({"principal", "principle"}),
]

_INDEX: dict[str, frozenset[str]] = {}
for _g in GROUPS:
    for _w in _g:
        _INDEX[_w] = _g


def group_of(word: str):
    """The confusion group containing `word` (lowercased), or None."""
    return _INDEX.get(word.lower())

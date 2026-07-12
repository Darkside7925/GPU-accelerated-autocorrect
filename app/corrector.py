"""Stage 1: SymSpell CPU spell correction (the sub-millisecond hot path)."""

from __future__ import annotations

import importlib.resources
import re

from symspellpy import SymSpell, Verbosity

# never touch tokens with digits, URLs, paths, identifiers
_SKIP_RE = re.compile(r"[\d@/\\_:]|^https?", re.IGNORECASE)


class SpellCorrector:
    def __init__(self, personal_dict):
        self._personal = personal_dict
        self._sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        self._sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
        # personal words become first-class dictionary entries so SymSpell
        # both leaves them alone and can suggest them
        for w in personal_dict.all_words():
            self._sym.create_dictionary_entry(w, 10**9)

    def learn(self, word: str) -> None:
        """Add a personal word at runtime (undo-to-learn)."""
        self._sym.create_dictionary_entry(word.lower(), 10**9)

    def correct(self, word: str, min_len: int = 3) -> str | None:
        """Return corrected word, or None if no correction should be applied."""
        if len(word) < min_len or _SKIP_RE.search(word):
            return None
        core = word.strip("'\"")
        if not core or self._personal.contains(core):
            return None
        # ALL-CAPS words are usually intentional (acronyms, emphasis)
        if core.isupper():
            return None
        suggestions = self._sym.lookup(
            core.lower(), Verbosity.TOP, max_edit_distance=2,
            include_unknown=False, transfer_casing=True,
        )
        if not suggestions:
            return None
        best = suggestions[0]
        if best.distance == 0 or best.term.lower() == core.lower():
            return None
        corrected = best.term
        # preserve leading capital (transfer_casing usually handles it; belt & braces)
        if core[0].isupper() and corrected:
            corrected = corrected[0].upper() + corrected[1:]
        return corrected

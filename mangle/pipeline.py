"""The layered routing decision for a single finished word.

Cheapest layer first, passthrough always preferred:

  pass   -> the word is valid (dictionary or whitelist) or not eligible; do
            nothing. This is the top priority: never touch correct text.
  apply  -> Layer 1 (personal memory) knows this mangle, or Layer 2 (matcher)
            is confident. Correct it now, deterministically, no model.
  defer  -> the word is not valid and neither deterministic layer resolved it
            confidently. Hand it to Layer 3 (context LLM) at sentence end.

This module holds no keyboard or injection state. The engine calls on_word()
and acts on the verdict, so the routing logic is unit-testable on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# tokens we never try to correct: too short, has digits, looks like code/paths,
# or is an ALL-CAPS acronym
_SKIP_RE = re.compile(r"[\d@/\\_:]|^https?", re.IGNORECASE)


@dataclass
class WordResult:
    action: str            # "pass" | "apply" | "defer" | "context"
    intended: str = ""     # replacement word (action == "apply")
    layer: str = ""        # "memory" | "matcher" (action == "apply")
    confidence: float = 0.0
    original: str = ""     # the token as typed
    candidates: tuple = () # Layer 2's plausible words, hints for Layer 3 (defer)


class RecoveryPipeline:
    def __init__(self, cfg, memory, matcher, personal):
        self.cfg = cfg
        self.memory = memory        # Layer 1: TypoMemory
        self.matcher = matcher      # Layer 2: Layer2Matcher
        self.personal = personal    # whitelist + never-touch (PersonalDict)

    def _l2_gate(self) -> float:
        return self.cfg.get("layer2_apply_confidence", 0.75)

    def _l1_gate(self) -> float:
        return self.cfg.get("layer1_apply_confidence", 0.50)

    def is_valid(self, core: str) -> bool:
        """A real word or an intentional personal term: leave it alone."""
        if not core:
            return True
        if self.personal.contains(core.lower()):
            return True
        return self.matcher.is_dictionary_word(core)

    def on_word(self, word: str) -> WordResult:
        core = word.strip("\"")
        if _SKIP_RE.search(core) or (core.isupper() and len(core) > 1):
            return WordResult("pass", original=word)
        # a homophone-group member may be the wrong word for the sentence
        # (to/too, their/there, it's/its); flag it for the guarded context
        # check instead of blind passthrough. Checked before the dictionary so
        # contractions route here even if the frequency dict lacks them.
        if self.cfg.get("context_homophones", True):
            from mangle.homophones import group_of
            if group_of(core) is not None and not self.personal.contains(core.lower()):
                return WordResult("context", original=word)
        core = core.strip("'-")
        if not core or self.is_valid(core):
            return WordResult("pass", original=word)
        min_len = self.cfg.get("min_word_len", 3)
        if len(core) < min_len:
            return WordResult("pass", original=word)

        # Layer 1: personal typo memory (instant, deterministic)
        hit = self.memory.lookup(core, min_confidence=self._l1_gate())
        if hit and hit.lower() != core.lower():
            return WordResult("apply", intended=hit, layer="memory",
                              confidence=self.memory.confidence(core), original=word)

        # Layer 2: keyboard + phonetic matcher (deterministic)
        cand, conf = self.matcher.match(core, min_confidence=0.0)
        if cand and conf >= self._l2_gate() and cand.lower() != core.lower():
            return WordResult("apply", intended=cand, layer="matcher",
                              confidence=conf, original=word)

        # not valid, not resolved -> Layer 3 (context). Pass Layer 2's plausible
        # candidates as hints so the model picks the one that fits the sentence.
        cands = tuple(self.matcher.top_candidates(core, n=5))
        return WordResult("defer", original=word, candidates=cands)

"""Layer 4 of the design: the end-of-day learning loop.

Runs in a batch, not on the hot path. It reads the day's accepted corrections
and raw text and does two things:

  1. Promote recurring mangled -> intended pairs into Layer 1 (typo memory), so
     tomorrow they are corrected instantly with no model. Accepted corrections
     climb in confidence; rejected ones (undone) are removed and stay gone.
  2. Grow the never-touch whitelist from non-dictionary tokens the user types
     often and never corrects (names, slang, technical terms), so they stop
     being flagged.

Over time Layer 1 and the whitelist grow, and Layer 3 (the LLM) fires less. The
system gets faster and more personal the more it is used.

maybe_run_daily() is called on app startup and only does work if at least a day
has passed since the last run, so it is cheap to call every launch.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter

log = logging.getLogger("compact")

DAY_SECONDS = 86400
DEFAULT_WHITELIST_MIN_COUNT = 3   # times a non-word must recur before whitelisting
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def compact(memory, personal, is_word, since_ts: float = 0.0,
            whitelist_min_count: int = DEFAULT_WHITELIST_MIN_COUNT) -> dict:
    """Run one compaction pass over data at or after since_ts. `is_word(token)`
    returns True for dictionary words (typically matcher.is_dictionary_word)."""
    corrections = personal.iter_corrections(since_ts)

    # 1. promote accepted corrections; scrub rejected ones
    promoted = 0
    typo_originals = set()
    for _ts, original, corrected, _stage, undone in corrections:
        if undone:
            memory.demote(original, corrected)
            continue
        if original and corrected and original.lower() != corrected.lower():
            memory.record(original, corrected, source="compaction")
            typo_originals.add(original.lower())
            promoted += 1

    # 2. grow the whitelist from frequent, never-corrected non-words
    counts: Counter = Counter()
    for _ts, text in memory.iter_raw(since_ts):
        for tok in _TOKEN_RE.findall(text):
            t = tok.strip("'")
            low = t.lower()
            if (len(t) >= 3 and not is_word(t) and not personal.contains(low)
                    and low not in typo_originals):
                counts[low] += 1
    whitelisted = 0
    for word, n in counts.items():
        if n >= whitelist_min_count:
            personal.add(word, source="compaction")
            whitelisted += 1

    memory.set_meta("last_compaction", time.time())
    result = {"promoted": promoted, "whitelisted": whitelisted,
              "corrections_seen": len(corrections)}
    log.info("compaction: %s", result)
    return result


def maybe_run_daily(cfg, memory, personal, matcher=None,
                    interval_s: int = DAY_SECONDS) -> dict | None:
    """Run compaction if at least interval_s has passed since the last run."""
    last = float(memory.get_meta("last_compaction", 0) or 0)
    now = time.time()
    if last and now - last < interval_s:
        return None
    is_word = matcher.is_dictionary_word if matcher is not None else (lambda w: False)
    since = last if last else 0.0
    return compact(memory, personal, is_word, since_ts=since)


if __name__ == "__main__":
    # manual run: python -m mangle.compact
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from app import config as config_mod
    from app.personal_dict import PersonalDict
    from mangle.matcher import Layer2Matcher
    from mangle.typo_memory import TypoMemory

    personal = PersonalDict(config_mod.DB_PATH)
    memory = TypoMemory(config_mod.DB_PATH)
    matcher = Layer2Matcher(personal=personal)
    res = compact(memory, personal, matcher.is_dictionary_word, since_ts=0.0)
    print(f"promoted {res['promoted']} pairs, whitelisted {res['whitelisted']} words "
          f"from {res['corrections_seen']} logged corrections")

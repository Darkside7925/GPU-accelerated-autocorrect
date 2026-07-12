"""Derived typing analytics for the dashboard ("what I learned about how you
type"). Pure functions over rows pulled from the SQLite tables, so they are
easy to test and hold no state.
"""

from __future__ import annotations

import difflib
import time
from collections import Counter, defaultdict

# physical rows, for the keyboard heatmap layout
KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def key_heat(pairs) -> dict:
    """Which keys the user fumbles most. For each (mangled, intended, count),
    diff the two strings and blame the intended-side letters of every edit.
    Returns {letter: weighted_count} over a-z."""
    heat: Counter = Counter()
    for mangled, intended, count, *_ in pairs:
        sm = difflib.SequenceMatcher(a=mangled.lower(), b=intended.lower(), autojunk=False)
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op == "equal":
                continue
            for ch in intended.lower()[b0:b1]:
                if ch.isalpha():
                    heat[ch] += count
            for ch in mangled.lower()[a0:a1]:   # also blame the wrong key hit
                if ch.isalpha():
                    heat[ch] += count
    return dict(heat)


def top_mangles(pairs, n: int = 15):
    """Most frequently corrected mangles."""
    rows = sorted(pairs, key=lambda r: r[2], reverse=True)[:n]
    return [{"mangled": m, "intended": i, "count": c, "confidence": round(conf, 2)}
            for (m, i, c, conf, *_rest) in rows]


def confidence_hist(pairs) -> dict:
    """Distribution of Layer 1 confidence in 0.1 buckets, so you can see how
    many learned pairs are strong enough to auto-apply (>= 0.5)."""
    buckets = [0] * 10
    for _m, _i, _c, conf, *_ in pairs:
        idx = min(9, max(0, int(conf * 10)))
        buckets[idx] += 1
    labels = [f"{i / 10:.1f}" for i in range(10)]
    return {"labels": labels, "counts": buckets}


def length_profile(pairs) -> dict:
    """Distribution of mangled-token lengths, weighted by count. Shows whether
    the user's typos cluster at short words or long ones."""
    dist: Counter = Counter()
    for mangled, _intended, count, *_ in pairs:
        dist[len(mangled)] += count
    return {str(k): dist[k] for k in sorted(dist)}


def corrections_over_time(corrections, days: int = 30) -> dict:
    """Per-day counts split by layer, plus the running share handled without the
    LLM. corrections rows are (ts, original, corrected, stage, undone)."""
    by_day_layer: dict = defaultdict(lambda: Counter())
    for ts, _o, _c, stage, undone in corrections:
        if undone:
            continue
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        by_day_layer[day][stage] += 1
    out_days, mem, matcher, llm, no_llm_share = [], [], [], [], []
    for day in sorted(by_day_layer)[-days:]:
        c = by_day_layer[day]
        m, mt, l = c.get("memory", 0), c.get("matcher", 0), c.get("llm", 0)
        total = m + mt + l
        out_days.append(day)
        mem.append(m)
        matcher.append(mt)
        llm.append(l)
        no_llm_share.append(round(100 * (m + mt) / total, 1) if total else 0.0)
    return {"days": out_days, "memory": mem, "matcher": matcher, "llm": llm,
            "no_llm_share": no_llm_share}


def hour_profile(corrections) -> dict:
    """Typo counts by hour of day (0-23), to reveal when the user is sloppiest."""
    hours = [0] * 24
    for ts, _o, _c, _stage, undone in corrections:
        if not undone:
            hours[int(time.strftime("%H", time.localtime(ts)))] += 1
    return {"hours": list(range(24)), "counts": hours}


def layer_totals(correction_counts) -> dict:
    """Totals per layer and the accepted/rejected split. correction_counts rows
    are (stage, undone, count)."""
    accepted: Counter = Counter()
    rejected = 0
    for stage, undone, cnt in correction_counts:
        if undone:
            rejected += cnt
        else:
            accepted[stage] += cnt
    total = sum(accepted.values())
    without_llm = accepted.get("memory", 0) + accepted.get("matcher", 0)
    return {
        "memory": accepted.get("memory", 0),
        "matcher": accepted.get("matcher", 0),
        "llm": accepted.get("llm", 0),
        "rejected": rejected,
        "total": total,
        "without_llm_pct": round(100 * without_llm / total, 1) if total else 0.0,
    }

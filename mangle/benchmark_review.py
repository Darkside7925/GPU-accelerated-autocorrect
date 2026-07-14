"""Gate benchmark for the adaptive review pass.

Measures, per model, using the production REVIEW_PROMPT and the production
align_review guards (so it tests exactly what ships):
  - fix rate     : sentences with a wrong word (hindsight homophone, missed
                   typo, doubled word) corrected to the intended sentence
  - passthrough  : clean sentences (slang, names, brands, tone traps) returned
                   with ZERO applied changes
Enable adaptive_review by default only if fix >= 70% and passthrough >= 95%.

    python -m mangle.benchmark_review gemma4:e4b
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app.config import options_for, load as load_config
from mangle.context_llm import REVIEW_PROMPT, align_review, clean_sentence

GATE_FIX = 0.70
GATE_PASS = 0.95

# (typed sentence, expected sentence after the review; equal = passthrough trap)
CASES = [
    # hindsight homophones: the wrong word only becomes visible in full context
    ("the water is way to hot for me", "the water is way too hot for me"),
    ("this movie is to long to finish tonight", "this movie is too long to finish tonight"),
    ("i left my keys over their on the table", "i left my keys over there on the table"),
    ("i think your going to like this", "i think you're going to like this"),
    ("she did better then me on the test", "she did better than me on the test"),
    ("we should of gone home earlier", "we should have gone home earlier"),
    ("its been a long day for everyone", "it's been a long day for everyone"),
    ("they went too the store yesterday", "they went to the store yesterday"),
    # a mangled word inside an otherwise clean sentence (missed-typo pickup)
    ("i realy think we should leave now", "i really think we should leave now"),
    ("the goverment made a new rule today", "the government made a new rule today"),
    ("can you beleive what happened last night", "can you believe what happened last night"),
    # doubled word (the one allowed deletion)
    ("the the water is cold today", "the water is cold today"),
    ("i want to to go home now", "i want to go home now"),
    # formatting must survive a fix (stars stay wrapped around the fixed word;
    # the apostrophe fix matches L1's seeded didnt->didn't policy)
    ("*whats hapening!* i just got here", "*what's happening!* i just got here"),
    ("*whats happening!* i just got here", "*what's happening!* i just got here"),
    # ---------------- passthrough traps: must come back with NO changes ----------------
    ("that rizz is actually crazy bro", None),
    ("tell Jhon i said hi when you see him", None),
    ("my starlink dish arrived this morning", None),
    ("the SymSpell index builds in milliseconds", None),
    ("he did good on the test", None),          # tone trap: NOT "well"
    ("this food is hella expensive here", None),  # slang stays
    ("i wanna grab some food real quick", None),
    ("the vibes at the party were immaculate", None),
    ("gonna head out in a bit", None),
    ("she lowkey carried the whole team", None),
    ("we grinded that project all night", None),  # verb-form trap: NOT "ground"
    ("no cap that was the best game ever", None),  # slang stays
    ("btw i finished the report already", None),   # shorthand stays
    ("idk what happened tbh it was weird", None),
    ("ngl that was smooth fr", None),
    ("i love waguri sm bro", None),                # lowercase name stays
    ("gojo would win ngl", None),
    ("this is _actually_ so good rn", None)        # emphasis stays
]


def review(cfg, model, sentence):
    opts = options_for(cfg, model)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": sentence},
        ],
        "stream": False, "think": False,
        "options": {"temperature": 0,
                    "num_predict": max(48, len(sentence.split()) * 3 + 16)},
        "keep_alive": "2h",
    }
    url = f"{cfg['ollama_url']}/api/chat"
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code == 400:
        payload.pop("think", None)
        r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return clean_sentence(r.json().get("message", {}).get("content", ""))


def apply_changes(sentence, changes, dedupe_at):
    """Apply align_review output the same way the engine does (positionally)."""
    import re
    spans = list(re.finditer(r"\S+", sentence))
    edits = []
    for idx, old_core, new_core in changes:
        tok = spans[idx].group(0)
        edits.append((spans[idx].start(), spans[idx].end(),
                      tok.replace(old_core, new_core, 1)))
    if dedupe_at >= 0:
        s = spans[dedupe_at]
        end = spans[dedupe_at + 1].start() if dedupe_at + 1 < len(spans) else s.end()
        edits.append((s.start(), end, ""))
    out = sentence
    for start, end, rep in sorted(edits, reverse=True):
        out = out[:start] + rep + out[end:]
    return out


def run(model, cfg, is_word):
    fix_total = fix_ok = pass_total = pass_ok = 0
    latencies = []
    for typed, expected in CASES:
        t0 = time.perf_counter()
        try:
            raw = review(cfg, model, typed)
        except requests.RequestException as e:
            print(f"  ERROR {typed!r}: {e}")
            raw = ""
        latencies.append(time.perf_counter() - t0)
        res = align_review(typed, raw, is_word=is_word) if raw else None
        applied = typed
        if res is not None:
            changes, dedupe_at = res
            if changes or dedupe_at >= 0:
                applied = apply_changes(typed, changes, dedupe_at)
        if expected is None:
            pass_total += 1
            ok = applied == typed
            pass_ok += ok
            print(f"  {'OK ' if ok else 'FAIL'} keep : {typed!r}"
                  + ("" if ok else f" -> {applied!r}  (raw {raw!r})"))
        else:
            fix_total += 1
            ok = applied.lower() == expected.lower()
            fix_ok += ok
            print(f"  {'OK ' if ok else 'FAIL'} fix  : {typed!r} -> {applied!r}"
                  + ("" if ok else f"  (want {expected!r}, raw {raw!r})"))
    lat = sorted(latencies)[len(latencies) // 2]
    fix = fix_ok / fix_total if fix_total else 0
    keep = pass_ok / pass_total if pass_total else 0
    print(f"\n{model}: fix {fix_ok}/{fix_total} ({fix:.0%})  "
          f"passthrough {pass_ok}/{pass_total} ({keep:.0%})  p50 {lat*1000:.0f}ms")
    gate = fix >= GATE_FIX and keep >= GATE_PASS
    print(f"GATE (fix >= {GATE_FIX:.0%}, passthrough >= {GATE_PASS:.0%}): "
          f"{'PASS' if gate else 'FAIL'}")
    return gate


def main():
    cfg = load_config()
    models = sys.argv[1:] or [cfg["active_model"]]
    print("building matcher for the is_word guard...")
    from mangle.matcher import Layer2Matcher
    from app.personal_dict import PersonalDict
    import tempfile, os
    matcher = Layer2Matcher(personal=PersonalDict(
        os.path.join(tempfile.mkdtemp(), "bench.db")))
    ok = True
    for model in models:
        print(f"\n=== {model} ===")
        ok = run(model, cfg, matcher.is_word) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

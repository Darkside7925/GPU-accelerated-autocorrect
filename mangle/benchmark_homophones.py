"""Gate benchmark for the guarded homophone context pass.

Measures, per model, using the production prompt + within-group guard:
  - flip accuracy : wrong homophones corrected to the intended member
  - keep rate     : already-correct homophones left alone (the passthrough gate)
Enable the feature only if flip >= 80% and keep >= 95%.

    python -m mangle.benchmark_homophones gemma4:e2b gemma4:e4b
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app.config import options_for, load as load_config
from mangle.context_llm import clean_output, mark_word
from mangle.homophones import group_of
from mangle.mangled_phrases import HOMOPHONE_CASES

GATE_FLIP = 0.80
GATE_KEEP = 0.95


def ask(cfg, model, sentence, token, group):
    options = ", ".join(sorted(group))
    system = (
        "You pick the correct word for a sentence. The sentence has one "
        f"word in double square brackets. Choose which of these fits that "
        f"position: {options}. Reply with exactly one of those options and "
        "nothing else. If unsure, reply with the bracketed word unchanged."
    )
    opts = options_for(cfg, model)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": mark_word(sentence, token)}],
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 8},
        "keep_alive": "10m",
    }
    t0 = time.perf_counter()
    r = requests.post(f"{cfg['ollama_url']}/api/chat", json=payload,
                      timeout=opts.get("timeout_s", 12.0) * 3)
    if r.status_code == 400:
        payload.pop("think", None)
        r = requests.post(f"{cfg['ollama_url']}/api/chat", json=payload,
                          timeout=opts.get("timeout_s", 12.0) * 3)
    r.raise_for_status()
    wall = (time.perf_counter() - t0) * 1000
    return clean_output(r.json().get("message", {}).get("content", "")), wall


def run(model, verbose=False):
    cfg = load_config()
    wrong = [c for c in HOMOPHONE_CASES if c["category"] == "wrong"]
    keep = [c for c in HOMOPHONE_CASES if c["category"] == "keep"]
    flips = kept = 0
    walls = []
    for case in HOMOPHONE_CASES:
        token, intended = case["mangled"], case["intended"]
        group = group_of(token)
        sentence = case["context"].format(token)
        try:
            answer, wall = ask(cfg, model, sentence, token, group)
        except requests.RequestException as e:
            print(f"  request error: {e}")
            continue
        walls.append(wall)
        # apply the production guard: outside-group answers keep the token
        final = answer.lower() if answer.lower() in group else token.lower()
        ok = final == intended.lower()
        if case["category"] == "wrong":
            flips += ok
        else:
            kept += ok
        if verbose:
            mark = "OK " if ok else "BAD"
            print(f"  {mark} [{case['category']:5}] {token:>8} -> {final:<8} "
                  f"(want {intended}) [{wall:.0f}ms]")
    flip_acc = flips / len(wrong)
    keep_acc = kept / len(keep)
    gate = flip_acc >= GATE_FLIP and keep_acc >= GATE_KEEP
    walls.sort()
    print(f"\n{model}: flip {flip_acc*100:.0f}%  keep {keep_acc*100:.0f}%  "
          f"wall p50 {walls[len(walls)//2]:.0f}ms  "
          f"-> {'GATE PASS' if gate else 'gate FAIL'}")
    return {"model": model, "flip": flip_acc, "keep": keep_acc, "gate": gate,
            "wall_p50": walls[len(walls) // 2] if walls else 0}


def main():
    models = sys.argv[1:] or ["gemma4:e2b"]
    results = [run(m, verbose=True) for m in models]
    print("\nSUMMARY")
    for r in results:
        print(f"  {r['model']:<14} flip {r['flip']*100:>4.0f}%  keep {r['keep']*100:>4.0f}%  "
              f"p50 {r['wall_p50']:>5.0f}ms  {'PASS' if r['gate'] else 'fail'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 0 feasibility benchmark: can a local model recover MANGLED typos
from sentence context, without wrecking already-correct tokens?

This is the blocking gate for Layer 3. It reuses benchmark.py's Ollama call and
eval_duration timing so numbers are comparable with the earlier autocorrect runs.

Fill-in-the-blank framing (this is exactly how Layer 3 will be used): the sentence
is shown with the target token marked [[likethis]], and the model must output only
the single intended word. Valid words never reach this layer in production, so the
passthrough set here uses the tokens that actually would reach it (names, slang,
technical terms) and checks the model leaves them alone.

Gate: a model qualifies only if passthrough >= 95% AND it recovers a meaningful
share of the mangles.

    python -m mangle.benchmark_recovery                      # all installed of the set
    python -m mangle.benchmark_recovery gemma4:e2b qwen3:0.6b
    python -m mangle.benchmark_recovery --json mangle/recovery_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import ollama_correct, list_models  # reuse HTTP + eval_duration timing
from mangle.mangled_phrases import RECOVERY_CASES
# same prompt + options + guard the app runs in production, so the benchmark is honest
from mangle.context_llm import (FILL_IN_BLANK_PROMPT, FILL_OPTIONS as OPTIONS,
                                overcorrection_guard)

# when True, apply the deployed capitalization/length guard before scoring, so
# the numbers reflect the real Layer 3 (model + guard), not the raw model alone
USE_GUARD = True

DEFAULT_MODELS = ["gemma4:e2b", "gemma3n:e2b", "gemma2:2b", "gemma3:1b", "qwen3:0.6b"]

PASSTHROUGH_GATE = 0.95
RECOVERY_FLOOR = 0.50  # "meaningful share" of mangles recovered


def marked_sentence(case: dict) -> str:
    return case["context"].format(f"[[{case['mangled']}]]")


def _clean(raw: str) -> str:
    """Model's single-word answer, stripped, for the guard check."""
    from mangle.context_llm import clean_output
    return clean_output(raw)


def score(case: dict, output: str) -> dict:
    raw = (output or "").strip()
    stripped = raw.strip('"\'`[]').strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    core = first_line.strip('.,!?;:"\'`()[]').strip()
    single = bool(core) and re.search(r"\s", core) is None
    intended = case["intended"].strip().lower()
    exact = core.lower() == intended
    contains = re.search(r"(?<!\w)" + re.escape(intended) + r"(?!\w)", raw.lower()) is not None
    return {"exact": exact, "contains": contains, "single": single, "out": first_line}


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]


def benchmark_model(model: str, cases=None, progress=None) -> dict:
    cases = cases or RECOVERY_CASES
    warm = ollama_correct(model, marked_sentence(cases[0]), FILL_IN_BLANK_PROMPT,
                          OPTIONS, timeout=180)
    if not warm["ok"]:
        return {"model": model, "ok": False, "error": warm["error"]}

    results = []
    for i, case in enumerate(cases):
        res = ollama_correct(model, marked_sentence(case), FILL_IN_BLANK_PROMPT,
                             OPTIONS, timeout=30)
        if res["ok"]:
            out = res["output"]
            if USE_GUARD:
                ans = _clean(out)
                # the deployed guard keeps the original token on a rejected fix
                if ans and overcorrection_guard(case["mangled"], ans):
                    out = case["mangled"]
                    res["guarded"] = True
            res["score"] = score(case, out)
        res["case"] = case
        results.append(res)
        if progress:
            progress(i + 1, len(cases))

    ok = [r for r in results if r["ok"]]
    recov = [r for r in ok if r["case"]["category"] in ("mash", "transposition")]
    passth = [r for r in ok if r["case"]["category"] == "passthrough"]

    def acc(rows, key="exact"):
        return sum(r["score"][key] for r in rows) / len(rows) if rows else 0.0

    def cat_acc(cat):
        rows = [r for r in ok if r["case"]["category"] == cat]
        return acc(rows) if rows else None

    eval_ms = [r["eval_ms"] for r in ok]
    wall_ms = [r["wall_ms"] for r in ok]
    total_tok = sum(r["eval_tokens"] for r in ok)
    total_s = sum(r["eval_ms"] for r in ok) / 1000

    recovery_acc = acc(recov)
    passthrough_rate = acc(passth)
    qualifies = passthrough_rate >= PASSTHROUGH_GATE and recovery_acc >= RECOVERY_FLOOR

    return {
        "model": model, "ok": True,
        "load_ms_first_request": warm["load_ms"],
        "n_cases": len(cases), "n_errors": len(results) - len(ok),
        "recovery_acc": recovery_acc,
        "recovery_contains": acc(recov, "contains"),
        "mash_acc": cat_acc("mash"),
        "transposition_acc": cat_acc("transposition"),
        "passthrough_rate": passthrough_rate,
        "discipline_rate": acc(ok, "single"),
        "qualifies": qualifies,
        "eval_ms_p50": pct(eval_ms, 50), "eval_ms_p95": pct(eval_ms, 95),
        "wall_ms_p50": pct(wall_ms, 50), "wall_ms_p95": pct(wall_ms, 95),
        "tok_per_s": total_tok / total_s if total_s else 0,
        "results": results,
    }


def print_report(summaries: list[dict], verbose=False) -> None:
    for s in summaries:
        if not s["ok"]:
            print(f"\n!! {s['model']}: FAILED - {s['error']}")
            continue
        gate = "QUALIFIES" if s["qualifies"] else "does NOT qualify"
        print(f"\n{'=' * 72}\nMODEL: {s['model']}   [{gate} for Layer 3]\n{'=' * 72}")
        print(f"  recovery accuracy (exact) : {s['recovery_acc'] * 100:6.1f} %"
              f"   (loose/contains {s['recovery_contains'] * 100:.0f}%)")
        if s["mash_acc"] is not None:
            print(f"    - mash (heavy scramble) : {s['mash_acc'] * 100:6.1f} %")
        if s["transposition_acc"] is not None:
            print(f"    - transposition         : {s['transposition_acc'] * 100:6.1f} %")
        print(f"  PASSTHROUGH (gate >=95%)  : {s['passthrough_rate'] * 100:6.1f} %"
              f"   {'PASS' if s['passthrough_rate'] >= PASSTHROUGH_GATE else 'FAIL'}")
        print(f"  single-word discipline    : {s['discipline_rate'] * 100:6.1f} %")
        print(f"  GPU eval (eval_duration)  : p50 {s['eval_ms_p50']:6.0f} ms   p95 {s['eval_ms_p95']:6.0f} ms")
        print(f"  wall clock per request    : p50 {s['wall_ms_p50']:6.0f} ms   p95 {s['wall_ms_p95']:6.0f} ms")
        print(f"  generation speed          : {s['tok_per_s']:6.0f} tok/s")
        if s["n_errors"]:
            print(f"  request errors            : {s['n_errors']}")
        if verbose:
            for r in s["results"]:
                if not r["ok"]:
                    print(f"    ERR {marked_sentence(r['case'])[:48]:<48} -> {r['error']}")
                    continue
                sc = r["score"]
                mark = "OK " if sc["exact"] else ("~  " if sc["contains"] else "BAD")
                cat = r["case"]["category"][:4]
                print(f"    {mark} [{cat}] {r['case']['mangled']:>12} -> {sc['out'][:28]:<28}"
                      f" (want {r['case']['intended']}) [{r['eval_ms']:.0f}ms]")

    ok_s = [s for s in summaries if s["ok"]]
    if len(ok_s) > 1:
        print(f"\n{'=' * 72}\nSIDE-BY-SIDE  (gate: passthrough >=95% AND recovery >=50%)\n{'=' * 72}")
        hdr = f"{'model':<20}{'recovery':>10}{'mash':>7}{'passthru':>10}{'disc':>7}{'eval p50':>10}{'gate':>6}"
        print(hdr + "\n" + "-" * len(hdr))
        for s in sorted(ok_s, key=lambda x: (-x["qualifies"], -x["recovery_acc"])):
            mash = f"{s['mash_acc'] * 100:.0f}%" if s["mash_acc"] is not None else "-"
            print(f"{s['model']:<20}{s['recovery_acc'] * 100:>9.1f}%{mash:>7}"
                  f"{s['passthrough_rate'] * 100:>9.0f}%{s['discipline_rate'] * 100:>6.0f}%"
                  f"{s['eval_ms_p50']:>9.0f}m{'  OK' if s['qualifies'] else ' no':>6}")
        winners = [s for s in ok_s if s["qualifies"]]
        if winners:
            best = max(winners, key=lambda x: x["recovery_acc"])
            print(f"\nWINNER: {best['model']}  "
                  f"(recovery {best['recovery_acc'] * 100:.0f}%, "
                  f"passthrough {best['passthrough_rate'] * 100:.0f}%, "
                  f"eval p50 {best['eval_ms_p50']:.0f}ms)")
        else:
            print("\nNO MODEL QUALIFIES. Do not build Layer 3 on these results.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 mangled-typo recovery benchmark")
    ap.add_argument("models", nargs="*", help="models (default: the 5-model set, installed only)")
    ap.add_argument("--json", metavar="FILE")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # models can emit stray non-Latin characters; never let printing them crash
    # the run on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    installed = set(list_models())
    requested = args.models or DEFAULT_MODELS
    models, missing = [], []
    for m in requested:
        (models if m in installed else missing).append(m)
    if missing:
        print(f"NOTE: not installed, skipping: {', '.join(missing)}")
    if not models:
        print("No requested models are installed.", file=sys.stderr)
        return 1

    summaries = []
    for model in models:
        print(f"benchmarking {model} ({len(RECOVERY_CASES)} cases)...", flush=True)
        summaries.append(benchmark_model(
            model, progress=lambda i, n: print(f"  {i}/{n}", end="\r", flush=True)))
        # write after each model so a later crash never loses completed results
        if args.json:
            Path(args.json).write_text(json.dumps(summaries, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    print_report(summaries, verbose=args.verbose)
    if args.json:
        print(f"\nfull results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

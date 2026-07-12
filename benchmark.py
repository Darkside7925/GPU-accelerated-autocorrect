"""Ollama autocorrect model benchmark harness.

Fires a fixed set of realistic typo / context-error phrases at one or more
Ollama models and reports, per model:

  - pure inference time from Ollama's own eval_duration / prompt_eval_duration
    (GPU time, excludes HTTP + model-load overhead)
  - wall-clock time per request (what the app will actually feel)
  - tokens generated and tok/s
  - correction accuracy: exact-match, targeted-fix rate, and false-positive
    rate on already-correct "passthrough" sentences

Usage:
    python benchmark.py qwen3:0.6b gnokit/improve-grammar
    python benchmark.py --json results.json qwen3:0.6b

Designed to be imported by the app later as its built-in "test model" feature:
    from benchmark import benchmark_model, DEFAULT_SYSTEM_PROMPT
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time

import requests

from test_phrases import TEST_CASES

# 127.0.0.1, NOT localhost: on Windows "localhost" resolves IPv6-first and adds
# a constant ~2s stall per request when Ollama only listens on IPv4 (measured).
OLLAMA_URL = "http://127.0.0.1:11434"

# minimal prompt won the bench_prompts.py comparison on qwen3:0.6b
# (68.2% targeted-fix vs 59.1% verbose, 100% passthrough for both);
# this is also the prompt the app uses, so in-app benchmarks match reality
DEFAULT_SYSTEM_PROMPT = (
    "Correct all spelling and grammar errors. Output only the corrected text."
)

DEFAULT_OPTIONS = {
    "temperature": 0,
    "num_predict": 80,   # sentences in the test set are short; hard cap runaway output
}


# ---------------------------------------------------------------- scoring

def _normalize(text: str) -> str:
    """Normalize for exact-match comparison: case, whitespace, edge punctuation."""
    text = text.strip().strip('"“”').strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".!")
    return text.lower()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Word-boundary match so 'teh' is not found inside 'the'."""
    return re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", text.lower()) is not None


def score_case(case: dict, output: str) -> dict:
    exact = _normalize(output) == _normalize(case["expected"])
    required_ok = all(_contains_phrase(output, p) for p in case["required"])
    forbidden_ok = not any(_contains_phrase(output, p) for p in case["forbidden"])
    return {
        "exact": exact,
        "fixed": exact or (required_ok and forbidden_ok),
        "required_ok": required_ok,
        "forbidden_ok": forbidden_ok,
    }


# ---------------------------------------------------------------- ollama

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3 etc.)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # unterminated think block (num_predict cut it off) -> nothing usable
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def ollama_correct(
    model: str,
    text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    options: dict | None = None,
    timeout: float = 60.0,
) -> dict:
    """One correction request via /api/chat. Returns output text + raw timings."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,  # disable qwen3-style thinking; ignored by non-reasoning models
        "options": dict(DEFAULT_OPTIONS, **(options or {})),
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        if r.status_code == 400:
            # older Ollama rejects the "think" field -> retry without it
            payload.pop("think", None)
            t0 = time.perf_counter()
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "wall_ms": (time.perf_counter() - t0) * 1000}
    wall_ms = (time.perf_counter() - t0) * 1000
    data = r.json()
    output = strip_thinking(data.get("message", {}).get("content", ""))
    return {
        "ok": True,
        "output": output,
        "wall_ms": wall_ms,
        "total_ms": data.get("total_duration", 0) / 1e6,
        "load_ms": data.get("load_duration", 0) / 1e6,
        "prompt_eval_ms": data.get("prompt_eval_duration", 0) / 1e6,
        "eval_ms": data.get("eval_duration", 0) / 1e6,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "eval_tokens": data.get("eval_count", 0),
    }


# ---------------------------------------------------------------- benchmark

def benchmark_model(
    model: str,
    cases: list[dict] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    options: dict | None = None,
    timeout: float = 60.0,
    progress=None,
) -> dict:
    """Run the full test set against one model. Returns a summary dict."""
    cases = cases or TEST_CASES

    # warm-up: load the model into VRAM so measured runs exclude load time
    warm = ollama_correct(model, "warm up", system_prompt, options, timeout=max(timeout, 120))
    if not warm["ok"]:
        return {"model": model, "ok": False, "error": warm["error"]}

    results = []
    for i, case in enumerate(cases):
        res = ollama_correct(model, case["input"], system_prompt, options, timeout)
        if res["ok"]:
            res["score"] = score_case(case, res["output"])
        res["case"] = case
        results.append(res)
        if progress:
            progress(i + 1, len(cases), res)

    ok = [r for r in results if r["ok"]]
    scored = [r for r in ok if r["case"]["category"] != "passthrough"]
    passthrough = [r for r in ok if r["case"]["category"] == "passthrough"]
    eval_ms = [r["eval_ms"] for r in ok]
    infer_ms = [r["prompt_eval_ms"] + r["eval_ms"] for r in ok]
    wall_ms = [r["wall_ms"] for r in ok]
    total_eval_tokens = sum(r["eval_tokens"] for r in ok)
    total_eval_s = sum(r["eval_ms"] for r in ok) / 1000

    def pct(vals, p):
        if not vals:
            return 0.0
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]

    by_cat = {}
    for cat in ("typo", "context"):
        cat_r = [r for r in scored if r["case"]["category"] == cat]
        if cat_r:
            by_cat[cat] = sum(r["score"]["fixed"] for r in cat_r) / len(cat_r)

    return {
        "model": model,
        "ok": True,
        "load_ms_first_request": warm["load_ms"],
        "n_cases": len(cases),
        "n_errors": len(results) - len(ok),
        "exact_acc": sum(r["score"]["exact"] for r in scored) / len(scored) if scored else 0,
        "fixed_acc": sum(r["score"]["fixed"] for r in scored) / len(scored) if scored else 0,
        "acc_by_category": by_cat,
        "passthrough_ok": (
            sum(r["score"]["exact"] for r in passthrough) / len(passthrough)
            if passthrough else None
        ),
        "eval_ms_mean": statistics.mean(eval_ms) if eval_ms else 0,
        "eval_ms_p50": pct(eval_ms, 50),
        "eval_ms_p95": pct(eval_ms, 95),
        "infer_ms_p50": pct(infer_ms, 50),
        "infer_ms_p95": pct(infer_ms, 95),
        "wall_ms_p50": pct(wall_ms, 50),
        "wall_ms_p95": pct(wall_ms, 95),
        "tok_per_s": total_eval_tokens / total_eval_s if total_eval_s else 0,
        "eval_tokens_mean": total_eval_tokens / len(ok) if ok else 0,
        "results": results,
    }


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


# ---------------------------------------------------------------- report

def print_report(summaries: list[dict], verbose: bool = False) -> None:
    for s in summaries:
        if not s["ok"]:
            print(f"\n!! {s['model']}: FAILED - {s['error']}")
            continue
        print(f"\n{'=' * 72}")
        print(f"MODEL: {s['model']}")
        print(f"{'=' * 72}")
        print(f"  first-request load time : {s['load_ms_first_request']:8.0f} ms (one-time)")
        print(f"  pure eval (GPU gen)     : p50 {s['eval_ms_p50']:6.0f} ms   p95 {s['eval_ms_p95']:6.0f} ms")
        print(f"  prompt+eval (inference) : p50 {s['infer_ms_p50']:6.0f} ms   p95 {s['infer_ms_p95']:6.0f} ms")
        print(f"  wall clock per request  : p50 {s['wall_ms_p50']:6.0f} ms   p95 {s['wall_ms_p95']:6.0f} ms")
        print(f"  generation speed        : {s['tok_per_s']:8.1f} tok/s "
              f"(avg {s['eval_tokens_mean']:.0f} tokens/reply)")
        print(f"  accuracy (targeted fix) : {s['fixed_acc'] * 100:6.1f} %")
        print(f"  accuracy (exact match)  : {s['exact_acc'] * 100:6.1f} %")
        for cat, acc in s["acc_by_category"].items():
            print(f"    - {cat:<10}          : {acc * 100:6.1f} %")
        if s["passthrough_ok"] is not None:
            print(f"  passthrough unchanged   : {s['passthrough_ok'] * 100:6.1f} % "
                  f"(correct text left alone)")
        if s["n_errors"]:
            print(f"  request errors          : {s['n_errors']}")
        if verbose:
            print(f"\n  {'input':<45} -> output")
            for r in s["results"]:
                if not r["ok"]:
                    print(f"  ERR {r['case']['input'][:42]:<42} -> {r['error']}")
                    continue
                mark = "OK " if r["score"]["fixed"] else "BAD"
                print(f"  {mark} {r['case']['input'][:42]:<42} -> {r['output'][:60]}"
                      f"  [{r['eval_ms']:.0f}ms eval]")

    ok_s = [s for s in summaries if s["ok"]]
    if len(ok_s) > 1:
        print(f"\n{'=' * 72}")
        print("SIDE-BY-SIDE")
        print(f"{'=' * 72}")
        hdr = f"{'model':<28}{'fix acc':>8}{'exact':>8}{'eval p50':>10}{'wall p50':>10}{'tok/s':>8}"
        print(hdr)
        print("-" * len(hdr))
        for s in sorted(ok_s, key=lambda x: (-x["fixed_acc"], x["wall_ms_p50"])):
            print(f"{s['model']:<28}{s['fixed_acc'] * 100:>7.1f}%{s['exact_acc'] * 100:>7.1f}%"
                  f"{s['eval_ms_p50']:>9.0f}m{s['wall_ms_p50']:>9.0f}m{s['tok_per_s']:>8.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Ollama models for autocorrect")
    ap.add_argument("models", nargs="*", help="Ollama model names (default: all installed)")
    ap.add_argument("--json", metavar="FILE", help="also dump full results to JSON")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--num-predict", type=int, default=DEFAULT_OPTIONS["num_predict"])
    ap.add_argument("-v", "--verbose", action="store_true", help="print every case")
    args = ap.parse_args()

    models = args.models or list_models()
    if not models:
        print("No models specified and none installed in Ollama.", file=sys.stderr)
        return 1

    options = {"num_predict": args.num_predict}
    summaries = []
    for model in models:
        print(f"benchmarking {model} ({len(TEST_CASES)} cases)...", flush=True)
        s = benchmark_model(
            model, options=options, timeout=args.timeout,
            progress=lambda i, n, r: print(f"  {i}/{n}", end="\r", flush=True),
        )
        summaries.append(s)

    print_report(summaries, verbose=args.verbose)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\nfull results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

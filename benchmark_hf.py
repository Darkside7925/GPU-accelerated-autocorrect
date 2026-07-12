"""HuggingFace seq2seq backend for the autocorrect benchmark.

For T5-style correction models that Ollama cannot run (encoder-decoder
architecture), e.g.:

    python benchmark_hf.py ai-forever/T5-large-spell vennify/t5-base-grammar-correction

Uses the exact same test set and scoring as benchmark.py so results are
directly comparable with the Ollama models. "eval time" here is pure GPU
generate() time (cuda-synchronized), the closest equivalent of Ollama's
eval_duration.

Both supported models were trained with the "grammar: " task prefix.
--beams N runs beam search (vennify's model card recommends 5); default is
greedy to match the temperature-0 setup used for the Ollama models.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import truststore

truststore.inject_into_ssl()  # this machine's TLS is intercepted; use OS certs

from benchmark import print_report
from test_phrases import TEST_CASES

PREFIX = "grammar: "


def benchmark_hf_model(model_id: str, beams: int = 1, cases=None, progress=None) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cases = cases or TEST_CASES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000

    def generate(text: str) -> tuple[str, float, int]:
        enc = tokenizer(PREFIX + text, return_tensors="pt").to(device)
        max_new = int(enc["input_ids"].shape[1] * 1.5) + 16
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=max_new, num_beams=beams, do_sample=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) * 1000
        n_tokens = out.shape[1]
        return tokenizer.decode(out[0], skip_special_tokens=True).strip(), gen_ms, n_tokens

    generate("warm up the model")  # cuda kernels, cache

    from benchmark import score_case  # same scorer as the Ollama run

    results = []
    for i, case in enumerate(cases):
        output, gen_ms, n_tok = generate(case["input"])
        res = {
            "ok": True, "output": output, "wall_ms": gen_ms,
            "total_ms": gen_ms, "load_ms": 0.0,
            "prompt_eval_ms": 0.0, "eval_ms": gen_ms,
            "prompt_tokens": 0, "eval_tokens": n_tok,
            "score": score_case(case, output), "case": case,
        }
        results.append(res)
        if progress:
            progress(i + 1, len(cases), res)

    label = model_id if beams == 1 else f"{model_id} (beams={beams})"
    scored = [r for r in results if r["case"]["category"] != "passthrough"]
    passthrough = [r for r in results if r["case"]["category"] == "passthrough"]
    eval_ms = sorted(r["eval_ms"] for r in results)
    total_tok = sum(r["eval_tokens"] for r in results)
    total_s = sum(eval_ms) / 1000

    def pct(vals, p):
        return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]

    by_cat = {}
    for cat in ("typo", "context"):
        cat_r = [r for r in scored if r["case"]["category"] == cat]
        if cat_r:
            by_cat[cat] = sum(r["score"]["fixed"] for r in cat_r) / len(cat_r)

    # free VRAM before the next model loads
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": label, "ok": True,
        "load_ms_first_request": load_ms,
        "n_cases": len(cases), "n_errors": 0,
        "exact_acc": sum(r["score"]["exact"] for r in scored) / len(scored),
        "fixed_acc": sum(r["score"]["fixed"] for r in scored) / len(scored),
        "acc_by_category": by_cat,
        "passthrough_ok": (
            sum(r["score"]["exact"] for r in passthrough) / len(passthrough)
            if passthrough else None
        ),
        "eval_ms_mean": statistics.mean(eval_ms),
        "eval_ms_p50": pct(eval_ms, 50), "eval_ms_p95": pct(eval_ms, 95),
        "infer_ms_p50": pct(eval_ms, 50), "infer_ms_p95": pct(eval_ms, 95),
        "wall_ms_p50": pct(eval_ms, 50), "wall_ms_p95": pct(eval_ms, 95),
        "tok_per_s": total_tok / total_s if total_s else 0,
        "eval_tokens_mean": total_tok / len(results),
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark HF seq2seq correction models")
    ap.add_argument("models", nargs="+", help="HF model ids")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--json", metavar="FILE")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    summaries = []
    for model_id in args.models:
        print(f"benchmarking {model_id} (beams={args.beams}, {len(TEST_CASES)} cases)...",
              flush=True)
        summaries.append(benchmark_hf_model(
            model_id, beams=args.beams,
            progress=lambda i, n, r: print(f"  {i}/{n}", end="\r", flush=True),
        ))

    print_report(summaries, verbose=args.verbose)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\nfull results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

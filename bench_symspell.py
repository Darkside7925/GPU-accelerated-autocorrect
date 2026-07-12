"""Micro-benchmark for Stage 1: SymSpell per-word lookup latency in Python.

Decides whether the Python hot path is fast enough at 143 WPM (~2.4 words/s,
so the budget per word is huge; the real question is whether lookup stays
sub-millisecond so the keyboard hook thread never blocks typing).
"""

import statistics
import time

from symspellpy import SymSpell, Verbosity
import importlib.resources

WORDS = [
    "teh", "jsut", "recieve", "definately", "seperate", "accomodate",
    "wich", "becuase", "freind", "adress", "hte", "taht", "woudl",
    "thier", "chekc", "runing", "meetign", "probaly", "aproach",
    "hello", "world", "benchmark", "keyboard", "correct",  # valid words too
]

def main():
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
    t0 = time.perf_counter()
    sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
    load_s = time.perf_counter() - t0
    print(f"dictionary load: {load_s:.2f}s ({sym.word_count} words)")

    # warm up
    for w in WORDS:
        sym.lookup(w, Verbosity.TOP, max_edit_distance=2, include_unknown=True)

    times = []
    for _ in range(200):
        for w in WORDS:
            t0 = time.perf_counter()
            sym.lookup(w, Verbosity.TOP, max_edit_distance=2, include_unknown=True)
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    print(f"lookups: {len(times)}")
    print(f"  mean : {statistics.mean(times):.4f} ms")
    print(f"  p50  : {times[len(times)//2]:.4f} ms")
    print(f"  p95  : {times[int(len(times)*0.95)]:.4f} ms")
    print(f"  p99  : {times[int(len(times)*0.99)]:.4f} ms")
    print(f"  max  : {times[-1]:.4f} ms")

    # sanity: show a few corrections
    for w in ["teh", "jsut", "recieve", "thier", "hello"]:
        s = sym.lookup(w, Verbosity.TOP, max_edit_distance=2, include_unknown=True)[0]
        print(f"  {w} -> {s.term} (dist {s.distance})")

if __name__ == "__main__":
    main()

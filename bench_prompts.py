"""Prompt-variant benchmark: same model, same test set, different system prompts.

Finds the best prompt config for the winning model rather than assuming the
first prompt written is optimal.

    python bench_prompts.py qwen3:0.6b
"""

import sys

from benchmark import DEFAULT_SYSTEM_PROMPT, benchmark_model, print_report

FEWSHOT_PROMPT = (
    "You are an autocorrect engine. Fix spelling and wrong word choices in the "
    "user's text. Keep the meaning, tone, and wording otherwise identical. If "
    "the text is already correct, return it unchanged. Reply with ONLY the "
    "corrected text.\n\n"
    "Examples:\n"
    "Input: their going to teh store\n"
    "Output: they're going to the store\n"
    "Input: I recieved you're package yesterday\n"
    "Output: I received your package yesterday\n"
    "Input: this option is more expensive then that one\n"
    "Output: this option is more expensive than that one\n"
    "Input: the meeting starts at noon\n"
    "Output: the meeting starts at noon"
)

MINIMAL_PROMPT = (
    "Correct all spelling and grammar errors. Output only the corrected text."
)

VARIANTS = [
    ("baseline", DEFAULT_SYSTEM_PROMPT),
    ("fewshot", FEWSHOT_PROMPT),
    ("minimal", MINIMAL_PROMPT),
]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:0.6b"
    summaries = []
    for name, prompt in VARIANTS:
        print(f"benchmarking {model} [{name}]...", flush=True)
        s = benchmark_model(model, system_prompt=prompt)
        s["model"] = f"{model} [{name}]"
        summaries.append(s)
    print_report(summaries)


if __name__ == "__main__":
    main()

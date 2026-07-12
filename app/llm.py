"""Stage 2: async LLM sentence correction via local Ollama (GPU).

A single worker thread pulls completed sentences off a queue, sends them to
the active Ollama model, filters the result against the personal dictionary,
and hands the corrected sentence back to the engine via callback. The engine
decides whether it is still safe to apply (idle gating / cursor drift).
"""

from __future__ import annotations

import difflib
import queue
import re
import threading

import requests

# benchmarked best of 3 variants on qwen3:0.6b (68.2% vs 59.1% for the verbose
# prompt, same 100% passthrough) - see bench_prompts.py
SYSTEM_PROMPT = (
    "Correct all spelling and grammar errors. Output only the corrected text."
)

_WORD_SPLIT = re.compile(r"(\W+)")


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def filter_protected(original: str, corrected: str, personal) -> str:
    """Revert any LLM change that touches a personal-dictionary word."""
    orig_tokens = _WORD_SPLIT.split(original)
    corr_tokens = _WORD_SPLIT.split(corrected)
    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens, autojunk=False)
    out = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            out.extend(corr_tokens[b0:b1])
        elif any(personal.contains(t.strip("'\"").lower())
                 for t in orig_tokens[a0:a1] if t.strip()):
            out.extend(orig_tokens[a0:a1])  # protected word touched -> keep original
        else:
            out.extend(corr_tokens[b0:b1])
    return "".join(out)


class LLMWorker:
    def __init__(self, cfg: dict, personal, on_result):
        """on_result(job_id, original_sentence, corrected_sentence) - called
        from the worker thread whenever the model output differs."""
        self._cfg = cfg
        self._personal = personal
        self._on_result = on_result
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, daemon=True, name="llm-worker")
        self._thread.start()
        self.last_error: str | None = None

    def submit(self, job_id: int, sentence: str) -> None:
        try:
            self._queue.put_nowait((job_id, sentence))
        except queue.Full:
            pass  # typing faster than the GPU corrects; drop rather than lag

    def _run(self) -> None:
        while True:
            job_id, sentence = self._queue.get()
            try:
                corrected = self._correct(sentence)
            except requests.RequestException as e:
                self.last_error = str(e)
                continue
            if corrected is None:
                continue
            corrected = filter_protected(sentence, corrected, self._personal)
            if corrected.strip() and corrected != sentence:
                self._on_result(job_id, sentence, corrected)

    def _correct(self, sentence: str) -> str | None:
        from app.config import options_for  # late import to avoid cycle
        model = self._cfg["active_model"]
        opts = options_for(self._cfg, model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": opts.get("temperature", 0),
                "num_predict": opts.get("num_predict", 80),
            },
            "keep_alive": "30m",
        }
        url = f"{self._cfg['ollama_url']}/api/chat"
        r = requests.post(url, json=payload, timeout=opts.get("timeout_s", 10.0))
        if r.status_code == 400:  # older Ollama without "think" support
            payload.pop("think", None)
            r = requests.post(url, json=payload, timeout=opts.get("timeout_s", 10.0))
        r.raise_for_status()
        out = strip_thinking(r.json().get("message", {}).get("content", ""))
        if not out:
            return None
        # guard against chatty models: reject wildly different lengths
        if len(out) > len(sentence) * 2 + 40 or len(out) < len(sentence) // 3:
            return None
        return out


def list_models(ollama_url: str) -> list[str]:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []

"""Config load/save for the autocorrect app."""

from __future__ import annotations

import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.json"
DB_PATH = APP_DIR / "personal.db"

DEFAULTS = {
    "enabled": True,
    "active_model": "gemma4:e4b",          # benchmark winner: 100% recovery, 100% homophone flip
    # 127.0.0.1, NOT localhost: avoids a measured ~2s IPv6-fallback stall per
    # request on Windows when Ollama listens on IPv4 only
    "ollama_url": "http://127.0.0.1:11434",
    "toggle_hotkey": "<ctrl>+<alt>+a",     # global on/off toggle
    "stage2_enabled": True,
    "stage2_idle_ms": 350,                 # apply LLM fixes only after this typing pause
    "stage2_fire_idle_ms": 500,            # send deferred words to the LLM after this pause,
                                           # even without a sentence terminator
    "sync_settle_ms": 120,                 # only render corrections after this much idle, so
                                           # no keystroke is still in flight (prevents mis-placed fixes)
    "transactional_sync": True,            # hold+replay printable keys during the few-ms injection
                                           # so fast typing cannot interleave with a correction
    "context_homophones": True,            # guarded LLM check of valid homophones (to/too, their/
                                           # there); answers outside the confusion group are refused
    "stage2_max_drift_chars": 100,         # largest backspace reach when rendering a correction
    "undo_window_s": 4.0,                  # backspace within this window counts as a rejection
    "reject_threshold": 3,                 # stop correcting a word after this many rejections
    "min_word_len": 3,                     # don't autocorrect 1-2 letter words
    # layered recovery gates
    "llm_only": False,                     # LLM-only mode: skip the deterministic L1 (memory) and
                                           # L2 (keyboard/phonetic) auto-fixes and send every word
                                           # needing correction straight to the context LLM. Valid
                                           # words still pass through untouched. Slower, but the LLM
                                           # (not a heuristic) makes every call, using the sentence.
    "layer1_apply_confidence": 0.50,       # min personal-memory confidence to auto-apply
    "layer2_apply_confidence": 0.72,       # min matcher confidence to auto-apply, else defer to L3.
                                           # tuned to catch clear near-misses without misfiring
    "layer2_short_confidence": 0.85,       # higher bar for short words (<=4 chars): keyboard
                                           # matching is unreliable there, context should decide
    "hint_min_confidence": 0.55,           # only hint the LLM with Layer 2 candidates this good
    "join_split_words": True,              # fix wrong spaces: inc rease -> increase, itsthe -> its the
    "join_fuzzy_confidence": 0.45,         # merge a mistyped 2nd fragment (inc erease -> increase) if the
                                           # fix still starts with the correctly-typed prefix
    # dashboard
    "dashboard_enabled": True,
    "dashboard_hostname": "grammer.local", # add "127.0.0.1 grammer.local" to the hosts file
    "dashboard_port": 80,                  # falls back to 8080 if 80 is taken
    "model_options": {
        # per-model overrides; anything missing falls back to _default
        "_default": {"num_predict": 80, "temperature": 0, "timeout_s": 10.0},
    },
}


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k == "model_options":
                    cfg["model_options"].update(v)
                else:
                    cfg[k] = v
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def options_for(cfg: dict, model: str) -> dict:
    opts = dict(cfg["model_options"]["_default"])
    opts.update(cfg["model_options"].get(model, {}))
    return opts

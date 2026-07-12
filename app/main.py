"""Autocorrect entry point.

Run with:  pythonw -m app.main   (no console)
      or:  python -m app.main    (console + logs, for debugging)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config as config_mod
from app.engine import AutocorrectEngine
from app.personal_dict import PersonalDict
from app.tray import TrayApp
from mangle.context_llm import ContextLLMWorker
from mangle.matcher import Layer2Matcher
from mangle.pipeline import RecoveryPipeline
from mangle.typo_memory import TypoMemory


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(config_mod.APP_DIR / "app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("main")
    cfg = config_mod.load()

    personal = PersonalDict(config_mod.DB_PATH)      # never-touch whitelist + log
    memory = TypoMemory(config_mod.DB_PATH)          # Layer 1: personal typo memory
    from mangle.common_typos import seed as seed_common
    seeded = seed_common(memory)                     # one-time: instant common typos
    if seeded:
        log.info("seeded %d common typos into Layer 1", seeded)
    log.info("whitelist %d words, %d learned typo pairs",
             len(personal.all_words()), memory.stats()["pairs"])

    matcher = Layer2Matcher(personal=personal)       # Layer 2: keyboard + phonetic
    log.info("layer 2 matcher indexed")
    pipeline = RecoveryPipeline(cfg, memory, matcher, personal)

    # run the end-of-day compaction if a day has passed since the last one
    try:
        from mangle.compact import maybe_run_daily
        maybe_run_daily(cfg, memory, personal, matcher)
    except Exception:
        log.exception("compaction check")

    engine = AutocorrectEngine(cfg, pipeline, personal, layer3=None)
    layer3 = ContextLLMWorker(cfg, personal, memory, on_result=engine.stage2_result,
                              is_word=matcher.is_dictionary_word)
    engine.layer3 = layer3
    engine.start()

    _warm_layer3(cfg, log)  # load the model into VRAM so the first fix is not cold

    def benchmark_fn(model, opts):
        from mangle.benchmark_recovery import benchmark_model
        return benchmark_model(model)

    dashboard = _start_dashboard(cfg, memory, personal, engine, log)

    tray = TrayApp(cfg, engine, benchmark_fn, dashboard_url=dashboard)
    log.info("tray starting; model=%s", cfg["active_model"])
    tray.run()  # blocks until Quit
    memory.close()
    personal.close()
    return 0


def _warm_layer3(cfg, log):
    """Fire one tiny request in the background so Ollama loads the Layer 3 model
    into VRAM now, instead of the first real correction paying an 8-11s cold
    load and missing its moment."""
    import threading

    def warm():
        import requests
        try:
            requests.post(f"{cfg['ollama_url']}/api/chat", timeout=120, json={
                "model": cfg["active_model"],
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "keep_alive": "2h", "options": {"num_predict": 1},
            })
            log.info("layer 3 model %s warmed", cfg["active_model"])
        except Exception:
            pass

    threading.Thread(target=warm, daemon=True).start()


def _start_dashboard(cfg, memory, personal, engine, log):
    """Launch the local dashboard in a daemon thread. Returns its URL or None."""
    if not cfg.get("dashboard_enabled", True):
        return None
    try:
        from dashboard.app import start_in_thread
        return start_in_thread(cfg, memory, personal, engine)
    except Exception:
        log.exception("dashboard failed to start")
        return None


if __name__ == "__main__":
    sys.exit(main())

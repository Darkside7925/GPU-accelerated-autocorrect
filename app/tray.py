"""System tray UI: toggle, model hot-swap dropdown, built-in model benchmark."""

from __future__ import annotations

import logging
import os
import threading

import pystray
from PIL import Image, ImageDraw
from pystray import Menu, MenuItem as Item

from app import config as config_mod
from app.llm import list_models

log = logging.getLogger("tray")


def _make_icon_image(enabled: bool) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (46, 160, 67, 255) if enabled else (110, 110, 110, 255)
    d.ellipse((4, 4, 60, 60), fill=color)
    d.text((20, 14), "A", fill=(255, 255, 255, 255), font_size=36)
    return img


class TrayApp:
    def __init__(self, cfg: dict, engine, benchmark_fn, dashboard_url=None):
        """benchmark_fn(model_name, options) -> summary dict (run off-thread)."""
        self.cfg = cfg
        self.engine = engine
        self.benchmark_fn = benchmark_fn
        self.dashboard_url = dashboard_url
        self._models: list[str] = []
        self._bench_running = False
        self.icon = pystray.Icon(
            "autocorrect", _make_icon_image(engine.enabled), "Autocorrect", self._menu()
        )
        engine.on_toggle = self._on_toggle
        threading.Thread(target=self._refresh_models, daemon=True).start()

    # ------------------------------------------------------------ menu

    def _menu(self) -> Menu:
        model_items = [
            Item(
                m,
                self._set_model(m),
                radio=True,
                checked=(lambda item, m=m: self.cfg["active_model"] == m),
            )
            for m in self._models
        ] or [Item("(no models found - is Ollama running?)", None, enabled=False)]
        model_items.append(Menu.SEPARATOR)
        model_items.append(Item("Refresh model list", lambda: self._refresh_models()))

        return Menu(
            Item(
                "Enabled  (Ctrl+Alt+A)",
                lambda: self.engine.toggle(),
                checked=lambda item: self.engine.enabled,
            ),
            Item(
                "Sentence pass (LLM)",
                self._toggle_stage2,
                checked=lambda item: self.cfg.get("stage2_enabled", True),
            ),
            Item("Model", Menu(*model_items)),
            Item("Benchmark current model", self._benchmark_current),
            Menu.SEPARATOR,
            Item("Open dashboard", self._open_dashboard, default=True,
                 visible=bool(self.dashboard_url)),
            Item("Run compaction now", self._compact_now),
            Menu.SEPARATOR,
            Item("Quit", self._quit),
        )

    def _open_dashboard(self):
        if self.dashboard_url:
            import webbrowser
            webbrowser.open(self.dashboard_url)

    def _compact_now(self):
        def run():
            try:
                from mangle.compact import compact
                memory = self.engine.memory
                personal = self.engine.personal
                is_word = self.engine.pipeline.matcher.is_dictionary_word
                res = compact(memory, personal, is_word, since_ts=0.0)
                self.icon.notify(
                    f"Compaction: +{res['promoted']} pairs, +{res['whitelisted']} words",
                    "Sumizome")
            except Exception as e:
                self.icon.notify(f"Compaction failed: {e}", "Sumizome")
        threading.Thread(target=run, daemon=True).start()

    def _set_model(self, model: str):
        def handler():
            self.cfg["active_model"] = model
            config_mod.save(self.cfg)
            log.info("model switched to %s", model)
            self.icon.notify(f"Model: {model}", "Autocorrect")
        return handler

    def _toggle_stage2(self):
        self.cfg["stage2_enabled"] = not self.cfg.get("stage2_enabled", True)
        config_mod.save(self.cfg)

    def _refresh_models(self):
        self._models = list_models(self.cfg["ollama_url"])
        self.icon.menu = self._menu()
        self.icon.update_menu()

    def _on_toggle(self, enabled: bool):
        self.cfg["enabled"] = enabled
        config_mod.save(self.cfg)
        self.icon.icon = _make_icon_image(enabled)

    # ------------------------------------------------------- benchmark

    def _benchmark_current(self):
        if self._bench_running:
            self.icon.notify("A benchmark is already running", "Autocorrect")
            return
        self._bench_running = True
        model = self.cfg["active_model"]
        self.icon.notify(f"Benchmarking {model}...", "Autocorrect")
        threading.Thread(target=self._run_benchmark, args=(model,), daemon=True).start()

    def _run_benchmark(self, model: str):
        try:
            opts = config_mod.options_for(self.cfg, model)
            summary = self.benchmark_fn(model, opts)
            if not summary["ok"]:
                self.icon.notify(f"Benchmark failed: {summary['error'][:120]}", "Autocorrect")
                return
            msg = (
                f"{model}: fix {summary['fixed_acc'] * 100:.0f}%, "
                f"wall p50 {summary['wall_ms_p50']:.0f}ms, "
                f"{summary['tok_per_s']:.0f} tok/s"
            )
            self.icon.notify(msg, "Benchmark done")
            report = config_mod.APP_DIR / "bench_report.txt"
            import contextlib, io
            from benchmark import print_report
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_report([summary], verbose=True)
            report.write_text(buf.getvalue(), encoding="utf-8")
            os.startfile(str(report))
        except Exception:
            log.exception("benchmark")
            self._bench_running = False
            return
        finally:
            self._bench_running = False

    def _quit(self):
        self.engine.stop()
        self.icon.stop()

    def run(self):
        self.icon.run()

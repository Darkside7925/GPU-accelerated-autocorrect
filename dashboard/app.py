"""Local dashboard for the Sumizome typo recovery engine.

Flask app, served on the loopback interface only, in a daemon thread alongside
the tray app. It exposes everything: the full personal typo table, the
whitelist, the derived typing insights, a live pipeline playground, editable
settings, the learning-loop status, the model and benchmark controls, the raw
log, and import/export. It is for the user's own eyes, so nothing is hidden.

Reads go through a private read-only SQLite connection in autocommit mode. WAL
lets it read the latest committed rows while the engine writer thread runs, so
the dashboard never shares a connection with the engine and cannot race it.
Writes (edits to the typo table and whitelist) go through the engine's own
thread-safe, lock-protected objects, so edits made here show up live in the
running engine.

Charts are hand-rolled inline SVG in the frontend, so there is no CDN and no
vendored library: fully offline by construction. No em-dashes anywhere.
"""

from __future__ import annotations

import logging
import socket
import sqlite3
import threading
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory

from dashboard import insights

log = logging.getLogger("dashboard")
STATIC = Path(__file__).resolve().parent / "static"
RESULTS_JSON = Path(__file__).resolve().parent.parent / "mangle" / "recovery_results.json"


def create_app(cfg, memory, personal, engine, db_path) -> Flask:
    app = Flask(__name__, static_folder=None)
    # own read-only connection, autocommit so every SELECT sees latest committed
    ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                         check_same_thread=False, isolation_level=None)
    ro.row_factory = sqlite3.Row
    rlock = threading.Lock()

    def q(sql, args=()):
        with rlock:
            try:
                return ro.execute(sql, args).fetchall()
            except sqlite3.OperationalError:
                return []  # table may not exist yet on a brand new db

    def pairs_rows():
        # exclude the bulk seeded dictionary so the table and insights reflect
        # what YOU actually type, not 58k codespell entries
        return [(r["mangled"], r["intended"], r["count"], r["confidence"],
                 r["first_seen"], r["last_seen"], r["source"])
                for r in q("SELECT mangled, intended, count, confidence, first_seen, "
                           "last_seen, source FROM typo_memory WHERE source != 'dataset' "
                           "ORDER BY count DESC, confidence DESC LIMIT 3000")]

    def correction_rows(since=0.0):
        return [(r["ts"], r["original"], r["corrected"], r["stage"], r["undone"])
                for r in q("SELECT ts, original, corrected, stage, undone FROM "
                           "correction_log WHERE ts >= ? ORDER BY ts", (since,))]

    def correction_counts():
        return [(r[0], r[1], r[2]) for r in
                q("SELECT stage, undone, COUNT(*) FROM correction_log GROUP BY stage, undone")]

    def whitelist_words():
        return [r["word"] for r in q("SELECT word FROM personal_words ORDER BY word")]

    def scalar(sql, args=(), default=0):
        rows = q(sql, args)
        return rows[0][0] if rows and rows[0][0] is not None else default

    # ------------------------------------------------------------ page
    @app.route("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.route("/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    # ------------------------------------------------------------ reads
    @app.route("/api/overview")
    def overview():
        today = time.strftime("%Y-%m-%d")
        today_start = time.mktime(time.strptime(today, "%Y-%m-%d"))
        return jsonify({
            "layers": insights.layer_totals(correction_counts()),
            "corrections_today": scalar(
                "SELECT COUNT(*) FROM correction_log WHERE ts >= ? AND undone = 0",
                (today_start,)),
            "memory": {
                "pairs": scalar("SELECT COUNT(*) FROM typo_memory WHERE source != 'dataset'"),
                "dictionary": scalar("SELECT COUNT(*) FROM typo_memory WHERE source = 'dataset'"),
                "raw_entries": scalar("SELECT COUNT(*) FROM raw_log"),
            },
            "whitelist_size": scalar("SELECT COUNT(*) FROM personal_words"),
            "model": cfg.get("active_model"),
            "engine_enabled": engine.enabled,
        })

    @app.route("/api/typos")
    def typos():
        return jsonify([
            {"mangled": m, "intended": i, "count": c, "confidence": round(conf, 2),
             "first_seen": fs, "last_seen": ls, "source": src}
            for (m, i, c, conf, fs, ls, src) in pairs_rows()
        ])

    @app.route("/api/whitelist")
    def whitelist():
        return jsonify(whitelist_words())

    @app.route("/api/insights")
    def insights_route():
        pairs = pairs_rows()
        corr = correction_rows()
        return jsonify({
            "key_heat": insights.key_heat(pairs),
            "keyboard_rows": insights.KEYBOARD_ROWS,
            "top_mangles": insights.top_mangles(pairs),
            "length_profile": insights.length_profile(pairs),
            "confidence_hist": insights.confidence_hist(pairs),
            "over_time": insights.corrections_over_time(corr),
            "hour_profile": insights.hour_profile(corr),
        })

    @app.route("/api/learning")
    def learning():
        last = scalar("SELECT value FROM meta WHERE key = 'last_compaction'", default=None)
        return jsonify({
            "last_compaction": float(last) if last else None,
            "pairs": scalar("SELECT COUNT(*) FROM typo_memory"),
            "whitelist_size": scalar("SELECT COUNT(*) FROM personal_words"),
            "over_time": insights.corrections_over_time(correction_rows()),
            "layers": insights.layer_totals(correction_counts()),
        })

    @app.route("/api/rawlog")
    def rawlog():
        since = float(request.args.get("since", 0))
        raws = [{"ts": r["ts"], "text": r["text"]} for r in
                q("SELECT ts, text FROM raw_log WHERE ts >= ? ORDER BY ts DESC LIMIT 500", (since,))]
        corr = [{"ts": ts, "original": o, "corrected": c, "stage": s, "undone": bool(u)}
                for (ts, o, c, s, u) in correction_rows(since)][-500:]
        return jsonify({"raw": raws, "corrections": corr})

    @app.route("/api/model")
    def model():
        from mangle.context_llm import FILL_IN_BLANK_PROMPT
        installed = []
        try:
            r = requests.get(f"{cfg['ollama_url']}/api/tags", timeout=3)
            installed = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        bench = None
        if RESULTS_JSON.exists():
            import json
            bench = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        return jsonify({
            "active": cfg.get("active_model"),
            "installed": installed,
            "options": cfg.get("model_options", {}),
            "prompt": FILL_IN_BLANK_PROMPT,
            "last_benchmark": bench,
        })

    @app.route("/api/settings")
    def get_settings():
        keys = ["stage2_enabled", "llm_only", "stage2_idle_ms", "stage2_max_drift_chars",
                "undo_window_s", "min_word_len", "layer1_apply_confidence",
                "layer2_apply_confidence", "ollama_url", "active_model",
                "dashboard_hostname", "dashboard_port"]
        return jsonify({k: cfg.get(k) for k in keys})

    @app.route("/api/version")
    def version():
        from app.updater import check_for_update
        return jsonify(check_for_update())

    @app.route("/api/health")
    def health():
        l3 = engine.layer3
        return jsonify({
            "engine_enabled": engine.enabled,
            "model": cfg.get("active_model"),
            "layer3_queue": l3._queue.qsize() if l3 else 0,
            "layer3_last_error": getattr(l3, "last_error", None),
            "stage2_enabled": cfg.get("stage2_enabled", True),
        })

    # ------------------------------------------------------------ writes
    @app.route("/api/typos", methods=["POST"])
    def add_typo():
        d = request.get_json(force=True)
        memory.record(d["mangled"], d["intended"], source="manual")
        return jsonify({"ok": True})

    @app.route("/api/typos", methods=["DELETE"])
    def del_typo():
        d = request.get_json(force=True)
        memory.demote(d["mangled"], d.get("intended"))
        return jsonify({"ok": True})

    @app.route("/api/whitelist", methods=["POST"])
    def add_white():
        personal.add(request.get_json(force=True)["word"], source="manual")
        return jsonify({"ok": True})

    @app.route("/api/whitelist", methods=["DELETE"])
    def del_white():
        personal.remove(request.get_json(force=True)["word"])
        return jsonify({"ok": True})

    @app.route("/api/settings", methods=["POST"])
    def set_settings():
        from app import config as config_mod
        d = request.get_json(force=True)
        numeric = {"stage2_idle_ms", "stage2_max_drift_chars", "min_word_len", "dashboard_port"}
        floats = {"undo_window_s", "layer1_apply_confidence", "layer2_apply_confidence"}
        for k, v in d.items():
            if k in numeric:
                cfg[k] = int(v)
            elif k in floats:
                cfg[k] = float(v)
            else:
                cfg[k] = v
        config_mod.save(cfg)
        return jsonify({"ok": True})

    @app.route("/api/model", methods=["POST"])
    def set_model():
        from app import config as config_mod
        name = request.get_json(force=True)["model"]
        cfg["active_model"] = name
        config_mod.save(cfg)
        return jsonify({"ok": True, "active": name})

    @app.route("/api/toggle", methods=["POST"])
    def toggle():
        engine.toggle()
        return jsonify({"enabled": engine.enabled})

    @app.route("/api/compact", methods=["POST"])
    def run_compact():
        from mangle.compact import compact
        res = compact(memory, personal, engine.pipeline.matcher.is_dictionary_word, since_ts=0.0)
        return jsonify(res)

    @app.route("/api/benchmark", methods=["POST"])
    def run_benchmark():
        model_name = (request.get_json(silent=True) or {}).get("model", cfg["active_model"])
        if getattr(app, "_bench_running", False):
            return jsonify({"ok": False, "error": "already running"})
        app._bench_running = True

        def worker():
            import json
            from mangle.benchmark_recovery import benchmark_model
            try:
                s = benchmark_model(model_name)
                RESULTS_JSON.write_text(json.dumps([s], indent=2, ensure_ascii=False),
                                        encoding="utf-8")
            except Exception:
                log.exception("benchmark")
            finally:
                app._bench_running = False

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "started": model_name})

    # -------------------------------------------------- pipeline playground
    @app.route("/api/test", methods=["POST"])
    def test_sentence():
        """Route a sentence through the real pipeline, word by word, and show
        what each layer does. Deferred tokens optionally hit Layer 3 live."""
        d = request.get_json(force=True)
        sentence = d.get("sentence", "")
        use_llm = d.get("use_llm", True)
        import re as _re
        tokens = _re.findall(r"[A-Za-z']+|[^A-Za-z']+", sentence)
        words, deferred, context, hints = [], [], [], {}
        for tok in tokens:
            if not tok[:1].isalpha():
                words.append({"text": tok, "kind": "sep"})
                continue
            res = engine.pipeline.on_word(tok)
            entry = {"text": tok, "kind": res.action, "layer": res.layer,
                     "intended": res.intended, "confidence": round(res.confidence, 2)}
            if res.action == "defer":
                deferred.append(tok)
                if res.candidates:
                    hints[tok] = list(res.candidates)
                    entry["hints"] = list(res.candidates)
            elif res.action == "context":
                context.append(tok)
            words.append(entry)

        llm_results = {}
        if use_llm and (deferred or context):
            llm_results = _live_layer3(cfg, sentence, deferred, context, hints, personal)
        for w in words:
            if w.get("kind") in ("defer", "context") and w["text"] in llm_results:
                w["intended"] = llm_results[w["text"]]
                w["layer"] = "llm"
        return jsonify({"words": words, "deferred": deferred,
                        "context": context, "llm": llm_results})

    return app


def _live_layer3(cfg, sentence, deferred, context, hints, personal) -> dict:
    """Synchronous Layer 3 for the playground: same prompts and guards as the
    real worker (mangle recovery with candidate hints + guarded homophone
    check), but with no memory side effects. Returns {token: recovered_or_kept}."""
    from mangle.context_llm import (FILL_IN_BLANK_PROMPT, mark_word, clean_output,
                                    overcorrection_guard)
    from mangle.homophones import group_of
    from app.config import options_for
    out = {}
    model = cfg["active_model"]
    opts = options_for(cfg, model)

    def ask(system, token):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": mark_word(sentence, token)}],
            "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": opts.get("num_predict", 16)},
            "keep_alive": "2h",
        }
        r = requests.post(f"{cfg['ollama_url']}/api/chat", json=payload,
                          timeout=opts.get("timeout_s", 15.0))
        if r.status_code == 400:
            payload.pop("think", None)
            r = requests.post(f"{cfg['ollama_url']}/api/chat", json=payload,
                              timeout=opts.get("timeout_s", 15.0))
        r.raise_for_status()
        return clean_output(r.json().get("message", {}).get("content", ""))

    try:
        for token in deferred:
            core = token.strip("'\"-")
            if not core or personal.contains(core.lower()):
                continue
            system = FILL_IN_BLANK_PROMPT
            if hints.get(token):
                system += ("\nThe intended word is most likely one of these keyboard-"
                           "close options: " + ", ".join(hints[token]) + ". Prefer one "
                           "of them if it fits the sentence.")
            word = ask(system, token)
            out[token] = (word if word and word.lower() != token.lower()
                          and not overcorrection_guard(token, word) else token)
        for token in context:
            group = group_of(token)
            if not group:
                continue
            options = ", ".join(sorted(group))
            system = ("You pick the correct word for a sentence. The bracketed word "
                      f"must be one of: {options}. Reply with exactly one of those and "
                      "nothing else.")
            word = ask(system, token)
            out[token] = word.lower() if word.lower() in group else token
    except requests.RequestException as e:
        out["_error"] = f"LLM offline: {type(e).__name__}"
    return out


def _free_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def start_in_thread(cfg, memory, personal, engine):
    """Start the dashboard in a daemon thread. Returns the URL to open, or None."""
    from app.config import DB_PATH
    host = "127.0.0.1"
    hostname = cfg.get("dashboard_hostname", "grammer.local")
    preferred = int(cfg.get("dashboard_port", 80))
    port = preferred if _free_port(host, preferred) else 8080
    if port != preferred:
        log.warning("port %d busy, using %d", preferred, port)

    app = create_app(cfg, memory, personal, engine, DB_PATH)

    def run():
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

    threading.Thread(target=run, daemon=True, name="dashboard").start()
    suffix = "" if port == 80 else f":{port}"
    url = f"http://{hostname}{suffix}"
    log.info("dashboard at %s  (or http://127.0.0.1%s)", url, suffix)
    return url

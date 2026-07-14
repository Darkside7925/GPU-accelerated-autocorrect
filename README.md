# GPU-accelerated autocorrect

A system-wide Windows autocorrect that recovers your intended word from a heavy mangle, the kind of wreck a fast typist actually produces (`yhoguhr` -> `thought`, `ijnsftead` -> `instead`, `tgjionk` -> `think`), that normal edit-distance spellcheckers cap out at 2-3 edits and give up on. It runs a four-layer pipeline where the cheapest layer wins first: a personal typo memory, a keyboard-and-phonetic matcher, and a local LLM on your GPU only as a last resort. The thing that personalizes is your accumulated data, not the model, so it gets faster and more accurate the more you use it.

Built for a ~143 WPM typist whose typos are not standard. Built benchmark-first: the model layer was gated on measured evidence before a single line of it was wired in.

> **What it does:** instant sub-millisecond correction on the keyboard hot path (no model ever touches a keystroke) - a learned `mangled -> intended` table that grows every day - a deterministic keyboard-adjacency + phonetic matcher for first-time mangles - a tightly-scoped fill-in-the-blank LLM (local, via Ollama) that only ever sees words the deterministic layers could not resolve - an end-of-day learning loop that promotes recurring mangles and grows a never-touch whitelist - a full local dashboard that exposes every bit of it, including a live pipeline playground and the typing analytics it has learned about you.

> **Honest status:** this is a personal tool that works, not a shipped product. The global keyboard hook, the four-layer correction pipeline, the learning loop, and the dashboard are all real and tested. The correction behavior depends on a low-level Windows keyboard hook and SendInput injection, which is inherently app-specific: it works in normal editable text fields (editors, chat boxes, browsers) and deliberately does nothing in places it cannot reason about safely. Passwords, terminals with their own line editing, and elevated windows are either skipped or unreliable by design. Layer 3 needs Ollama running with a model pulled. Treat it as a fast-moving personal build with real bugs still surfacing, not a daily driver for everyone.

---

## Quick Start

```powershell
# 1. dependencies
pip install requests symspellpy pynput pystray pillow jellyfish rapidfuzz flask

# 2. a local model for Layer 3 (any Ollama model works, this one won the benchmark)
ollama pull gemma4:e4b

# 3. copy the example config and run
copy config.example.json config.json
python -m app.main          # console + logs (good for the first run)
pythonw -m app.main         # background, no console window
```

- **Ctrl+Alt+A** toggles correction on and off. The tray menu does the same, plus model hot-swap, run-benchmark, open-dashboard, and run-compaction.
- **Backspace right after a correction** to undo it and get your word back. It is not disabled on a single backspace, that would be too trigger-happy. Only when the same correction gets rejected repeatedly (`reject_threshold`, default 3 times) does the engine conclude the fix is genuinely unwanted and stop making it. It learns from the pattern, not one keystroke. And once you backspace a fix, that exact correction will not re-fire for the rest of the current sentence, so deleting back to keep what you typed does not immediately trigger the same change again.
- **Updates** are checked against GitHub on startup. If a newer version is published, the tray shows "Update available" and the dashboard shows a banner. The check compares this build's version to `app/version.py` on the repo's main branch; it never auto-installs.

Then open the dashboard. One-time setup (Administrator, once) maps a friendly name to the loopback address:

```powershell
python -m dashboard.setup_hosts   # adds "127.0.0.1 grammer.local" to the hosts file
```

Now `http://grammer.local` is your dashboard. Without the setup it still works at `http://127.0.0.1` (or `:8080` if port 80 is taken).

> **Why grammer.local and not grammer.com:** the obvious idea is to hijack `grammer.com` in the hosts file. It does not work: `grammer.com` is a real HTTPS site, browsers cache its HSTS policy and force `https://`, and they will not even let you click through a certificate warning to a plain-http local server. A reserved `.local` name is never HSTS-preloaded, so plain http just works. This is the kind of thing the benchmark-first habit catches early.

---

## The benchmark (this is the point)

Prior work established three rules the hard way, so this project does not relitigate them: isolated-word LLM correction fails (no context, `ijnsftead` becomes `industrial`), generative models overcorrect (they rewrite text that was already fine), and the model is not the thing that should personalize. Passthrough, leaving correct text untouched, is the top priority. A wrong fix is worse than a missed one.

So before building the LLM layer, a feasibility benchmark measured whether a local model can actually recover heavily-mangled typos from sentence context without wrecking correct text. The harness (`mangle/benchmark_recovery.py`) fires 40 fixed sentences (`mangle/mangled_phrases.py`), each with one marked token, in three categories: heavy mash, transpositions, and a passthrough set of intentional non-dictionary tokens (names, slang, brands like `rizz`, `kubernetes`, `SymSpell`). It reports pure GPU inference time from Ollama's `eval_duration`, tokens/sec, recovery accuracy, and passthrough rate. The gate: **passthrough >= 95% AND meaningful recovery.**

Two findings drove the final design:

1. **A naive prompt makes even strong models overcorrect.** On the first run, no model cleared the passthrough gate: the capable ones kept "helpfully" rewriting intentional words (`rizz` -> `charisma`, `SymSpell` -> `Spell`). That failure is exactly what the gate exists to catch, and it is why the benchmark was built first.
2. **A conservative prompt plus a deterministic guard fixes it.** The prompt now forbids synonym and slang swaps, and a guard (`mangle/context_llm.py:overcorrection_guard`) refuses to let Layer 3 change any capitalized or CamelCase token, or make a large length change. Real typos of ordinary words are lowercase and similar-length, so the guard lifts passthrough to 100% without costing recovery.

### Results (RTX 5070 Ti, deployed prompt + guard, gate = passthrough >= 95% and recovery >= 50%)

| model | recovery | passthrough | discipline | GPU eval p50 | wall p50 | gate |
|---|---|---|---|---|---|---|
| **gemma4:e4b** (chosen) | **100%** | 100% | 100% | 19 ms | 460 ms | PASS |
| gemma4:e2b | 93.3% | 100% | 100% | 15 ms | 412 ms | PASS |
| gemma2:2b | 83.3% | 100% | 100% | 22 ms | 228 ms | PASS |
| gemma3n:e2b | 80.0% | 100% | 100% | 26 ms | 458 ms | PASS |
| gemma3:1b | 30.0% | 100% | 98% | 16 ms | fail (recovery) |
| qwen3:0.6b | 16.7% | 100% | 100% | 11 ms | fail (recovery) |

gemma4:e4b is the chosen model: perfect recovery on the set and 100% homophone flip (see `mangle/benchmark_homophones.py`), at ~460 ms async wall time. gemma4:e2b (93.3% recovery, 80% homophone flip) is the lighter tray-swap option. Wall time looks high but Layer 3 is asynchronous and idle-gated: it lands after you pause typing and never blocks a keystroke. Also worth knowing: a bundled ~58k-pair codespell dictionary (`mangle/misspellings.txt`) is seeded into Layer 1 on first run, so the overwhelming majority of everyday misspellings are corrected instantly with no model at all. Re-run any benchmark from the dashboard's Model tab, or:

```powershell
python -m mangle.benchmark_recovery gemma4:e2b gemma3n:e2b gemma2:2b gemma3:1b qwen3:0.6b --json mangle/recovery_results.json -v
```

The deterministic hot path was measured too: SymSpell-backed Layer 2 lookups run at p50 0.009 ms / p99 0.05 ms, four orders of magnitude inside the budget, which is why the whole thing is Python and no C++ hook was needed.

---

## How it works

Cheapest layer first, passthrough always preferred. Per keystroke, nothing happens but buffering. On a word boundary (space, comma, sentence end) the finished word is routed:

- **Layer 1, personal typo memory** (`mangle/typo_memory.py`). A SQLite table of your `mangled -> intended` words with a confidence score. A known mangle is corrected instantly and deterministically, no model. This is the primary mechanism and it is the one that grows.
- **Layer 2, keyboard and phonetic matcher** (`mangle/matcher.py`). For a first-time mangle. It does not use plain edit distance (which fails on mash). It unions candidates from metaphone/nysiis phonetic buckets, a SymSpell distance-3 lookup, and a first-and-last-letter length bucket, then scores each with a keyboard-weighted Damerau-Levenshtein (substitutions between physically close QWERTY keys are cheap; transpositions are cheap because fast typists swap adjacent keys) blended with word frequency. An ambiguity gate collapses confidence when the top two candidates are near-ties (`teh` -> `the` vs `tee`), so those defer to Layer 3 rather than auto-applying a guess. Median 1.4 ms.
- **Layer 3, context recovery** (`mangle/context_llm.py`). Only tokens Layers 1 and 2 could not resolve reach it. It shows the model the sentence with just that token marked and asks for the single word it was meant to be. Valid words never arrive here, so it is non-destructive by construction. Every recovery it makes is written back into Layer 1, so the next time that mangle appears it is instant. This is the mechanism by which the LLM fires less over time.
- **Homophone context check** (`mangle/homophones.py`, guarded). A valid word that belongs to a curated confusion group (to/too/two, their/there/they're, its/it's, then/than, ...) is not blindly passed through: at idle it gets a group-CONSTRAINED LLM check whose answer is accepted only if it is another member of the same group. The worst case is a same-group swap, never a rewrite, so passthrough safety is preserved. These fixes are not written to Layer 1 (they are per-sentence, not a stable mapping). Toggle with `context_homophones`.

**LLM-only mode** (`llm_only`, off by default, toggle in the dashboard Settings). When on, the deterministic Layers 1 and 2 stop auto-applying and every word that needs correcting is sent straight to the context LLM, which decides using the whole sentence. Valid words still pass through untouched and Layer 2 still supplies its keyboard-close candidates to the model as hints; it just never fixes a word on its own. Slower and heavier on the GPU, but every call is made by the model with context rather than a heuristic. Learned results are still cached into Layer 1, so a word you have corrected once this way is instant next time.

### Transactional injection (fast typing cannot scramble a correction)

The hook is observe-only, so your keystrokes reach the screen a beat before the correction worker sees them (pynput delivery lag). Applying a backspace-and-retype against that stale view is what used to scramble fast typing (`holland` -> `Hollaond`, a correction landing on the next word). A correction is now a short transaction:

- An in-flight counter (`_hook_seen` incremented in the hook, `_processed` in the worker) proves nothing is still queued in pynput before we touch the screen.
- During the few-millisecond injection the hook HOLDS plain printable keys (`suppress_event`) and replays them in order immediately after, so your keys physically cannot interleave with the correction. Our own injected events are tagged in `dwExtraInfo` so the hook always knows them.
- Anything unholdable (a Ctrl/Alt/Win chord, backspace, navigation, a mouse click, a dead key, or a 250 ms watchdog) aborts the transaction instead of holding, so Alt-Tab and every system behavior stay untouched. An aborted correction just retries at the next pause.

Toggle with `transactional_sync`.

The learning loop closes it: every sentence and every accepted or rejected correction is logged locally. An end-of-day pass (`mangle/compact.py`) promotes recurring mangles into Layer 1 and grows the never-touch whitelist from words you type often and never fix. Reject-to-learn is the interactive path: each time you backspace a correction it counts as a rejection, and once a correction has been rejected `reject_threshold` times the engine stops making it (whitelists the word and drops the mapping). A single backspace is only an undo, so one stray keystroke never disables a word.

### Wrong-space fixes (join and split)

- **Join** (`inc rease` -> `increase`) is deterministic and instant: if a finished word is not real on its own but merging it with the word right before it makes a real word, the two are merged and the stray space removed. It even handles a mistyped second half (`inc erease` -> `increase`) by fuzzy-correcting the merge, but only when the fix still starts with the prefix you typed correctly, which keeps it anchored and safe. Two genuinely separate words are never merged.
- **Split** (`itsthe` -> `its the`, `alot` -> `a lot`) is handled by the context layer, not a heuristic. A deterministic splitter cannot tell a real word that is simply missing from its dictionary (`dueling`, `starlink`) from a genuine missing space, so it mangles valid words. Instead an unresolved run-on is handed to the LLM, which uses the sentence to decide. A split is only ever applied when the model returns exactly two real words whose letters, with the space removed, still spell what you typed, so it can insert a space but never change a letter. Real words, names, and brands are left alone.

Passthrough is defended four ways, on purpose: valid dictionary words never reach Layer 3 (structural), the whitelist skips known personal terms, the overcorrection guard blocks the rewrite of names and brands, and repeated backspacing eventually protects anything that still slips through.

---

## The dashboard

Local, offline, and for your eyes only. Flask in a daemon thread on the loopback interface, reading through its own read-only SQLite connection so it never contends with the engine's writer thread. Charts are hand-rolled inline SVG, so there is no CDN and no vendored library: it runs with no internet.

| Section | What it shows |
|---|---|
| Overview | corrections today and all-time, a gauge of the share handled without the LLM (the getting-faster-over-time metric), and where corrections are handled by layer |
| Playground | type a mangled sentence and watch the real pipeline route each word live: green Layer 1, blue Layer 2, purple Layer 3, amber deferred, with the final corrected sentence |
| Typo memory | the full learned table, searchable and sortable, add and delete mappings, export and import your data as JSON |
| Whitelist | your never-touch vocabulary, add and remove |
| Typing insights | a keyboard heat map of the keys you fumble most, your most frequent mangles, the Layer 1 confidence distribution, typo length profile, hour-of-day error pattern, and corrections over time by layer |
| Learning loop | last compaction, pairs promoted, whitelist growth, and the without-LLM share trending over time; run compaction on demand |
| Model and benchmark | hot-swap the Layer 3 model, run the recovery benchmark, and read the last results |
| Settings | tune the layer confidence gates, the idle and drift windows, and the model, saved live to config.json |
| Raw log | every sentence you typed and every correction, accepted or rejected |
| Health | engine state, Layer 3 queue depth, last error |

---

## Layout

```
app/
  engine.py          observe-only WH_KEYBOARD_LL hook (Alt-Tab safe), the layered
                     router, transactional injection, join/split, reject-to-learn
  personal_dict.py   never-touch whitelist + correction + rejection log (SQLite)
  updater.py         checks GitHub for a newer app/version.py
  version.py         the app version (bump + push to notify existing installs)
  config.py          config load/save, layer gates, dashboard settings
  tray.py            tray icon, model hot-swap, dashboard and compaction actions
  main.py            wiring and entry point
mangle/
  typo_memory.py     Layer 1: personal typo memory + raw log (SQLite)
  matcher.py         Layer 2: phonetic buckets + keyboard-weighted Damerau
  pipeline.py        the pass / apply / defer routing decision
  context_llm.py     Layer 3: fill-in-the-blank recovery + the overcorrection guard
  compact.py         the end-of-day learning loop
  mangled_phrases.py the 40-case feasibility test set
  benchmark_recovery.py  the Phase 0 gate benchmark (reuses benchmark.py timing)
dashboard/
  app.py             Flask app (loopback only), data + edit endpoints
  insights.py        derived typing analytics (pure functions)
  static/            index.html, style.css, app.js (offline SVG charts, playground)
  setup_hosts.py     one-time grammer.local mapping (run as admin)
benchmark.py         the original Ollama timing harness (eval_duration), reused above
```

---

## Requirements and limitations

- **Windows.** The keyboard hook is Windows-specific (WH_KEYBOARD_LL via pynput, SendInput for injection).
- **Ollama** running locally for Layer 3, with a model pulled. Layers 1 and 2 work without it; the app degrades gracefully when the model is offline.
- **A GPU helps.** The benchmark numbers are from an RTX 5070 Ti. Any Ollama-capable machine works; latency scales with the model and hardware.
- **It will not correct everywhere.** Password fields, terminals with their own line discipline, and elevated windows are skipped or unreliable by design. This is a safety choice, not a bug: when the tool cannot be sure what is on screen at the cursor, it does nothing.
- **It is personal.** Your typo memory, whitelist, and logs live in a local SQLite file that is gitignored and never leaves your machine.

Not production-ready, not bug-free, and not trying to be. It is an evidence-first personal build where the hard parts already work and the remaining work is debugging and finishing.

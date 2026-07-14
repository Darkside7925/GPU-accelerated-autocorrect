"""Keyboard engine: global hook, keystroke buffering, correction injection.

Safety rules that make this coexist with Windows instead of fighting it:

  * The WH_KEYBOARD_LL hook (via pynput) NEVER suppresses events. Every key,
    including Alt-Tab / Win / Ctrl combos, passes straight through to the OS.
    We only observe. This is what guarantees system shortcuts keep working.
  * The hook callback does almost nothing: it tags the event and drops it on
    a queue. All real work (SymSpell, SQLite, SendInput) happens on a worker
    thread, so the hook can never stall the input pipeline.
  * Our own SendInput events carry LLKHF_INJECTED, and the hook ignores
    injected events, so corrections don't re-enter the pipeline.
  * Any modifier chord (Ctrl/Alt/Win + key), navigation key, or mouse click
    invalidates the buffers - if we can't be sure what's on screen at the
    cursor, we do nothing rather than guess.
  * Stage-2 (LLM) corrections are only applied while the user is idle and
    only if they haven't typed far past the sentence; otherwise dropped.
  * Enter/Tab never trigger injection: in chat apps Enter has already sent
    the text, and backspacing into a sent message would corrupt the next one.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time

from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode

from app import winject

log = logging.getLogger("engine")

# set SUMIZOME_DEBUG=1 to log every key the hook receives and every word decision
DEBUG = bool(os.environ.get("SUMIZOME_DEBUG"))

LLKHF_INJECTED = 0x10

WORD_TRIGGERS = set(" ,;:")          # finish current word, stage-1 correct
SENTENCE_TERMINATORS = set(".!?")    # also fire the sentence at the LLM
WORD_CHARS_EXTRA = set("'-")         # part of a word besides letters

_MODIFIERS = {
    Key.ctrl, Key.ctrl_l, Key.ctrl_r,
    Key.alt, Key.alt_l, Key.alt_r, Key.alt_gr,
    Key.cmd, Key.cmd_l, Key.cmd_r,
}
_NAV_KEYS = {
    Key.left, Key.right, Key.up, Key.down, Key.home, Key.end,
    Key.page_up, Key.page_down, Key.delete, Key.esc, Key.tab,
}

# virtual keys a transaction may hold: letters, digits, space, main punctuation,
# numpad. Everything else (backspace, enter, nav, F-keys) aborts instead.
_HOLDABLE_VKS = (
    set(range(0x41, 0x5B)) | set(range(0x30, 0x3A)) | {0x20}
    | set(range(0xBA, 0xC1)) | set(range(0xDB, 0xDF))
    | set(range(0x60, 0x70))
)


def _user32():
    try:
        import ctypes
        return ctypes.windll.user32
    except Exception:
        return None


class AutocorrectEngine:
    def __init__(self, cfg, pipeline, personal, layer3=None, on_toggle=None):
        self.cfg = cfg
        self.pipeline = pipeline            # RecoveryPipeline (Layers 1 + 2)
        self.memory = pipeline.memory       # Layer 1 store (for undo demote, raw log)
        self.personal = personal
        self.layer3 = layer3                # context LLM worker (Layer 3), may be None
        self.on_toggle = on_toggle          # tray callback (icon refresh)

        self.enabled = cfg.get("enabled", True)
        self._events: queue.Queue = queue.Queue()
        self._injector = keyboard.Controller()

        # --- text model (worker thread only) ---
        # Two strings for the current segment (since the last reset):
        #   _synced  = what is physically on screen right now (every key the user
        #              typed, plus whatever we have injected)
        #   _tail    = what SHOULD be on screen (the model, with corrections)
        # They diverge only where a correction has been decided but not yet
        # rendered. _sync() reconciles them with ONE atomic injection, and only
        # ever runs when the worker has drained the event queue (the user has
        # paused), so the screen we edit is the screen we think it is. This is
        # what stops fast typing from being scrambled or skipped.
        self._synced = ""
        self._tail = ""
        self._word = ""        # trailing word currently being typed
        self._gen = 0          # bumped whenever the model may no longer match screen
        self._last_key_ts = 0.0

        # undo-to-learn state (set when a correction is rendered)
        self._last_corr = None
        # words/phrases the user rejected in THIS segment: never re-correct them
        # here, so backspacing a fix does not immediately trigger it again
        self._suppressed = set()

        # tokens deferred to Layer 3 within the current segment, plus valid
        # homophones flagged for the guarded context check (last 2, deduped)
        self._deferred = []
        self._cand_hints = {}   # token -> Layer 2 candidate hints for the model
        self._context_checks = []
        self._job_id = 0
        self._jobs = {}         # job_id -> {"gen","deferred"}

        # hook-thread scratch
        self._hook_injected = False
        self._mods_down = set()

        # --- transactional sync state ---
        # _hook_seen counts real keydowns at HOOK time (hook thread is the only
        # writer); _processed counts them at worker-dispatch time (worker is the
        # only writer). Equal counters mean nothing is in flight in pynput's
        # hook -> message-loop -> queue pipeline, which is the blind spot that
        # settle-only timing could not see.
        self._hook_seen = 0
        self._processed = 0
        self._txn = False          # filter holds printable keys while True
        self._txn_abort = False    # something unholdable arrived; skip injecting
        self._txn_started = 0.0
        self._held = []            # chars suppressed during the txn, in order

        self._kb_listener = None
        self._mouse_listener = None
        self._worker = threading.Thread(target=self._run, daemon=True, name="engine")

    # ------------------------------------------------------------ lifecycle

    def start(self):
        self._worker.start()
        self._kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            win32_event_filter=self._win32_filter,
        )
        self._kb_listener.start()
        self._mouse_listener = mouse.Listener(on_click=self._on_click)
        self._mouse_listener.start()
        log.info("engine started (enabled=%s)", self.enabled)

    def stop(self):
        if self._kb_listener:
            self._kb_listener.stop()
        if self._mouse_listener:
            self._mouse_listener.stop()

    def toggle(self):
        self.enabled = not self.enabled
        self._events.put(("reset", None, time.monotonic()))
        log.info("autocorrect %s", "ENABLED" if self.enabled else "DISABLED")
        if self.on_toggle:
            self.on_toggle(self.enabled)

    # ------------------------------------------------------- hook callbacks
    # These run on pynput's hook/message threads: keep them tiny and never
    # raise (an exception would kill the listener).

    def _win32_filter(self, msg, data):
        # Runs synchronously on the hook thread for EVERY keyboard event, before
        # the OS delivers it anywhere. Three jobs:
        #   1. count real keydowns (_hook_seen) so the worker can tell when it
        #      has truly seen everything the user typed (pipeline-lag detector);
        #   2. during a correction transaction, HOLD plain printable keydowns
        #      (suppress + remember) so the screen cannot change under the
        #      correction, and replay them right after;
        #   3. never touch injected events, modifiers, or anything that will not
        #      resolve to exactly one printable char (those abort the txn
        #      instead), so Alt-Tab and system behavior stay untouched.
        # Identification uses the per-event injected flag and our dwExtraInfo
        # magic, both read HERE at event time (no deferred-dispatch race).
        if data.flags & LLKHF_INJECTED:
            return True                          # ours or another tool's; pass
        if msg not in (0x100, 0x104):            # WM_KEYDOWN / WM_SYSKEYDOWN only
            return True                          # keyups etc.: never held
        if self._txn:
            try:
                held = self._txn_consider(data.vkCode, data.scanCode)
            except Exception:
                held = False
                log.exception("hook filter")
            if held:
                # suppressed: no on_press will fire and the worker will enqueue
                # a synthesized copy itself, so do NOT count this keydown
                self.suppress_event_hook()       # raises; nothing below runs
        # every keydown that PASSES gets counted; its on_press enqueues exactly
        # one worker event, so _hook_seen == _processed means nothing in flight
        self._hook_seen += 1
        return True

    def _txn_consider(self, vk, scan) -> bool:
        """During a transaction, decide whether this real keydown can be held
        (True) or must abort the transaction and pass through (False)."""
        if time.monotonic() - self._txn_started > 0.25:
            self._txn_abort = True               # watchdog: never hold for long
            return False
        u32 = _user32()
        if u32 is None or any(u32.GetAsyncKeyState(m) & 0x8000
                              for m in (0x11, 0x12, 0x5B, 0x5C)):
            self._txn_abort = True               # Ctrl/Alt/Win chord: hands off
            return False
        if vk not in _HOLDABLE_VKS:
            self._txn_abort = True               # backspace/nav/enter/F-keys...
            return False
        ch = winject.vk_to_char(vk, scan)
        if ch is None:
            self._txn_abort = True               # dead key / odd layout state
            return False
        self._held.append(ch)
        return True

    def suppress_event_hook(self):
        """Indirection so tests can stub suppression; in production this raises
        pynput's SuppressException via the listener."""
        self._kb_listener.suppress_event()

    def _on_press(self, key, injected=False):
        try:
            # `injected` is pynput's per-event flag for THIS key (True when the
            # event carries LLKHF_INJECTED / came from SendInput, e.g. our own
            # _inject()). Ignoring our own synthetic keys here is what prevents a
            # correction from re-entering the pipeline.
            if injected:
                return
            if key in _MODIFIERS:
                self._mods_down.add(key)
                self._events.put(("chord", None, time.monotonic()))
                return
            # hotkey: Ctrl+Alt+A toggles on/off. Still enqueue a chord event so
            # the _hook_seen / _processed counters stay 1:1 per real keydown.
            mods = self._mods_down
            ctrl = any(k in mods for k in (Key.ctrl, Key.ctrl_l, Key.ctrl_r))
            alt = any(k in mods for k in (Key.alt, Key.alt_l, Key.alt_r))
            if ctrl and alt and isinstance(key, KeyCode) and key.vk == 0x41:  # 'A'
                self.toggle()
                self._events.put(("chord", None, time.monotonic()))
                return
            if ctrl or alt or any(k in mods for k in (Key.cmd, Key.cmd_l, Key.cmd_r)):
                self._events.put(("chord", None, time.monotonic()))
                return
            self._events.put(("key", key, time.monotonic()))
        except Exception:
            log.exception("hook on_press")

    def _on_release(self, key, injected=False):
        try:
            if injected:
                return
            self._mods_down.discard(key)
        except Exception:
            pass

    def _on_click(self, x, y, button, pressed, injected=False):
        if pressed and not injected:
            if self._txn:
                self._txn_abort = True   # caret may move; do not inject
            self._events.put(("click", None, time.monotonic()))

    # ---------------------------------------------------------- LLM callback

    def stage2_result(self, job_id, recoveries):
        """Called from the Layer 3 worker thread with {typed_token: recovered}."""
        self._events.put(("llm", (job_id, recoveries), time.monotonic()))

    # -------------------------------------------------------------- worker

    def _run(self):
        while True:
            try:
                try:
                    kind, payload, ts = self._events.get(timeout=0.05)
                except queue.Empty:
                    # Reconcile only when (a) a short settle has passed AND (b)
                    # the hook-side counter matches the processed counter, i.e.
                    # no keystroke is still in flight inside pynput's pipeline.
                    # (b) is what the settle alone could never guarantee.
                    settle = self.cfg.get("sync_settle_ms", 120) / 1000
                    if time.monotonic() - self._last_key_ts >= settle:
                        self._sync()
                        self._maybe_fire_deferred()
                    continue
                self._dispatch(kind, payload, ts)
            except Exception:
                log.exception("engine worker")

    def _dispatch(self, kind, payload, ts):
        if kind == "key":
            self._processed += 1
            self._last_key_ts = ts
            if DEBUG:
                self._keys_seen = getattr(self, "_keys_seen", 0) + 1
                ch = getattr(payload, "char", None)
                log.info("DEBUG key #%d received: %r", self._keys_seen, ch or payload)
            self._handle_key(payload)
        elif kind == "skey":
            # a char the filter held during a transaction; the screen already
            # got it via replay, this is the model's copy (not counted: its
            # keydown was suppressed and never counted either)
            self._last_key_ts = ts
            if DEBUG:
                log.info("DEBUG held char replayed: %r", payload)
            self._handle_char(payload)
        elif kind == "chord":
            self._processed += 1
            self._last_key_ts = ts
            self._invalidate()
        elif kind == "click":
            self._last_key_ts = ts
            self._invalidate()
        elif kind == "llm":
            self._handle_llm_result(*payload)
        elif kind == "reset":
            self._invalidate()

    def _drain(self, max_events: int = 512):
        for _ in range(max_events):
            try:
                kind, payload, ts = self._events.get_nowait()
            except queue.Empty:
                return
            self._dispatch(kind, payload, ts)

    def _invalidate(self):
        # We can no longer trust that the screen matches our model (a click, a
        # navigation key, or a backspace past our segment). Forget it and start a
        # fresh segment; never inject based on a stale model.
        self._gen += 1
        self._synced = ""
        self._tail = ""
        self._word = ""
        self._deferred = []
        self._cand_hints = {}
        self._context_checks = []
        self._suppressed = set()
        self._last_corr = None

    # ------------------------------------------------------------ key logic

    def _handle_key(self, key):
        if not self.enabled:
            return

        if key == Key.backspace:
            if self._maybe_undo():
                return
            if self._synced and self._tail:
                # the physical backspace already removed the last on-screen char;
                # mirror it in both the screen model and the target model
                self._synced = self._synced[:-1]
                self._tail = self._tail[:-1]
                self._word = self._word[:-1] if self._word else ""
            else:
                self._invalidate()  # backspacing past our segment -> give up
            return

        if key in _NAV_KEYS or key == Key.enter:
            # Enter/Tab/nav: the cursor may have jumped; never inject on a model
            # we cannot trust. Just reset the segment.
            self._invalidate()
            return

        ch = key.char if isinstance(key, KeyCode) else (" " if key == Key.space else None)
        if ch is None or not ch.isprintable():
            self._invalidate()
            return
        self._handle_char(ch)

    def _handle_char(self, ch):
        """A printable character reached the screen (physically typed, or held
        during a transaction and replayed). Update both models and route
        finished words."""
        self._last_corr = None  # any new typing forfeits the undo window

        # ch is on screen (_synced) and, until a correction says otherwise, it
        # is what we want (_tail)
        self._synced += ch
        self._tail += ch

        if ch in WORD_TRIGGERS or ch in SENTENCE_TERMINATORS:
            word = self._word
            self._word = ""
            if word:
                self._correct_word(word, ch)
            if ch in SENTENCE_TERMINATORS:
                self._submit_sentence(ch)
            return

        if ch.isalpha() or ch in WORD_CHARS_EXTRA:
            self._word += ch
        else:
            self._word = ""  # digits/symbols end the word without correcting

    # -------------------------------------------------- layers 1 and 2 (hot path)

    def _try_join(self, word, trigger) -> bool:
        """A word wrongly split by a stray space (inc rease -> increase): if the
        finished word is not valid on its own but joining it to the immediately
        preceding word makes a real word, merge the two into one. The MODEL is
        rewritten; _sync renders it on the next pause."""
        if not self.cfg.get("join_split_words", True):
            return False
        core = word.strip("'-")
        if len(core) < 2 or self.pipeline.is_valid(core):
            return False
        before = self._tail[: -(len(word) + len(trigger))]   # text before this word
        m = re.search(r"(?<![A-Za-z'])([A-Za-z]{1,15}) $", before)
        if not m:
            return False                                     # need exactly one space
        prev = m.group(1)
        merged = prev + core
        # merge only when the CURRENT word is a fragment (already checked invalid
        # above). Two valid words are never merged because a valid current word
        # returns early above.
        if self.pipeline.matcher.is_word(merged):
            joined = merged                                  # inc + rease = increase
        else:
            # the second fragment may itself be mistyped: inc + erease =
            # incerease -> increase. Accept a fuzzy fix only when it still starts
            # with the prefix you typed correctly, which keeps it anchored and safe.
            # A 1-2 char fragment (vs, to, of) is too short to fuzzy-merge safely.
            if len(core) < 3:
                return False
            cand, conf = self.pipeline.matcher.match(merged, 0.0)
            if (cand and cand.lower() != merged.lower()
                    and cand.lower().startswith(prev.lower())
                    and conf >= self.cfg.get("join_fuzzy_confidence", 0.45)):
                joined = cand
            else:
                return False
        n = len(prev) + 1 + len(word) + len(trigger)         # prev + space + word + trigger
        self._tail = self._tail[:-n] + _match_case(prev, joined) + trigger
        if prev in self._deferred:
            self._deferred.remove(prev)
        self._cand_hints.pop(prev, None)
        log_id = self.personal.log_correction(prev + " " + word, joined, "join")
        self._last_corr = {"original": prev + " " + word, "corrected": joined,
                           "layer": "join", "trigger": trigger,
                           "ts": time.monotonic(), "log_id": log_id}
        log.info("join: %r + %r -> %r", prev, word, joined)
        return True

    def _correct_word(self, word, trigger):
        """Route a finished word through the deterministic layers. A Layer 1 or
        Layer 2 hit rewrites the MODEL only (_tail); the physical screen keeps
        the typo until _sync renders it on the next pause. An unresolved word is
        deferred to Layer 3. Valid words are left alone (passthrough)."""
        if word.lower() in self._suppressed:
            return   # you rejected correcting this here; leave it alone this segment
        if self._try_join(word, trigger):
            return
        res = self.pipeline.on_word(word)
        if DEBUG:
            log.info("DEBUG word=%r -> %s intended=%r conf=%.2f",
                     word, res.action, res.intended, res.confidence)
        if res.action == "defer":
            self._deferred.append(word)
            if res.candidates:
                self._cand_hints[word] = list(res.candidates)
            self._submit_deferred()   # fire Layer 3 now for max head start before Enter
            return
        if res.action == "context":
            # valid homophone (to/too, their/there): guarded LLM check at the
            # next idle fire; keep only the 2 most recent, deduped
            if word not in self._context_checks:
                self._context_checks = (self._context_checks + [word])[-2:]
            return
        if res.action != "apply":
            return
        corrected = res.intended
        # both models currently end with "word + trigger"; fix only the target
        self._tail = self._tail[: -(len(word) + len(trigger))] + corrected + trigger
        log_id = self.personal.log_correction(word, corrected, res.layer)
        self._last_corr = {
            "original": word, "corrected": corrected, "layer": res.layer,
            "trigger": trigger, "ts": time.monotonic(), "log_id": log_id,
        }
        log.info("%s: %r -> %r (%.2f)", res.layer, word, corrected, res.confidence)

    # ------------------------------------------------- render the model to screen

    def _sync(self):
        """Make the screen (_synced) match the model (_tail) with ONE atomic
        injection, executed as a short TRANSACTION: while it runs (a few ms),
        the hook holds plain printable keystrokes and we replay them right
        after, so the user's typing physically cannot interleave with the
        correction. Anything unholdable (chords, backspace, nav, mouse click)
        aborts the injection instead; we simply retry at the next idle."""
        if self._tail == self._synced:
            self._trim()
            return
        if not self.cfg.get("transactional_sync", True):
            self._render_diff()
            self._trim()
            return
        if self._hook_seen != self._processed:
            return  # keystrokes still in flight to us; retry next idle tick
        self._held = []
        self._txn_abort = False
        self._txn_started = time.monotonic()
        self._txn = True
        try:
            # catch anything that slipped in between the check and opening
            self._drain()
            if self._txn_abort or self._hook_seen != self._processed:
                if DEBUG:
                    log.info("DEBUG txn abort (pre-inject)")
                return
            if self._tail != self._synced:
                self._render_diff()
        finally:
            # replay held keys while STILL holding the door, so nothing can land
            # between the correction and the replay; loop because new keys may
            # arrive during the replay itself
            for _ in range(4):
                held, self._held = self._held, []
                if not held:
                    break
                self._replay(held)
            self._txn = False
            held, self._held = self._held, []
            if held:                      # boundary stragglers
                self._replay(held)
            self._trim()

    def _replay(self, held):
        chars = "".join(held)
        if DEBUG:
            log.info("DEBUG txn replaying %d held chars: %r", len(chars), chars)
        winject.send_unicode_chars(chars)             # to the screen, in order
        now = time.monotonic()
        for ch in chars:                              # to the model, in order
            self._events.put(("skey", ch, now))

    def _render_diff(self):
        """Compute and inject the minimal edit turning _synced into _tail."""
        i, n = 0, min(len(self._synced), len(self._tail))
        while i < n and self._synced[i] == self._tail[i]:
            i += 1
        backspaces = len(self._synced) - i
        # Never backspace far to fix an old word; that would erase everything
        # the user typed after it. Past the window, accept the screen as final.
        if backspaces > self.cfg.get("stage2_max_drift_chars", 100):
            self._synced = self._tail
            return
        self._inject(backspaces, self._tail[i:])
        self._synced = self._tail

    def _trim(self):
        """Keep the tracked segment small so corrections stay local and cheap."""
        if len(self._synced) > 160:
            self._synced = self._tail = self._tail[-80:]

    def _inject(self, backspaces, text):
        """One atomic Win32 injection: backspaces then text, uninterruptible by
        the user's keystrokes."""
        winject.send(backspaces, text)

    def _maybe_undo(self):
        lc = self._last_corr
        self._last_corr = None
        if not lc or time.monotonic() - lc["ts"] > self.cfg.get("undo_window_s", 4.0):
            return False
        # Only treat this backspace as an undo if the correction is actually on
        # screen (rendered, not since drifted): the screen shows
        # "<corrected><trigger>" and the user's backspace removes the trigger.
        rendered = lc["corrected"] + lc["trigger"]
        if self._synced != self._tail or not self._synced.endswith(rendered):
            return False
        base = self._synced[: -len(rendered)]
        self._synced = base + lc["corrected"]               # user deleted the trigger
        self._tail = base + lc["original"] + lc["trigger"]  # _sync restores original
        self.personal.mark_undone(lc["log_id"])
        # Never redo THIS correction in THIS segment: once you backspace it, the
        # word (and, for a join, each of its parts) is left alone until the
        # segment resets, so fixing it by hand does not trigger it again.
        self._suppressed.add(lc["original"].lower())
        for part in lc["original"].split():
            self._suppressed.add(part.lower())
        # Reject-to-LEARN, by pattern, not on a single backspace. Restoring the
        # word once is just an undo. Only when the same correction gets rejected
        # repeatedly does the engine conclude it is genuinely unwanted and stop
        # making it (whitelist the word and drop any learned mapping).
        n = self.personal.record_rejection(lc["original"], lc["corrected"])
        threshold = self.cfg.get("reject_threshold", 3)
        if n >= threshold:
            self.personal.add(lc["original"], source="rejected")
            self.memory.demote(lc["original"])
            self.personal.clear_rejection(lc["original"])
            log.info("stopped correcting %r -> %r after %d rejections",
                     lc["original"], lc["corrected"], n)
        else:
            log.info("undo: restored %r (rejection %d of %d before it stops)",
                     lc["original"], n, threshold)
        return True

    # ------------------------------------------- Layer 3 (context, async, gated)

    def _submit_sentence(self, term):
        if self._tail.strip():
            self.memory.log_raw(self._tail)     # raw text for the learning loop
        self._submit_deferred()

    def _maybe_fire_deferred(self):
        """Idle backstop: fire any words still waiting (the eager path in
        _correct_word already submits the moment a word defers). Context
        homophone checks only fire here, at idle, never eagerly: they are
        usually correct as typed and can wait for a pause."""
        self._submit_deferred(include_context=True)

    def _submit_deferred(self, include_context=False):
        """Queue the current segment plus its unresolved words (and, at idle,
        flagged homophones) to Layer 3. Only one request in flight at a time;
        when it finishes we pump the backlog (see _handle_llm_result)."""
        context = self._context_checks if include_context else []
        if (not (self._deferred or context) or self._jobs
                or not self.cfg.get("stage2_enabled", True) or self.layer3 is None):
            return
        deferred, self._deferred = self._deferred, []
        hints = {t: self._cand_hints.pop(t) for t in deferred if t in self._cand_hints}
        if include_context:
            self._context_checks = []
        self._job_id += 1
        self._jobs = {self._job_id: {"gen": self._gen, "deferred": list(deferred)}}
        self.layer3.submit(self._job_id, self._tail, list(deferred), list(context), hints)

    def _handle_llm_result(self, job_id, recoveries):
        """recoveries maps each typed token to its recovered word. Apply each
        into the MODEL only; _sync renders them on the next pause. The whole
        result is dropped if the segment changed since submission."""
        job = self._jobs.pop(job_id, None)
        if job is None or job["gen"] != self._gen:
            self._submit_deferred()   # segment gone, but pump anything new
            return
        for token, recovered in recoveries.items():
            if not recovered or recovered.lower() == token.lower():
                continue
            # replace the LAST whole-word occurrence: it is the one most
            # recently typed, which is the one that was flagged (matters for
            # frequent homophones like "to" appearing several times)
            pat = re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)")
            matches = list(pat.finditer(self._tail))
            if not matches:
                continue
            m = matches[-1]
            fixed = _match_case(token, recovered)
            self._tail = self._tail[: m.start()] + fixed + self._tail[m.end():]
            self.personal.log_correction(token, recovered, "llm")
            log.info("llm: %r -> %r", token, recovered)
        self._submit_deferred()       # fire words that deferred while we waited


def _match_case(src: str, target: str) -> str:
    """Transfer the casing shape of the typed token onto the recovered word."""
    if src.isupper() and len(src) > 1:
        return target.upper()
    if src[:1].isupper():
        return target[:1].upper() + target[1:]
    return target

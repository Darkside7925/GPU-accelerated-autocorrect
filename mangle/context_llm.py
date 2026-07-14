"""Layer 3: context recovery via a local Ollama model (async, tightly scoped).

Only tokens that Layers 1 and 2 could not resolve reach this worker. For each
such token it runs a fill-in-the-blank query: the sentence is shown with just
that token marked, and the model returns the single word it was meant to be.
Valid words never arrive here, so this layer is structurally non-destructive.

Every recovery is written back into Layer 1 (typo memory), so the next time the
same mangle appears it is corrected instantly with no model call. That is the
mechanism by which the system gets faster and needs the LLM less over time.

The corrected full sentence is handed back to the engine, which applies it with
the same idle-and-drift-gated rewrite used for all non-destructive corrections.
"""

from __future__ import annotations

import queue
import re
import threading

import requests

from app.config import options_for

# Canonical prompt and options: the Phase 0 benchmark imports these so it tests
# exactly what the app runs in production.
FILL_IN_BLANK_PROMPT = (
    "You correct ONE mistyped word using the meaning of the whole sentence. The "
    "user gives one sentence with exactly one word wrapped in double square "
    "brackets, like [[wrod]]. Read the ENTIRE sentence first and work out what "
    "the person is actually saying, then reply with ONLY the word the bracketed "
    "word was meant to be.\n"
    "Rules:\n"
    "- Context decides the answer. Choose the word that fits the sentence's "
    "meaning and grammar, not merely one that looks or sounds similar. If two "
    "corrections are equally plausible in spelling, pick the one the surrounding "
    "words call for.\n"
    "- NEVER change a name. If the bracketed word is, or could be, a person's "
    "name, username, handle, place, company, brand, or product, reply with it "
    "EXACTLY as typed. Treat a word that is capitalized, prefixed with '@', or "
    "introduced like a name (after 'Mr', 'Mrs', 'Ms', 'Dr', 'named', 'aka', or "
    "'name is') as a name and keep it unchanged, even if it looks misspelled.\n"
    "- Keep slang, abbreviations, code, file names, and technical terms EXACTLY "
    "unchanged.\n"
    "- Fix only an obvious misspelling of an ordinary English word. Never swap a "
    "word for a synonym or a different word that only means or sounds similar. "
    "Do not translate slang. Do not shorten or expand the word.\n"
    "- MISSING SPACE: if the bracketed word is clearly two ordinary words typed "
    "with no space between them (like 'itsthe' for 'its the', or 'alot' for 'a "
    "lot'), reply with the two words separated by ONE space. Do this only when "
    "both are real words and the letters are otherwise correct. NEVER split a "
    "real single word (dueling, running, awesome), a name, a brand (starlink), "
    "or a compound word. If in doubt, do not split.\n"
    "- If you are not certain it is a plain misspelling of a common word, reply "
    "with the bracketed word unchanged. When in doubt, do nothing.\n"
    "Reply with the intended word, or the two words if it was a missing space: "
    "no brackets, no quotes, no punctuation, no explanation."
)
FILL_OPTIONS = {"num_predict": 16, "temperature": 0}


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _word_re(token: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)")


def mark_word(sentence: str, token: str) -> str:
    """Wrap the LAST whole-word occurrence of token in [[ ]] for the prompt.
    The most recently typed occurrence is the flagged one, which matters for
    frequent homophones like "to"."""
    matches = list(_word_re(token).finditer(sentence))
    if not matches:
        return sentence
    m = matches[-1]
    return sentence[: m.start()] + f"[[{token}]]" + sentence[m.end():]


def replace_word(sentence: str, token: str, word: str) -> str:
    """Replace the first whole-word occurrence of token with word."""
    return _word_re(token).sub(word.replace("\\", r"\\"), sentence, count=1)


# words that, appearing right before a token, frame it as a name we must not
# touch (people type their own and friends' names, which are in no dictionary).
# High precision only: cues that overwhelmingly precede a name, not a normal word.
_NAME_CUES = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "prof", "professor", "sir", "madam", "mister",
    "named", "aka", "nicknamed", "surname",
})


def looks_like_name(sentence: str, token: str) -> bool:
    """True if the sentence frames the bracketed token as a name, so it must be
    kept exactly. Uses the LAST occurrence (the one mark_word flags) and looks at
    what immediately precedes it: an '@' handle prefix, a title (Mr/Dr/...), or a
    'name is X' / 'named X' introduction."""
    matches = list(_word_re(token).finditer(sentence))
    if not matches:
        return False
    start = matches[-1].start()
    if start > 0 and sentence[start - 1] == "@":
        return True
    before = re.findall(r"[A-Za-z']+", sentence[:start].lower())
    if not before:
        return False
    if before[-1] in _NAME_CUES:
        return True
    # "my name is X", "the name's X", "this is X" (self-introduction)
    if before[-1] == "is" and len(before) >= 2 and before[-2] in ("name", "this"):
        return True
    return False


def overcorrection_guard(token: str, candidate: str) -> bool:
    """Deterministic passthrough defense. Returns True if the model's candidate
    should be REJECTED (keep the original token). Real typos of ordinary words
    are lowercase and close in length, so we refuse:
      - any change to a token carrying an uppercase letter (names, brands,
        CamelCase technical terms like SymSpell or Sumizome), and
      - a large length change (e.g. slang rizz turned into charisma),
    which is where a capable model tends to overcorrect intentional words."""
    if candidate.lower() == token.lower():
        return False
    if any(c.isupper() for c in token):
        return True
    if abs(len(candidate) - len(token)) > max(3, int(0.6 * len(token))):
        return True
    return False


def clean_output(raw: str) -> str:
    out = strip_thinking(raw or "")
    out = out.strip().strip('"\'`[]').strip()
    out = out.splitlines()[0].strip() if out else ""
    return out.strip('.,!?;:"\'`()[]').strip()


class ContextLLMWorker:
    def __init__(self, cfg, personal, memory, on_result, is_word=None):
        """on_result(job_id, {token: recovered}), called from the worker thread.
        is_word(word) -> bool is used to reject a recovery that is not a real
        word, so the small model can never inject a non-word like 'wagster'."""
        self._cfg = cfg
        self._personal = personal
        self._memory = memory
        self._on_result = on_result
        self._is_word = is_word
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, daemon=True, name="layer3")
        self._thread.start()
        self.last_error: str | None = None

    def submit(self, job_id: int, sentence: str, deferred: list[str],
               context: list[str] | None = None, hints: dict | None = None) -> None:
        try:
            self._queue.put_nowait((job_id, sentence, deferred, context or [], hints or {}))
        except queue.Full:
            pass  # typing faster than the GPU can recover; drop rather than lag

    def _run(self) -> None:
        while True:
            job_id, sentence, deferred, context, hints = self._queue.get()
            recoveries: dict = {}
            try:
                recoveries = self._recover(sentence, deferred, hints)
                recoveries.update(self._check_homophones(sentence, context))
            except requests.RequestException as e:
                self.last_error = str(e)
            # ALWAYS answer, even with nothing to change or on error: the engine
            # keeps one job in flight and only frees the slot on a result, so a
            # silent drop here would block every future Layer 3 request
            self._on_result(job_id, recoveries)

    def _check_homophones(self, sentence: str, context: list[str]) -> dict:
        """Guarded context check for valid homophones: the model may only swap
        the token for another member of its confusion group ("to" -> "too"),
        never anything else, so a wrong answer is structurally bounded. These
        are NOT recorded into typo memory: to->too is per-sentence, not a
        stable mapping."""
        from mangle.homophones import group_of
        out: dict[str, str] = {}
        for token in context:
            group = group_of(token)
            if group is None or self._personal.contains(token.lower()):
                continue
            word = self._ask_homophone(sentence, token, group)
            if (word and word.lower() in group
                    and word.lower() != token.lower()):
                out[token] = word.lower()
        return out

    def _ask_homophone(self, sentence: str, token: str, group) -> str:
        """Constrained choice: which member of the confusion group belongs in
        the marked position? The model picks from the listed options only."""
        options = ", ".join(sorted(group))
        system = (
            "You pick the correct word for a sentence. The sentence has one "
            f"word in double square brackets. Choose which of these fits that "
            f"position: {options}. Reply with exactly one of those options and "
            "nothing else. If unsure, reply with the bracketed word unchanged."
        )
        model = self._cfg["active_model"]
        opts = options_for(self._cfg, model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": mark_word(sentence, token)},
            ],
            "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 8},
            "keep_alive": "2h",
        }
        url = f"{self._cfg['ollama_url']}/api/chat"
        r = requests.post(url, json=payload, timeout=opts.get("timeout_s", 12.0))
        if r.status_code == 400:
            payload.pop("think", None)
            r = requests.post(url, json=payload, timeout=opts.get("timeout_s", 12.0))
        r.raise_for_status()
        return clean_output(r.json().get("message", {}).get("content", ""))

    def _recover(self, sentence: str, deferred: list[str], hints: dict | None = None) -> dict:
        """Return {typed_token: recovered_word} for the tokens the model fixed.
        The engine applies these into its model; we do not rebuild the sentence
        here, so a token that the user has since edited away simply will not be
        found and applied."""
        hints = hints or {}
        out: dict[str, str] = {}
        for token in deferred:
            core = token.strip("'\"-")
            if not core or self._personal.contains(core.lower()):
                continue  # whitelisted in the meantime, leave it alone
            if looks_like_name(sentence, token):
                continue  # framed as a name (Mr X, @handle, "name is X"): keep it
            word = self._ask(sentence, token, hints.get(token))
            if not word:
                continue
            if re.search(r"\s", word):
                # the model split a run-on into words (itsthe -> its the). Accept
                # only a PURE split: exactly two real words whose letters, with
                # the space removed, still spell what was typed. That makes it
                # impossible for the model to invent or change letters while
                # splitting, so a wrong split can only ever be a wrong space.
                split = self._accept_split(core, word)
                if split:
                    out[token] = split
                    self._memory.record(core, split.lower(), source="llm")
                continue
            if word.lower() == token.lower():
                continue  # model kept it as-is: nothing to do
            if overcorrection_guard(token, word):
                continue  # looks like an overcorrection of an intentional word
            # reject a recovery that is not a real word (small models sometimes
            # return plausible-looking non-words like "wagster" or "podden");
            # allow contractions and hyphenates, which dictionaries often omit
            if (self._is_word is not None and not self._is_word(word)
                    and "'" not in word and "-" not in word):
                continue
            out[token] = word
            self._memory.record(core, word.lower(), source="llm")
        return out

    def _accept_split(self, core: str, answer: str) -> str | None:
        """Validate a Layer-3 split. Returns the 'w1 w2' string to apply, or None.
        Guarded hard: splitting must be enabled, the answer must be exactly two
        real words, and joining them back (no spaces) must reproduce the typed
        token letter for letter. So the model may only INSERT a space, never
        change letters, and both sides must be dictionary words."""
        if not self._cfg.get("join_split_words", True):
            return None
        parts = answer.split()
        if len(parts) != 2 or not all(p.isalpha() for p in parts):
            return None
        if "".join(parts).lower() != core.lower():
            return None
        if self._is_word is None or not all(self._is_word(p) for p in parts):
            return None
        return answer.lower()

    def _ask(self, sentence: str, token: str, hints=None) -> str:
        model = self._cfg["active_model"]
        opts = options_for(self._cfg, model)
        system = FILL_IN_BLANK_PROMPT
        if hints:
            # keyboard-plausible candidates from Layer 2: strong hints, but the
            # model may still override them if the sentence points elsewhere
            system += ("\nThe intended word is most likely one of these keyboard-"
                       "close options: " + ", ".join(hints) + ". Prefer one of "
                       "them if it fits the sentence.")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": mark_word(sentence, token)},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": FILL_OPTIONS["temperature"],
                "num_predict": opts.get("num_predict", FILL_OPTIONS["num_predict"]),
            },
            "keep_alive": "2h",   # stay warm between corrections
        }
        url = f"{self._cfg['ollama_url']}/api/chat"
        r = requests.post(url, json=payload, timeout=opts.get("timeout_s", 12.0))
        if r.status_code == 400:  # older Ollama without the "think" field
            payload.pop("think", None)
            r = requests.post(url, json=payload, timeout=opts.get("timeout_s", 12.0))
        r.raise_for_status()
        return clean_output(r.json().get("message", {}).get("content", ""))


def _match_case(src: str, target: str) -> str:
    if src.isupper() and len(src) > 1:
        return target.upper()
    if src[:1].isupper():
        return target[:1].upper() + target[1:]
    return target

"""Layer 1: personal typo memory (the core of the whole system).

A learned lookup table mapping mangled inputs to intended words. Every confirmed
correction is stored here; on a known mangle this returns the intended word
instantly and deterministically, with no model involved. As it grows it covers
more of the user's real mistakes and Layer 3 (the LLM) fires less.

Shares the app's SQLite file (personal.db) but owns its own tables, so it sits
alongside PersonalDict (whitelist + correction_log) without disturbing it. WAL
mode is enabled so the engine, the compaction job, and the dashboard can touch
the file concurrently without lock storms.

Confidence model: each pair carries a float in [0, 1]. A record() bumps it by a
source-dependent amount (a user-confirmed pair jumps immediately; an LLM-derived
pair climbs only as it recurs). lookup() gates on a minimum confidence so a
single unproven guess never becomes a deterministic auto-correction.
"""

from __future__ import annotations

import sqlite3
import threading
import time

# how much one record() call raises confidence, by where the pair came from
SOURCE_BUMP = {
    "user": 0.60,        # user accepted a correction, or retyped this word
    "undo_confirm": 0.60,
    "compaction": 0.30,  # promoted by the end-of-day batch
    "llm": 0.20,         # produced by Layer 3, not yet reinforced
    "review": 0.20,      # produced by the adaptive review pass, same policy
    "manual": 0.60,      # added by hand in the dashboard
}
DEFAULT_LOOKUP_CONFIDENCE = 0.50


class TypoMemory:
    def __init__(self, db_path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS typo_memory (
                   mangled TEXT NOT NULL,
                   intended TEXT NOT NULL,
                   count INTEGER NOT NULL DEFAULT 1,
                   first_seen REAL NOT NULL,
                   last_seen REAL NOT NULL,
                   confidence REAL NOT NULL DEFAULT 0.0,
                   source TEXT NOT NULL DEFAULT 'llm',
                   PRIMARY KEY (mangled, intended)
               )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS raw_log (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts REAL NOT NULL,
                   text TEXT NOT NULL
               )"""
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()
        # mangled(lower) -> (intended, confidence) for the current best pick
        self._cache: dict[str, tuple[str, float]] = {}
        self._reload_cache()

    # ------------------------------------------------------------- cache

    def _reload_cache(self) -> None:
        # Touches self._conn, so it must hold the lock. Callers (``__init__`` and
        # ``demote``) invoke this WITHOUT already holding the lock, so acquiring a
        # plain (non-reentrant) Lock here is safe.
        with self._lock:
            rows = self._conn.execute(
                "SELECT mangled, intended, confidence FROM typo_memory"
            ).fetchall()
        best: dict[str, tuple[str, float]] = {}
        for mangled, intended, conf in rows:
            key = mangled.lower()
            if key not in best or conf > best[key][1]:
                best[key] = (intended, conf)
        self._cache = best

    # ------------------------------------------------------------- lookup

    def lookup(self, mangled: str, min_confidence: float = DEFAULT_LOOKUP_CONFIDENCE):
        """Return the intended word for a known mangle, or None. Deterministic,
        O(1), no model. Casing of `mangled` is transferred onto the result."""
        hit = self._cache.get(mangled.lower())
        if not hit or hit[1] < min_confidence:
            return None
        return _match_case(mangled, hit[0])

    def confidence(self, mangled: str) -> float:
        hit = self._cache.get(mangled.lower())
        return hit[1] if hit else 0.0

    # ------------------------------------------------------------- writes

    def record(self, mangled: str, intended: str, source: str = "llm") -> None:
        """Reinforce (or create) a mangled -> intended mapping."""
        m = mangled.lower().strip()
        intended = intended.strip()
        if not m or not intended or m == intended.lower():
            return
        bump = SOURCE_BUMP.get(source, 0.20)
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT count, confidence FROM typo_memory WHERE mangled=? AND intended=?",
                (m, intended),
            ).fetchone()
            if row:
                count, conf = row[0] + 1, min(1.0, row[1] + bump)
                self._conn.execute(
                    "UPDATE typo_memory SET count=?, confidence=?, last_seen=?, source=? "
                    "WHERE mangled=? AND intended=?",
                    (count, conf, now, source, m, intended),
                )
            else:
                count, conf = 1, min(1.0, bump)
                self._conn.execute(
                    "INSERT INTO typo_memory "
                    "(mangled, intended, count, first_seen, last_seen, confidence, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (m, intended, count, now, now, conf, source),
                )
            self._conn.commit()
        cur = self._cache.get(m)
        if cur is None or conf >= cur[1]:
            self._cache[m] = (intended, conf)

    def bulk_seed(self, pairs, source: str = "dataset", confidence: float = 0.60) -> int:
        """Insert many mangled -> intended pairs in ONE transaction (fast enough
        for tens of thousands), skipping any mangle already present so a user's
        own edits and learned pairs always win. Returns how many were added."""
        now = time.time()
        added = 0
        with self._lock:
            cur = self._conn.cursor()
            for mangled, intended in pairs:
                m = mangled.lower().strip()
                intended = intended.strip()
                if not m or not intended or m == intended.lower() or m in self._cache:
                    continue
                cur.execute(
                    "INSERT OR IGNORE INTO typo_memory (mangled, intended, count, "
                    "first_seen, last_seen, confidence, source) VALUES (?,?,1,?,?,?,?)",
                    (m, intended, now, now, confidence, source),
                )
                if cur.rowcount:
                    self._cache[m] = (intended, confidence)
                    added += 1
            self._conn.commit()
        return added

    def demote(self, mangled: str, intended: str | None = None) -> None:
        """User rejected a correction: remove the offending pair(s) so we never
        auto-apply it again. Called from undo-to-learn."""
        m = mangled.lower().strip()
        with self._lock:
            if intended is None:
                self._conn.execute("DELETE FROM typo_memory WHERE mangled=?", (m,))
            else:
                self._conn.execute(
                    "DELETE FROM typo_memory WHERE mangled=? AND intended=?",
                    (m, intended.strip()),
                )
            self._conn.commit()
        self._reload_cache()

    def log_raw(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO raw_log (ts, text) VALUES (?, ?)", (time.time(), text)
            )
            self._conn.commit()

    # ------------------------------------------------------- reads (dashboard)

    def iter_raw(self, since_ts: float = 0.0):
        with self._lock:
            return self._conn.execute(
                "SELECT ts, text FROM raw_log WHERE ts >= ? ORDER BY ts", (since_ts,)
            ).fetchall()

    def all_pairs(self):
        with self._lock:
            return self._conn.execute(
                "SELECT mangled, intended, count, confidence, first_seen, last_seen, source "
                "FROM typo_memory ORDER BY count DESC, confidence DESC"
            ).fetchall()

    def get_meta(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            n_pairs = self._conn.execute(
                "SELECT COUNT(*) FROM typo_memory"
            ).fetchone()[0]
            n_raw = self._conn.execute("SELECT COUNT(*) FROM raw_log").fetchone()[0]
        n_mangles = len(self._cache)
        return {"pairs": n_pairs, "distinct_mangles": n_mangles, "raw_entries": n_raw}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _match_case(src: str, target: str) -> str:
    """Transfer the casing shape of the typed token onto the recovered word."""
    if src.isupper() and len(src) > 1:
        return target.upper()
    if src[:1].isupper():
        return target[:1].upper() + target[1:]
    return target

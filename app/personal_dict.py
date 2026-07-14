"""SQLite-backed personal dictionary: words that must never be autocorrected.

Words get here two ways:
  - manually (dashboard / direct DB insert)
  - automatically, once a correction has been REJECTED (backspaced away)
    enough times to look like a pattern, recorded with source='rejected'.
    A single backspace is only an undo; repeated rejection is what teaches it.
"""

from __future__ import annotations

import sqlite3
import threading
import time


class PersonalDict:
    def __init__(self, db_path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS personal_words (
                   word TEXT PRIMARY KEY,
                   added_at REAL NOT NULL,
                   source TEXT NOT NULL DEFAULT 'manual'
               )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS correction_log (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts REAL NOT NULL,
                   original TEXT NOT NULL,
                   corrected TEXT NOT NULL,
                   stage TEXT NOT NULL,
                   undone INTEGER NOT NULL DEFAULT 0
               )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS rejections (
                   word TEXT PRIMARY KEY,
                   corrected TEXT NOT NULL,
                   count INTEGER NOT NULL DEFAULT 0,
                   last_ts REAL NOT NULL
               )"""
        )
        self._conn.commit()
        self._cache = {
            row[0].lower()
            for row in self._conn.execute("SELECT word FROM personal_words")
        }

    def contains(self, word: str) -> bool:
        return word.lower() in self._cache

    def add(self, word: str, source: str = "manual") -> None:
        w = word.lower()
        if not w or w in self._cache:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO personal_words (word, added_at, source) VALUES (?, ?, ?)",
                (w, time.time(), source),
            )
            self._conn.commit()
            self._cache.add(w)

    def all_words(self) -> set[str]:
        return set(self._cache)

    def remove(self, word: str) -> None:
        w = word.lower()
        with self._lock:
            self._conn.execute("DELETE FROM personal_words WHERE word = ?", (w,))
            self._conn.commit()
        self._cache.discard(w)

    def log_correction(self, original: str, corrected: str, stage: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO correction_log (ts, original, corrected, stage) VALUES (?, ?, ?, ?)",
                (time.time(), original, corrected, stage),
            )
            self._conn.commit()
            return cur.lastrowid

    def mark_undone(self, log_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE correction_log SET undone = 1 WHERE id = ?", (log_id,))
            self._conn.commit()

    def record_rejection(self, word: str, corrected: str) -> int:
        """Count one rejection (a backspaced correction) for `word`; returns the
        new running count. The engine only stops correcting once this crosses a
        threshold, so a single accidental backspace never disables a word."""
        w = word.lower().strip()
        with self._lock:
            self._conn.execute(
                "INSERT INTO rejections (word, corrected, count, last_ts) "
                "VALUES (?, ?, 1, ?) ON CONFLICT(word) DO UPDATE SET "
                "count = count + 1, corrected = excluded.corrected, last_ts = excluded.last_ts",
                (w, corrected, time.time()),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT count FROM rejections WHERE word = ?", (w,)).fetchone()
        return row[0] if row else 1

    def rejection_count(self, word: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM rejections WHERE word = ?", (word.lower().strip(),)
            ).fetchone()
        return row[0] if row else 0

    def clear_rejection(self, word: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM rejections WHERE word = ?", (word.lower().strip(),))
            self._conn.commit()

    def rejection_stats(self):
        """(word, corrected, count) rows in progress, for the dashboard."""
        with self._lock:
            return self._conn.execute(
                "SELECT word, corrected, count FROM rejections ORDER BY count DESC"
            ).fetchall()

    def iter_corrections(self, since_ts: float = 0.0):
        """(ts, original, corrected, stage, undone) rows, for the learning loop
        and the dashboard."""
        with self._lock:
            return self._conn.execute(
                "SELECT ts, original, corrected, stage, undone FROM correction_log "
                "WHERE ts >= ? ORDER BY ts", (since_ts,)
            ).fetchall()

    def correction_counts(self):
        """(stage, undone, count) aggregates for the dashboard per-layer stats."""
        with self._lock:
            return self._conn.execute(
                "SELECT stage, undone, COUNT(*) FROM correction_log GROUP BY stage, undone"
            ).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

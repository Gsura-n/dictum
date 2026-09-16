"""Local dictation history and corrections (SQLite in ~/.dictum/history.db).

Nothing leaves the machine. History is what makes personalization possible:
each dictation stores the raw transcript and what was pasted; a correction is
the text the user says it should have been. Corrections are the training data
for `dictum personalize`.

Off by default for privacy; enable with history.enabled: true in config.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .adapters import HOME

DB = HOME / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dictations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  mode TEXT, app TEXT,
  transcript TEXT NOT NULL,
  output TEXT NOT NULL,
  corrected TEXT,
  corrected_ts REAL
);
"""


@dataclass
class Entry:
    id: int
    ts: float
    mode: str
    app: str | None
    transcript: str
    output: str
    corrected: str | None


def _conn(db: Path = DB) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db)
    c.executescript(SCHEMA)
    return c


def record(transcript: str, output: str, mode: str, app: str | None = None, db: Path = DB) -> int:
    with _conn(db) as c:
        cur = c.execute("INSERT INTO dictations (ts, mode, app, transcript, output) VALUES (?,?,?,?,?)",
                        (time.time(), mode, app, transcript, output))
        return cur.lastrowid


def recent(limit: int = 20, db: Path = DB) -> list[Entry]:
    with _conn(db) as c:
        rows = c.execute("SELECT id, ts, mode, app, transcript, output, corrected FROM dictations "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [Entry(*r) for r in rows]


def get(entry_id: int, db: Path = DB) -> Entry | None:
    with _conn(db) as c:
        r = c.execute("SELECT id, ts, mode, app, transcript, output, corrected FROM dictations WHERE id=?",
                      (entry_id,)).fetchone()
    return Entry(*r) if r else None


def correct(entry_id: int, text: str, db: Path = DB) -> None:
    with _conn(db) as c:
        c.execute("UPDATE dictations SET corrected=?, corrected_ts=? WHERE id=?", (text, time.time(), entry_id))


def training_pairs(modes: tuple[str, ...] = ("dictation", "dictation_ft"), db: Path = DB) -> list[dict]:
    """Corrected dictations, plus confirmed-good ones (output accepted as is) marked with corrected == output."""
    with _conn(db) as c:
        rows = c.execute("SELECT id, transcript, corrected FROM dictations WHERE corrected IS NOT NULL AND mode IN "
                         f"({','.join('?' * len(modes))}) ORDER BY id", modes).fetchall()
    return [{"id": i, "src": t, "tgt": k} for i, t, k in rows]


def clear(db: Path = DB) -> int:
    with _conn(db) as c:
        n = c.execute("SELECT COUNT(*) FROM dictations").fetchone()[0]
        c.execute("DELETE FROM dictations")
    return n

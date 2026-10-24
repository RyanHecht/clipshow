"""SQLite storage: episodes, moments, FTS5 over dialogue, and float16 vector blobs."""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    season INTEGER,
    episode INTEGER,
    duration REAL,
    signature TEXT,
    sub_path TEXT,
    indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS moments (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL,
    kind TEXT NOT NULL,          -- 'dialogue' | 'silent'
    text TEXT NOT NULL,          -- the line(s) spoken in the moment
    context TEXT NOT NULL,       -- neighbouring lines, used for embedding
    frames TEXT NOT NULL,        -- JSON list of thumb paths relative to THUMBS_DIR
    description TEXT             -- Gemini-written reaction description
);
CREATE INDEX IF NOT EXISTS moments_episode ON moments(episode_id, start);
CREATE VIRTUAL TABLE IF NOT EXISTS moments_fts USING fts5(text, context, speaker, notes, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS jellyfin_items (
    file TEXT PRIMARY KEY,       -- episode file name (paths differ between Jellyfin and this server)
    item_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vectors (
    moment_id INTEGER NOT NULL REFERENCES moments(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,          -- 'text' | 'visual' | 'desc' | 'notes'
    vec BLOB NOT NULL,
    PRIMARY KEY (moment_id, kind)
);
"""


@dataclass
class Moment:
    id: int
    episode_id: int
    label: str
    path: str
    sub_path: str | None
    start: float
    end: float
    kind: str
    text: str
    context: str
    frames: list[str]
    description: str | None
    speaker: str | None = None  # "Arthur", or "Arthur & Buster" for a line two people share
    notes: str | None = None  # stage directions from a transcript
    score: float = 0.0
    hits: dict[str, int] = field(default_factory=dict)
    ds: float = 0.0  # suggested start/end adjustment (e.g. to include a neighbouring line the query quotes)
    de: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "episode": self.label,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "kind": self.kind,
            "text": self.text,
            "description": self.description,
            "speaker": self.speaker,
            "notes": self.notes,
            "thumb": self.frames[len(self.frames) // 2] if self.frames else None,
            "score": round(self.score, 5),
            "hits": self.hits,
            "ds": round(self.ds, 2),
            "de": round(self.de, 2),
        }


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn) -> None:
    """Brings databases created by earlier versions up to the current schema."""
    for table, cols in (("episodes", ("cuts", "cuts_sig", "speech")), ("moments", ("speaker", "notes"))):
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col in cols:
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
    fts_cols = [d[0] for d in conn.execute("SELECT * FROM moments_fts LIMIT 0").description]
    if fts_cols != ["text", "context", "speaker", "notes"]:
        conn.execute("DROP TABLE moments_fts")
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO moments_fts (rowid, text, context, speaker, notes) SELECT id, text, context, speaker, notes FROM moments")
    conn.commit()


def set_jellyfin_items(conn, items: dict[str, str]) -> None:
    conn.execute("DELETE FROM jellyfin_items")
    conn.executemany("INSERT INTO jellyfin_items (file, item_id) VALUES (?, ?)", list(items.items()))
    conn.commit()


def jellyfin_item(conn, episode_path: str) -> str | None:
    r = conn.execute("SELECT item_id FROM jellyfin_items WHERE file = ?", (Path(episode_path).name,)).fetchone()
    return r[0] if r else None


def get_episode(conn, path: str):
    return conn.execute("SELECT * FROM episodes WHERE path = ?", (path,)).fetchone()


def upsert_episode(conn, path, label, season, episode, duration, signature, sub_path) -> int:
    conn.execute(
        """INSERT INTO episodes (path, label, season, episode, duration, signature, sub_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET label=excluded.label, season=excluded.season,
             episode=excluded.episode, duration=excluded.duration,
             signature=excluded.signature, sub_path=excluded.sub_path, indexed_at=NULL""",
        (path, label, season, episode, duration, signature, sub_path),
    )
    return get_episode(conn, path)["id"]


def clear_episode_moments(conn, episode_id: int) -> None:
    ids = [r[0] for r in conn.execute("SELECT id FROM moments WHERE episode_id = ?", (episode_id,))]
    conn.executemany("DELETE FROM moments_fts WHERE rowid = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM moments WHERE episode_id = ?", (episode_id,))


def saved_descriptions(conn, episode_id: int) -> dict[str, list[tuple[float, float, str, bytes | None]]]:
    """Descriptions (and their vectors) of an episode's moments, grouped by line text, to survive a re-index."""
    rows = conn.execute(
        """SELECT m.start, m.end, m.text, m.description, v.vec FROM moments m
           LEFT JOIN vectors v ON v.moment_id = m.id AND v.kind = 'desc'
           WHERE m.episode_id = ? AND m.description IS NOT NULL""",
        (episode_id,),
    )
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["text"], []).append((r["start"], r["end"], r["description"], r["vec"]))
    return out


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(max(a1, b1) - min(a0, b0), 1e-6)


def restore_descriptions(conn, saved: dict, ids: list[int], spans, min_iou: float = 0.5) -> int:
    """Reuses a description when the line text matches and the old and new windows mostly overlap."""
    n = 0
    for mid, s in zip(ids, spans):
        cands = saved.get(s.text, [])
        best = max(cands, key=lambda c: _iou(s.start, s.end, c[0], c[1]), default=None)
        if best is None or _iou(s.start, s.end, best[0], best[1]) < min_iou:
            continue
        set_description(conn, mid, best[2])
        if best[3] is not None:
            conn.execute("INSERT OR REPLACE INTO vectors (moment_id, kind, vec) VALUES (?, 'desc', ?)", (mid, best[3]))
        n += 1
    return n


def insert_moment(conn, episode_id, start, end, kind, text, context, speaker=None, notes=None) -> int:
    cur = conn.execute(
        "INSERT INTO moments (episode_id, start, end, kind, text, context, frames, speaker, notes) VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?)",
        (episode_id, start, end, kind, text, context, speaker, notes),
    )
    conn.execute(
        "INSERT INTO moments_fts (rowid, text, context, speaker, notes) VALUES (?, ?, ?, ?, ?)",
        (cur.lastrowid, text, context, speaker, notes),
    )
    return cur.lastrowid


def set_annotations(conn, moment_id: int, speaker: str | None, notes: str | None) -> None:
    conn.execute("UPDATE moments SET speaker = ?, notes = ? WHERE id = ?", (speaker, notes, moment_id))
    conn.execute("UPDATE moments_fts SET speaker = ?, notes = ? WHERE rowid = ?", (speaker, notes, moment_id))


def speakers(conn) -> dict[str, int]:
    """Every individual speaker name with its number of moments."""
    out: dict[str, int] = {}
    for sp, n in conn.execute("SELECT speaker, count(*) FROM moments WHERE speaker IS NOT NULL GROUP BY speaker"):
        for name in sp.split(" & "):
            out[name] = out.get(name, 0) + n
    return out


def set_frames(conn, moment_id: int, frames: list[str]) -> None:
    conn.execute("UPDATE moments SET frames = ? WHERE id = ?", (json.dumps(frames), moment_id))


def set_description(conn, moment_id: int, description: str) -> None:
    conn.execute("UPDATE moments SET description = ? WHERE id = ?", (description, moment_id))


def put_vectors(conn, kind: str, ids: list[int], vecs: np.ndarray) -> None:
    vecs = vecs.astype(np.float16)
    conn.executemany(
        "INSERT OR REPLACE INTO vectors (moment_id, kind, vec) VALUES (?, ?, ?)",
        [(i, kind, v.tobytes()) for i, v in zip(ids, vecs)],
    )


def load_vectors(conn, kind: str) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute("SELECT moment_id, vec FROM vectors WHERE kind = ? ORDER BY moment_id", (kind,)).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r[1], dtype=np.float16) for r in rows]).astype(np.float32)
    return ids, mat


def set_analysis(conn, episode_id: int, cuts: list[float], speech: list[tuple[float, float]], file_sig: str) -> None:
    conn.execute(
        "UPDATE episodes SET cuts = ?, speech = ?, cuts_sig = ? WHERE id = ?",
        (json.dumps([round(c, 3) for c in cuts]), json.dumps(speech), file_sig, episode_id),
    )


def episode_states(conn) -> dict[str, dict]:
    """{episode path: {"signature", "indexed"}}: what workers need to decide what to (re)index."""
    return {r["path"]: {"signature": r["signature"], "indexed": bool(r["indexed_at"])}
            for r in conn.execute("SELECT path, signature, indexed_at FROM episodes")}


def cached_analysis(row, file_sig: str) -> tuple[list[float] | None, list[tuple[float, float]] | None]:
    """Shot cuts and voice activity from a previous run, if the video file hasn't changed since."""
    if row is None or row["cuts_sig"] != file_sig:
        return None, None
    cuts = json.loads(row["cuts"]) if row["cuts"] is not None else None
    speech = [tuple(x) for x in json.loads(row["speech"])] if row["speech"] is not None else None
    return cuts, speech


def mark_indexed(conn, episode_id: int) -> None:
    conn.execute("UPDATE episodes SET indexed_at = datetime('now') WHERE id = ?", (episode_id,))


MOMENT_SELECT = """
SELECT m.*, e.label, e.path, e.sub_path FROM moments m JOIN episodes e ON e.id = m.episode_id
"""


def row_to_moment(r) -> Moment:
    return Moment(
        id=r["id"], episode_id=r["episode_id"], label=r["label"], path=r["path"], sub_path=r["sub_path"],
        start=r["start"], end=r["end"], kind=r["kind"], text=r["text"], context=r["context"],
        frames=json.loads(r["frames"]), description=r["description"],
        speaker=r["speaker"], notes=r["notes"],
    )


def get_moments(conn, ids: list[int]) -> dict[int, Moment]:
    if not ids:
        return {}
    q = MOMENT_SELECT + f" WHERE m.id IN ({','.join('?' * len(ids))})"
    return {r["id"]: row_to_moment(r) for r in conn.execute(q, ids)}


def get_moment(conn, moment_id: int) -> Moment | None:
    return get_moments(conn, [moment_id]).get(moment_id)

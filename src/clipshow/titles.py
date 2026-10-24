"""Titles: one searchable show each, registered in the hub database with their own data folder.

    $CLIPSHOW_DATA/
      hub.sqlite              titles and their settings
      titles/<slug>/           index.sqlite, thumbs/, renders/, subs/, transcripts/

Settings hold everything specific to a show: theme-song detection, lines to skip, speaker aliases.
After changing segmentation settings, re-index the title (cached shot cuts and
voice activity make that fast).
"""

import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from pathlib import Path

from . import config

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


@dataclass
class Settings:
    language: str = "en"
    skip_theme_songs: bool = True  # drop the opening theme / closing credits song (runs of lyric cues near either end)
    drop_before_theme: bool = False  # also drop everything before the opening theme (e.g. network bumpers; not cold opens)
    theme_within: float = 180.0  # seconds from either end where the theme / credits song can be
    theme_match: str | None = None  # a phrase from the theme lyrics, so other songs near the start aren't mistaken for it
    skip_patterns: list[str] = field(default_factory=list)  # subtitle lines matching these (regex) are dropped
    aliases: dict[str, list[str]] = field(default_factory=dict)  # canonical speaker name -> other names for them

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    @cached_property
    def skip_re(self) -> re.Pattern | None:
        return re.compile("|".join(f"(?:{p})" for p in self.skip_patterns), re.I) if self.skip_patterns else None

    @cached_property
    def _canonical(self) -> dict[str, str]:
        out = {}
        for canon, others in self.aliases.items():
            for n in [canon, *others]:
                out[_key(n)] = canon
        return out

    def canonical(self, name: str) -> str:
        """The title's name for a speaker ("DAVID" -> "Mr. Read" given an alias)."""
        return self._canonical.get(_key(name), name)


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class Title:
    slug: str
    name: str
    media: str | None
    settings: Settings
    jellyfin_id: str | None = None
    theme: dict | None = None

    @property
    def root(self) -> Path:
        return config.DATA_ROOT / "titles" / self.slug

    @property
    def db_path(self) -> Path:
        return self.root / "index.sqlite"

    @property
    def thumbs(self) -> Path:
        return self.root / "thumbs"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    @property
    def subs(self) -> Path:
        return self.root / "subs"

    @property
    def transcripts(self) -> Path:
        return self.root / "transcripts"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.thumbs, self.renders, self.subs, self.transcripts):
            d.mkdir(parents=True, exist_ok=True)


# --- hub registry ------------------------------------------------------------------------------

HUB_SCHEMA = """
CREATE TABLE IF NOT EXISTS titles (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    media TEXT,                 -- media folder (as this host sees it)
    jellyfin_id TEXT,           -- the series' Jellyfin item id, if commissioned from Jellyfin
    settings TEXT NOT NULL DEFAULT '{}',
    theme TEXT,                 -- colours and artwork taken from the show's Jellyfin images
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,         -- random, in the visitor's cookie
    jf_token TEXT NOT NULL,      -- the user's Jellyfin access token (never sent to the browser)
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS favorites (
    id INTEGER PRIMARY KEY,
    owner TEXT NOT NULL,         -- 'user:<jellyfin user id>' or 'browser:<random id>' (signed-out visitors)
    title TEXT NOT NULL,
    moment_id INTEGER NOT NULL,  -- refreshed if a re-index renumbers moments (matched by episode file + window)
    episode_file TEXT NOT NULL,
    ds REAL NOT NULL DEFAULT 0,  -- trim the clip was starred with
    de REAL NOT NULL DEFAULT 0,
    clip_start REAL NOT NULL,    -- the starred clip's window in the episode (what stays rendered)
    clip_end REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (owner, title, moment_id, ds, de)
);
CREATE INDEX IF NOT EXISTS favorites_owner ON favorites(owner, title);
CREATE TABLE IF NOT EXISTS shares (
    token TEXT PRIMARY KEY,      -- random; the public link's only key
    owner TEXT NOT NULL,         -- 'user:<jellyfin user id>' or 'browser:<random id>'
    title TEXT NOT NULL,
    moment_id INTEGER NOT NULL,
    fmt TEXT NOT NULL,           -- gif | clip | frame
    ds REAL NOT NULL DEFAULT 0,
    de REAL NOT NULL DEFAULT 0,
    at REAL NOT NULL DEFAULT 0,  -- frame: seconds into the trimmed clip
    caption INTEGER NOT NULL DEFAULT 1,
    owner_name TEXT,             -- who shared it, for the admin list
    text TEXT NOT NULL DEFAULT '',    -- the line, for link previews
    source TEXT NOT NULL DEFAULT '',  -- e.g. "Arthur, S09E17"
    file TEXT NOT NULL,          -- the rendered file, relative to the data folder (never evicted)
    poster TEXT,                 -- a still for link previews of videos
    views INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_viewed TEXT
);
CREATE INDEX IF NOT EXISTS shares_owner ON shares(owner, title, moment_id);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,          -- 'index' | 'transcripts'
    status TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | failed | cancelled
    params TEXT NOT NULL DEFAULT '{}',
    progress TEXT,
    error TEXT,
    worker TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def hub() -> sqlite3.Connection:
    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DATA_ROOT / "hub.sqlite", check_same_thread=False, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(HUB_SCHEMA)
    if "theme" not in {r[1] for r in conn.execute("PRAGMA table_info(titles)")}:  # added after the first schema
        conn.execute("ALTER TABLE titles ADD COLUMN theme TEXT")
    return conn


def _row_to_title(r) -> Title:
    return Title(r["slug"], r["name"], r["media"], Settings.from_dict(json.loads(r["settings"] or "{}")), r["jellyfin_id"],
                json.loads(r["theme"]) if r["theme"] else None)


def list_titles() -> list[Title]:
    with hub() as h:
        return [_row_to_title(r) for r in h.execute("SELECT * FROM titles ORDER BY name")]


def get_title(slug: str) -> Title | None:
    with hub() as h:
        r = h.execute("SELECT * FROM titles WHERE slug = ?", (slug,)).fetchone()
    return _row_to_title(r) if r else None


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:63] or "title"


def create_title(name: str, media: str | None, settings: Settings | None = None, slug: str | None = None,
                jellyfin_id: str | None = None) -> Title:
    slug = slug or slugify(name)
    if not SLUG_RE.match(slug):
        raise ValueError(f"invalid title slug {slug!r}: use lowercase letters, digits and dashes")
    with hub() as h:
        if h.execute("SELECT 1 FROM titles WHERE slug = ?", (slug,)).fetchone():
            raise FileExistsError(f"a title called {slug!r} already exists")
        h.execute("INSERT INTO titles (slug, name, media, jellyfin_id, settings) VALUES (?, ?, ?, ?, ?)",
                  (slug, name, media, jellyfin_id, json.dumps((settings or Settings()).to_dict())))
    title = get_title(slug)
    title.ensure_dirs()
    return title


def set_jellyfin(slug: str, jellyfin_id: str | None, theme: dict | None) -> None:
    with hub() as h:
        h.execute("UPDATE titles SET jellyfin_id = ?, theme = ? WHERE slug = ?", (jellyfin_id, json.dumps(theme) if theme else None, slug))


def save_settings(slug: str, settings: Settings) -> None:
    with hub() as h:
        h.execute("UPDATE titles SET settings = ? WHERE slug = ?", (json.dumps(settings.to_dict()), slug))


def delete_title(slug: str) -> None:
    title = get_title(slug)
    with hub() as h:
        h.execute("DELETE FROM titles WHERE slug = ?", (slug,))
        h.execute("DELETE FROM shares WHERE title = ?", (slug,))
    if title:
        shutil.rmtree(title.root, ignore_errors=True)

"""Public share links: an unguessable token for one rendered clip, GIF or frame.

Clip Show itself can stay private (on a tailnet, say) while share links are served by a separate listener
that knows nothing but /s/<token>.<ext>: publish that one port (Tailscale Funnel, a Cloudflare Tunnel, a
reverse proxy) and the rest of the app stays unreachable. A shared file is rendered when the link is made
and never evicted, so strangers can only fetch what someone chose to share, not make the server render.
"""

import secrets
from pathlib import Path

from . import auth, config, titles

EXT = {"gif": "gif", "clip": "mp4", "frame": "png"}
MIME = {"gif": "image/gif", "clip": "video/mp4", "frame": "image/png"}
POLICIES = ("everyone", "signed_in", "off")


def allowed(user) -> bool:
    """Whether this visitor may make share links (the `share_links` setting)."""
    policy = auth.get_setting("share_links")
    return policy == "everyone" or (policy == "signed_in" and user is not None)


def base_url(fallback: str) -> str:
    """Where share links point: the public address if one is set, else the address the request came in on."""
    return (auth.get_setting("share_url") or fallback).rstrip("/")


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(config.DATA_ROOT.resolve()))


def create(owner: str, owner_name: str, title: titles.Title, moment_id: int, fmt: str, ds: float, de: float, at: float,
           caption: bool, text: str, source: str, file: Path, poster: Path | None) -> str:
    """Returns the share's token, reusing the existing one if this owner already shared exactly this."""
    key = (owner, title.slug, moment_id, fmt, round(ds, 2), round(de, 2), round(at, 2), int(caption))
    with titles.hub() as h:
        row = h.execute("""SELECT token FROM shares WHERE owner = ? AND title = ? AND moment_id = ? AND fmt = ?
                           AND ds = ? AND de = ? AND at = ? AND caption = ?""", key).fetchone()
        if row:
            h.execute("UPDATE shares SET file = ?, poster = ? WHERE token = ?",
                      (_rel(file), _rel(poster) if poster else None, row["token"]))
            return row["token"]
        token = secrets.token_urlsafe(9)
        h.execute("""INSERT INTO shares (token, owner, title, moment_id, fmt, ds, de, at, caption, owner_name, text, source,
                                         file, poster) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (token, *key, owner_name, text, source, _rel(file), _rel(poster) if poster else None))
        return token


def get(token: str):
    with titles.hub() as h:
        return h.execute("SELECT * FROM shares WHERE token = ?", (token,)).fetchone()


def viewed(token: str) -> None:
    with titles.hub() as h:
        h.execute("UPDATE shares SET views = views + 1, last_viewed = datetime('now') WHERE token = ?", (token,))


def path(row, kind: str = "file") -> Path | None:
    rel = row[kind]
    return config.DATA_ROOT / rel if rel else None


def revoke(token: str) -> bool:
    with titles.hub() as h:
        return h.execute("DELETE FROM shares WHERE token = ?", (token,)).rowcount > 0


def listing(limit: int = 200) -> list[dict]:
    with titles.hub() as h:
        rows = h.execute("SELECT * FROM shares ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        f = path(r)
        out.append({k: r[k] for k in ("token", "title", "fmt", "text", "source", "owner_name", "views", "created_at", "last_viewed")}
                   | {"ext": EXT[r["fmt"]], "missing": not (f and f.exists())})
    return out


def usage() -> dict:
    with titles.hub() as h:
        count = h.execute("SELECT count(*) FROM shares").fetchone()[0]
    size = sum(p.stat().st_size for p in pinned_files() if p.exists())
    return {"count": count, "mb": round(size / 2**20, 1)}


def pinned_files() -> set[Path]:
    """Render files behind a share link: the cache never evicts these."""
    with titles.hub() as h:
        rows = h.execute("SELECT file, poster FROM shares").fetchall()
    return {config.DATA_ROOT / rel for r in rows for rel in (r["file"], r["poster"]) if rel}

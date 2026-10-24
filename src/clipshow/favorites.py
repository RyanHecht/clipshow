"""Starred clips: per Jellyfin user, or per anonymous browser id for signed-out visitors.

Signed-out visitors keep their list in browser storage; the server only records the clips so it can keep
them rendered (favorites are never evicted from the render cache). Signing in moves them to the account.
"""

from pathlib import Path

from . import auth, render, store, titles
from .store import Moment

FORMATS_KEPT = ("clip", "gif")  # renders a favorite keeps (only those actually made take space)


def owner_for(user_id: str | None, browser_id: str | None) -> str | None:
    if user_id:
        return f"user:{user_id}"
    if browser_id and 8 <= len(browser_id) <= 64 and browser_id.replace("-", "").isalnum():
        return f"browser:{browser_id}"
    return None


def limit_for(owner: str | None) -> int:
    """How many favorites this owner may keep, across all titles (0 = favorites are off for them)."""
    key = "favorites_browser" if owner is None or owner.startswith("browser:") else "favorites_user"
    return max(0, int(auth.get_setting(key) or 0))


def add(owner: str, title: titles.Title, conn, m: Moment, ds: float, de: float) -> None:
    start, end = render.window(m, ds, de)
    limit = limit_for(owner)
    signed_out = owner.startswith("browser:")
    with titles.hub() as h:
        if limit == 0:
            raise PermissionError("Favorites are turned off" + (" unless you sign in with Jellyfin." if signed_out else "."))
        n = h.execute("SELECT count(*) FROM favorites WHERE owner = ?", (owner,)).fetchone()[0]
        exists = h.execute("SELECT 1 FROM favorites WHERE owner = ? AND title = ? AND moment_id = ? AND ds = ? AND de = ?",
                           (owner, title.slug, m.id, round(ds, 2), round(de, 2))).fetchone()
        if n >= limit and not exists:
            raise PermissionError(f"You've reached the limit of {limit} favorites. Remove some to add more"
                                  + (", or sign in with Jellyfin." if signed_out else "."))
        h.execute(
            """INSERT INTO favorites (owner, title, moment_id, episode_file, ds, de, clip_start, clip_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            (owner, title.slug, m.id, Path(m.path).name, round(ds, 2), round(de, 2), start, end))


def remove(owner: str, slug: str, moment_id: int, ds: float, de: float) -> None:
    with titles.hub() as h:
        h.execute("DELETE FROM favorites WHERE owner = ? AND title = ? AND moment_id = ? AND ds = ? AND de = ?",
                  (owner, slug, moment_id, round(ds, 2), round(de, 2)))


def claim(browser_owner: str, user_owner: str) -> int:
    """Moves a signed-out browser's favorites to the account it just signed in with (newest first, up to the
    account's limit; any beyond it are dropped, since the browser's own list is cleared too)."""
    with titles.hub() as h:
        room = max(0, limit_for(user_owner) - h.execute("SELECT count(*) FROM favorites WHERE owner = ?", (user_owner,)).fetchone()[0])
        moved = h.execute("""INSERT INTO favorites (owner, title, moment_id, episode_file, ds, de, clip_start, clip_end, created_at)
                     SELECT ?, title, moment_id, episode_file, ds, de, clip_start, clip_end, created_at FROM favorites WHERE owner = ?
                     ORDER BY created_at DESC, id DESC LIMIT ? ON CONFLICT DO NOTHING""", (user_owner, browser_owner, room)).rowcount
        h.execute("DELETE FROM favorites WHERE owner = ?", (browser_owner,))
        return moved


def _resolve(conn, fav) -> Moment | None:
    """The favorite's moment, re-found by episode file and window if a re-index renumbered moments."""
    m = store.get_moment(conn, fav["moment_id"])
    if m is not None and Path(m.path).name == fav["episode_file"]:
        return m
    rows = conn.execute(
        """SELECT m.id FROM moments m JOIN episodes e ON e.id = m.episode_id
           WHERE e.path LIKE ? AND m.end > ? AND m.start < ?""",
        (f"%/{fav['episode_file']}", fav["clip_start"], fav["clip_end"])).fetchall()
    best, best_overlap = None, 0.0
    for (mid,) in rows:
        cand = store.get_moment(conn, mid)
        overlap = min(cand.end, fav["clip_end"]) - max(cand.start, fav["clip_start"])
        if overlap > best_overlap:
            best, best_overlap = cand, overlap
    if best is not None:
        with titles.hub() as h:
            h.execute("UPDATE favorites SET moment_id = ? WHERE id = ?", (best.id, fav["id"]))
    return best


def list_for(owner: str, title: titles.Title, conn) -> list[tuple[Moment, float, float]]:
    with titles.hub() as h:
        favs = h.execute("SELECT * FROM favorites WHERE owner = ? AND title = ? ORDER BY created_at DESC, id DESC",
                         (owner, title.slug)).fetchall()
    out = []
    for f in favs:
        m = _resolve(conn, f)
        if m is not None:
            out.append((m, f["ds"], f["de"]))
    return out


def usage() -> dict:
    """Favorites across everyone, and how much of the render cache they hold."""
    with titles.hub() as h:
        count = h.execute("SELECT count(*) FROM favorites").fetchone()[0]
        owners = h.execute("SELECT count(DISTINCT owner) FROM favorites").fetchone()[0]
    size = sum(p.stat().st_size for p in pinned_files() if p.exists())
    return {"count": count, "owners": owners, "mb": round(size / 2**20, 1)}


def pinned_files() -> set[Path]:
    """Render files that belong to someone's favorite: the cache never evicts these."""
    from . import config

    with titles.hub() as h:
        rows = h.execute("SELECT DISTINCT title, moment_id, clip_start, clip_end FROM favorites").fetchall()
    out = set()
    for r in rows:
        base = config.DATA_ROOT / "titles" / r["title"] / "renders"
        for fmt in FORMATS_KEPT:
            out.add(base / f"{r['moment_id']}_{r['clip_start']:.2f}_{r['clip_end']:.2f}_{fmt}.{render.FORMATS[fmt]}")
    return out

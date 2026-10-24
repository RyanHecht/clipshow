"""Creating titles from Jellyfin shows, linking existing titles to them, and the indexing job queue."""

import json
from pathlib import Path

from . import auth, titles, store, theme
from .titles import Title


def link(title: Title, user: auth.User, series_id: str) -> dict:
    """Ties a title to a Jellyfin show or movie: episode ids (for "Open in Jellyfin") and the theme."""
    jf = user.client()
    item = jf.item(series_id, user.user_id)
    episodes = jf.episodes(series_id, user.user_id) if item.get("Type") == "Series" else [item]
    items = {Path(e["Path"]).name: e["Id"] for e in episodes if e.get("Path")}
    conn = store.connect(title.db_path)
    store.set_jellyfin_items(conn, items)
    indexed = {Path(r[0]).name for r in conn.execute("SELECT path FROM episodes")}
    conn.close()
    titles.set_jellyfin(title.slug, series_id, theme.build(title, jf, series_id))
    return {"jellyfin_episodes": len(items), "matched": len(indexed & set(items)), "indexed": len(indexed)}


def find_match(user: auth.User, name: str) -> list[dict]:
    """Jellyfin shows/movies whose name matches a title's, for linking titles made outside Jellyfin."""
    hits = user.client().children("", user.user_id, search=name)
    exact = [h for h in hits if h["Name"].lower() == name.lower()]
    return exact or hits[:10]


def commission(user: auth.User, series_id: str) -> Title:
    """Creates a title for a Jellyfin show (or movie) and queues it for indexing."""
    series = user.client().item(series_id, user.user_id)
    jf_path = series.get("Path")
    if not jf_path:
        raise ValueError("Jellyfin didn't report where this is stored")
    local = auth.map_path(jf_path)
    if not Path(local).exists():
        raise FileNotFoundError(
            f"Jellyfin keeps this at {jf_path}, which this server sees as {local}, but that doesn't exist here. "
            "Fill in this server's folder for that library under Folders.")
    for existing in titles.list_titles():
        if existing.jellyfin_id == series_id:
            raise FileExistsError(f"{series['Name']} already has a title ({existing.slug})")
    slug = titles.slugify(series["Name"])
    n = 2
    while titles.get_title(slug):
        slug, n = f"{titles.slugify(series['Name'])}-{n}", n + 1
    title = titles.create_title(series["Name"], local, slug=slug, jellyfin_id=series_id)
    link(title, user, series_id)
    enqueue(title.slug, "index")
    return titles.get_title(title.slug)


# --- jobs --------------------------------------------------------------------------------------

def enqueue(slug: str, kind: str, params: dict | None = None) -> int:
    with titles.hub() as h:
        busy = h.execute("SELECT id FROM jobs WHERE title = ? AND kind = ? AND status IN ('queued', 'running')", (slug, kind)).fetchone()
        if busy:
            return busy[0]
        return h.execute("INSERT INTO jobs (title, kind, params) VALUES (?, ?, ?)", (slug, kind, json.dumps(params or {}))).lastrowid


def jobs(slug: str | None = None, limit: int = 50) -> list[dict]:
    with titles.hub() as h:
        q = "SELECT * FROM jobs" + (" WHERE title = ?" if slug else "") + " ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in h.execute(q, ((slug, limit) if slug else (limit,)))]


def claim(worker: str) -> dict | None:
    """Atomically takes the oldest queued job."""
    with titles.hub() as h:
        r = h.execute(
            """UPDATE jobs SET status = 'running', worker = ?, updated_at = datetime('now')
               WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1) RETURNING *""", (worker,)).fetchone()
    return dict(r) if r else None


def update(job_id: int, status: str | None = None, progress: str | None = None, error: str | None = None) -> None:
    with titles.hub() as h:
        h.execute("""UPDATE jobs SET status = coalesce(?, status), progress = coalesce(?, progress), error = coalesce(?, error),
                     updated_at = datetime('now') WHERE id = ?""", (status, progress, error, job_id))

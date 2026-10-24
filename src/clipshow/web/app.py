"""Web app: the title directory at /, every title's search under /s/<slug>/, Jellyfin sign-in, and the admin page."""

import io
import json
import os
import re
import secrets
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import auth, casing, commission, favorites, indexer, shares, titles, store
from ..cache import renders as render_cache
from ..jellyfin import JellyfinError
from ..render import FORMATS, frame_png, line_at, render, window
from ..search import search as run_search
from .share import router as share_router

STATIC = Path(__file__).parent / "static"
MIME = {"gif": "image/gif", "clip": "video/mp4", "still": "image/jpeg"}

app = FastAPI(title="Clip Show")
if not auth.JELLYFIN_URL:
    raise SystemExit("Clip Show needs to know your Jellyfin server: set JELLYFIN_URL (e.g. http://jellyfin:8096) and start it again.")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(share_router)  # share links work here too, not only on the public share listener


class TitleCache:
    """Open titles: their settings and one database connection each, refreshed when the hub's row changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._open: dict[str, tuple[str, titles.Title, object]] = {}

    def get(self, slug: str) -> tuple[titles.Title, object]:
        with titles.hub() as h:
            row = h.execute("SELECT * FROM titles WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            raise HTTPException(404, f"There's no title called {slug!r}.")
        version = json.dumps([row["name"], row["media"], row["settings"], row["jellyfin_id"], row["theme"]])
        with self._lock:
            hit = self._open.get(slug)
            if hit is None or hit[0] != version:
                title = titles.get_title(slug)
                conn = hit[2] if hit else store.connect(title.db_path)
                self._open[slug] = hit = (version, title, conn)
            return hit[1], hit[2]

    def drop(self, slug: str) -> None:
        with self._lock:
            hit = self._open.pop(slug, None)
        if hit:
            hit[2].close()


_titles = TitleCache()


# --- who's asking ------------------------------------------------------------------------------

class NeedsSignIn(Exception):
    pass


def current_user(request: Request) -> auth.User | None:
    return auth.get_session(request.cookies.get(auth.SESSION_COOKIE))


def viewer(request: Request) -> auth.User | None:
    """Anyone may look unless the sign-in policy is 'required'."""
    user = current_user(request)
    if user is None and auth.get_setting("signin") == "required":
        raise NeedsSignIn()
    return user


def signed_in(request: Request) -> auth.User:
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "Sign in with your Jellyfin account first.")
    return user


def admin(request: Request) -> auth.User:
    user = signed_in(request)
    if not user.is_admin:
        raise HTTPException(403, "Only Jellyfin administrators can manage titles.")
    return user


@app.exception_handler(NeedsSignIn)
def _needs_sign_in(request: Request, _exc):
    if request.url.path.startswith("/api/") or "/api/" in request.url.path:
        return Response('{"detail": "Sign in with your Jellyfin account first."}', 401, media_type="application/json")
    return RedirectResponse(f"/login?next={request.url.path}", 303)


@app.exception_handler(JellyfinError)
def _jellyfin_error(_request: Request, exc: JellyfinError):
    return Response(json.dumps({"detail": str(exc)}), 502, media_type="application/json")


def _set_session_cookie(request: Request, response: Response, user: auth.User) -> None:
    response.set_cookie(auth.SESSION_COOKIE, user.session_id, max_age=auth.SESSION_DAYS * 86400, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")


# --- sign-in -----------------------------------------------------------------------------------

@app.get("/login")
def login_page():
    return FileResponse(STATIC / "login.html")


@app.get("/api/auth/state")
def auth_state(request: Request):
    user = current_user(request)
    configured = bool(auth.JELLYFIN_URL)
    qc = False
    if configured:
        try:
            qc = auth.jellyfin().quick_connect_enabled()
        except (JellyfinError, RuntimeError):
            pass
    return {
        "signin": auth.get_setting("signin"),
        "favorites_limit": favorites.limit_for(favorites.owner_for(user.user_id, None) if user else None),
        "share_links": shares.allowed(user),
        "configured": configured,
        "quick_connect": qc,
        "user": {"name": user.name, "admin": user.is_admin} if user else None,
    }


def _finish_login(request: Request, response: Response, result: dict) -> dict:
    user = auth.create_session(result)
    _set_session_cookie(request, response, user)
    return {"name": user.name, "admin": user.is_admin}


@app.post("/api/auth/login")
def login(request: Request, response: Response, username: str = Body(...), password: str = Body("")):
    try:
        result = auth.jellyfin().authenticate(username, password)
    except JellyfinError as e:
        raise HTTPException(401, "That username and password didn't work on Jellyfin.") from e
    return _finish_login(request, response, result)


@app.post("/api/auth/quickconnect")
def quick_connect_start():
    r = auth.jellyfin().quick_connect_start()
    return {"code": r["Code"], "secret": r["Secret"]}


@app.post("/api/auth/quickconnect/check")
def quick_connect_check(request: Request, response: Response, secret: str = Body(..., embed=True)):
    jf = auth.jellyfin()
    if not jf.quick_connect_state(secret).get("Authenticated"):
        return {"pending": True}
    return _finish_login(request, response, jf.quick_connect_finish(secret))


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    sid = request.cookies.get(auth.SESSION_COOKIE)
    if sid:
        auth.end_session(sid)
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


# --- directory ---------------------------------------------------------------------------------

# --- clips rendered ahead of time ---------------------------------------------------------------

_prefetcher = ThreadPoolExecutor(2, thread_name_prefix="prefetch")
_prefetch_queued = 0
_prefetch_lock = threading.Lock()
PREFETCH_BACKLOG = 24  # skip prefetching while this many renders are already waiting


def _prefetch_one(m, out_dir: Path, fix=None) -> None:
    global _prefetch_queued
    try:
        render(m, "clip", out_dir, m.ds, m.de, fix=fix)
    except Exception:  # noqa: BLE001 - a failed prefetch just means rendering on demand later
        pass
    finally:
        with _prefetch_lock:
            _prefetch_queued -= 1


def prefetch(moments, out_dir: Path, force: bool = False, fix=None) -> None:
    """Renders the top results' clips in the background (with their suggested trims), so play is instant.

    A play request for a clip that's still being prefetched waits for that render instead of starting another.
    """
    global _prefetch_queued
    n = len(moments) if force else int(auth.get_setting("prefetch") or 0)
    for m in moments[:n]:
        with _prefetch_lock:
            if _prefetch_queued >= PREFETCH_BACKLOG:
                return
            _prefetch_queued += 1
        _prefetcher.submit(_prefetch_one, m, out_dir, fix)


def _moment(conn, moment_id: int):
    m = store.get_moment(conn, moment_id)
    if m is None:
        raise HTTPException(404, "That moment no longer exists. Search again to refresh results.")
    return m


def _title_summary(s: titles.Title) -> dict:
    entry = {"slug": s.slug, "name": s.name, "episodes": 0, "moments": 0, "thumb": None, "theme": s.theme,
             "jellyfin": bool(s.jellyfin_id)}
    if s.db_path.exists():
        _, conn = _titles.get(s.slug)
        entry["episodes"] = conn.execute("SELECT count(*) FROM episodes WHERE indexed_at IS NOT NULL").fetchone()[0]
        entry["moments"] = conn.execute("SELECT count(*) FROM moments").fetchone()[0]
        if s.theme and "backdrop" in s.theme.get("images", []):
            entry["thumb"] = f"/t/{s.slug}/theme/backdrop"
        else:
            r = conn.execute("SELECT frames FROM moments WHERE kind = 'dialogue' ORDER BY id LIMIT 1 OFFSET 200").fetchone()
            if r and json.loads(r[0]):
                entry["thumb"] = f"/t/{s.slug}/thumbs/{json.loads(r[0])[0]}"
    return entry


@app.get("/")
def home(_=Depends(viewer)):
    return FileResponse(STATIC / "home.html")


@app.get("/api/titles")
def api_titles(_=Depends(viewer)):
    return {"titles": [_title_summary(s) for s in titles.list_titles()]}


# --- one title ----------------------------------------------------------------------------------

@app.get("/t/{slug}")
def title_redirect(slug: str):
    return RedirectResponse(f"/t/{slug}/")  # the page uses relative URLs


@app.get("/t/{slug}/")
def title_page(slug: str, _=Depends(viewer)):
    _titles.get(slug)
    return FileResponse(STATIC / "index.html")


@app.get("/t/{slug}/api/title")
def api_title(slug: str, user=Depends(viewer)):
    title, _ = _titles.get(slug)
    return {"slug": title.slug, "name": title.name, "theme": title.theme, "jellyfin": bool(title.jellyfin_id),
            "signin": auth.get_setting("signin"), "user": {"name": user.name, "admin": user.is_admin} if user else None}


@app.get("/t/{slug}/theme/{kind}")
def theme_image(slug: str, kind: str, _=Depends(viewer)):
    title, _ = _titles.get(slug)
    f = title.root / "theme" / kind
    if kind not in ("backdrop", "primary", "logo") or not f.is_file():
        raise HTTPException(404, "No such image.")
    return FileResponse(f, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/t/{slug}/thumbs/{path:path}")
def thumb(slug: str, path: str, _=Depends(viewer)):
    title, _ = _titles.get(slug)
    f = (title.thumbs / path).resolve()
    if not f.is_relative_to(title.thumbs.resolve()) or not f.is_file():
        raise HTTPException(404, "No such thumbnail.")
    return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/t/{slug}/api/search")
def api_search(slug: str, q: str, k: int = Query(24, le=100), _=Depends(viewer)):
    title, conn = _titles.get(slug)
    results = run_search(conn, q, k=k, settings=title.settings)
    prefetch(results, title.renders, fix=casing.for_db(conn).fix)
    return {"results": [_moment_dict(title, conn, m) for m in results]}


def _moment_dict(title, conn, m) -> dict:
    d = m.to_dict()
    d["text"] = casing.for_db(conn).fix(d["text"])  # ALL-CAPS captions shown in sentence case
    d["jellyfin"] = bool(title.jellyfin_id) and store.jellyfin_item(conn, m.path) is not None
    return d


# --- favorites ---------------------------------------------------------------------------------

def _owner(request: Request, user: auth.User | None) -> str:
    owner = favorites.owner_for(user.user_id if user else None, request.headers.get("X-Clipshow-Browser"))
    if owner is None:
        raise HTTPException(400, "Missing browser id for signed-out favorites.")
    return owner


@app.get("/t/{slug}/api/favorites")
def api_favorites(slug: str, request: Request, user=Depends(viewer)):
    title, conn = _titles.get(slug)
    owner = _owner(request, user)
    return {"favorites": [{**_moment_dict(title, conn, m), "ds": ds, "de": de}
                          for m, ds, de in favorites.list_for(owner, title, conn)]}


@app.post("/t/{slug}/api/favorites")
def api_favorite_add(slug: str, request: Request, moment_id: int = Body(...), ds: float = Body(0.0), de: float = Body(0.0),
                     user=Depends(viewer)):
    title, conn = _titles.get(slug)
    m = _moment(conn, moment_id)
    try:
        favorites.add(_owner(request, user), title, conn, m, ds, de)
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    m.ds, m.de = ds, de
    prefetch([m], title.renders, force=True, fix=casing.for_db(conn).fix)  # a favorite's clip is kept rendered
    return {"ok": True}


@app.delete("/t/{slug}/api/favorites")
def api_favorite_remove(slug: str, request: Request, moment_id: int = Body(...), ds: float = Body(0.0), de: float = Body(0.0),
                        user=Depends(viewer)):
    _titles.get(slug)
    favorites.remove(_owner(request, user), slug, moment_id, ds, de)
    return {"ok": True}


@app.post("/api/favorites/claim")
def api_favorites_claim(browser_id: str = Body(..., embed=True), user: auth.User = Depends(signed_in)):
    browser_owner = favorites.owner_for(None, browser_id)
    if browser_owner is None:
        raise HTTPException(400, "Invalid browser id.")
    return {"moved": favorites.claim(browser_owner, favorites.owner_for(user.user_id, None))}


@app.get("/t/{slug}/api/render/{moment_id}.{fmt}")
def api_render(slug: str, moment_id: int, fmt: str, ds: float = 0.0, de: float = 0.0, download: bool = False, _=Depends(viewer)):
    fmt = {"mp4": "clip"}.get(fmt, fmt)  # shared links end in .mp4, which chat apps recognise as video
    if fmt not in FORMATS:
        raise HTTPException(400, f"format must be one of {list(FORMATS)}")
    title, conn = _titles.get(slug)
    m = _moment(conn, moment_id)
    path = render(m, fmt, title.renders, ds, de, fix=casing.for_db(conn).fix)
    name = f"{m.label}_{int(m.start)}s.{FORMATS[fmt]}" if download else None
    return FileResponse(path, media_type=MIME[fmt], filename=name)


@app.get("/t/{slug}/api/frame/{moment_id}.png")
def api_frame(slug: str, moment_id: int, offset: float = Query(..., description="seconds into the (trimmed) clip"),
              ds: float = 0.0, de: float = 0.0, caption: bool = True, download: bool = False, _=Depends(viewer)):
    title, conn = _titles.get(slug)
    m = _moment(conn, moment_id)
    start, end = window(m, ds, de)
    t = min(max(start + offset, start), end)
    path = frame_png(m, t, title.renders, caption, fix=casing.for_db(conn).fix)
    name = f"{m.label}_{t:.1f}s.png" if download else None
    return FileResponse(path, media_type="image/png", filename=name)


# --- share links -------------------------------------------------------------------------------

@app.post("/t/{slug}/api/share")
def api_share(slug: str, request: Request, moment_id: int = Body(...), fmt: str = Body(...), ds: float = Body(0.0),
              de: float = Body(0.0), offset: float = Body(0.0), caption: bool = Body(True), user=Depends(viewer)):
    """Makes (or reuses) a public link to this clip, GIF or frame, rendering it now so the link works at once."""
    if fmt not in shares.EXT:
        raise HTTPException(400, f"format must be one of {list(shares.EXT)}")
    if not shares.allowed(user):
        raise HTTPException(403, "Share links are turned off" + (" unless you sign in with Jellyfin." if user is None
                                                                  and auth.get_setting("share_links") == "signed_in" else "."))
    owner = _owner(request, user)
    title, conn = _titles.get(slug)
    m = _moment(conn, moment_id)
    fix = casing.for_db(conn).fix
    start, end = window(m, ds, de)
    poster = None
    if fmt == "frame":
        t = min(max(start + offset, start), end)
        file = frame_png(m, t, title.renders, caption, fix=fix)
        text = (line_at(m, t) or m.text) if caption else ""
    else:
        file = render(m, fmt, title.renders, ds, de, fix=fix)
        poster = frame_png(m, (start + end) / 2, title.renders, True, fix=fix) if fmt == "clip" else None
        text, offset, caption = m.text, 0.0, True
    token = shares.create(owner, user.name if user else "Signed-out visitor", title, m.id, fmt, ds, de, offset, caption,
                          fix(text), f"{title.name}, {m.label}", file, poster)
    base = shares.base_url(str(request.base_url))
    return {"url": f"{base}/s/{token}.{shares.EXT[fmt]}", "page": f"{base}/s/{token}", "public": bool(auth.get_setting("share_url"))}


# --- open in Jellyfin --------------------------------------------------------------------------

def _seek_new_tab(user: auth.User, item_id: str, seconds: float, clicked: datetime) -> None:
    """Waits for the Jellyfin tab the user just opened to check in, then starts the episode at the moment there.

    Jellyfin's web client has no link that starts at a time, so this uses remote control: the newest web
    session for this user that has been active since the click gets a "play from" command.
    """
    jf = user.client()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(1.5)
        try:
            candidates = []
            for s in jf.sessions(user.user_id, active_within=60):
                if s.get("DeviceId") == jf.device_id or not s.get("SupportsRemoteControl"):
                    continue
                if "web" not in (s.get("Client") or "").lower():
                    continue
                seen = datetime.fromisoformat(s["LastActivityDate"].replace("Z", "+00:00"))
                if seen >= clicked:
                    candidates.append((seen, s["Id"]))
            if candidates:
                jf.play(max(candidates)[1], item_id, seconds)
                return
        except JellyfinError:
            return


@app.post("/t/{slug}/api/open/{moment_id}")
def api_open(slug: str, moment_id: int, ds: float = 0.0, user: auth.User = Depends(signed_in)):
    title, conn = _titles.get(slug)
    m = _moment(conn, moment_id)
    item_id = store.jellyfin_item(conn, m.path)
    if not item_id:
        raise HTTPException(404, "This episode isn't linked to Jellyfin.")
    base = auth.public_jellyfin_url()
    server_id = auth.get_setting("jellyfin_server_id") or ""
    clicked = datetime.now(timezone.utc)
    threading.Thread(target=_seek_new_tab, args=(user, item_id, max(0.0, m.start + ds), clicked), daemon=True).start()
    return {"url": f"{base}/web/#/details?id={item_id}&serverId={server_id}"}


# --- admin -------------------------------------------------------------------------------------

@app.get("/admin")
def admin_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login?next=/admin", 303)
    if not user.is_admin:
        raise HTTPException(403, "Only Jellyfin administrators can manage titles.")
    return FileResponse(STATIC / "admin.html")


@app.get("/api/admin/state")
def admin_state(_=Depends(admin)):
    all_jobs = commission.jobs(limit=200)
    latest = {}
    for j in all_jobs:
        latest.setdefault(j["title"], j)
    out = []
    for s in titles.list_titles():
        entry = _title_summary(s)
        entry.update({"media": s.media, "settings": s.settings.to_dict(), "job": latest.get(s.slug)})
        out.append(entry)
    return {"settings": {**auth.all_settings(), "jellyfin_url": auth.JELLYFIN_URL}, "titles": out, "jobs": all_jobs[:20],
            "render_cache": {"used_mb": round(render_cache.used() / 2**20, 1)}, "favorites": favorites.usage(),
            "shares": shares.usage(), "share_listener": bool(os.environ.get("CLIPSHOW_SHARE_PORT", "").strip("0"))}


@app.get("/api/admin/shares")
def admin_shares(_=Depends(admin)):
    return {"shares": shares.listing(), "base": auth.get_setting("share_url") or ""}


@app.delete("/api/admin/shares/{token}")
def admin_share_revoke(token: str, _=Depends(admin)):
    if not shares.revoke(token):
        raise HTTPException(404, "No such share link.")
    return {"revoked": token}


@app.put("/api/admin/settings")
def admin_settings(body: dict = Body(...), _=Depends(admin)):
    if "signin" in body and body["signin"] not in ("optional", "required"):
        raise HTTPException(400, "Sign-in must be 'optional' or 'required'.")
    if "share_links" in body and body["share_links"] not in shares.POLICIES:
        raise HTTPException(400, f"Share links must be one of {list(shares.POLICIES)}.")
    if "share_url" in body:
        body["share_url"] = str(body["share_url"] or "").strip().rstrip("/")
        if body["share_url"] and not re.fullmatch(r"https?://[^\s/]+(/[^\s]*)?", body["share_url"]):
            raise HTTPException(400, "The public address for share links must start with https:// (or http://).")
    if "path_maps" in body:
        maps = body["path_maps"]
        if not isinstance(maps, list) or not all(isinstance(m, list) and len(m) == 2 and all(isinstance(x, str) for x in m) for m in maps):
            raise HTTPException(400, "Path mappings must be pairs of folders.")
    for key, kind in (("render_cache_mb", float), ("prefetch", int), ("favorites_user", int), ("favorites_browser", int)):
        if key in body:
            try:
                body[key] = kind(body[key])
            except (TypeError, ValueError) as e:
                raise HTTPException(400, f"{key} must be a number.") from e
            if body[key] < 0:
                raise HTTPException(400, f"{key} can't be negative.")
    for key in ("jellyfin_public_url", "signin", "path_maps", "render_cache_mb", "prefetch", "favorites_user", "favorites_browser",
                "share_links", "share_url"):
        if key in body:
            auth.set_setting(key, body[key])
    if "render_cache_mb" in body:
        render_cache.evict()  # a lower cap applies right away
    return auth.all_settings()


@app.get("/api/admin/folders")
def admin_folders(user: auth.User = Depends(admin)):
    """Every Jellyfin library folder, with where this server finds it (via the folder mappings)."""
    out = []
    for lib in user.client().libraries():
        for loc in lib.get("Locations") or []:
            local = auth.map_path(loc)
            out.append({"library": lib["Name"], "type": lib.get("CollectionType") or "mixed", "jellyfin": loc,
                        "local": local, "found": Path(local).is_dir()})
    return {"folders": out}


def _item_entry(item: dict, by_jf: dict) -> dict:
    local = auth.map_path(item["Path"]) if item.get("Path") else None
    return {"id": item["Id"], "name": item["Name"], "type": item.get("Type"), "year": item.get("ProductionYear"),
            "path": item.get("Path"), "local": local, "found": bool(local and Path(local).exists()),
            "title": by_jf.get(item["Id"]), "episodes": item.get("RecursiveItemCount")}


@app.get("/api/admin/libraries")
def admin_libraries(user: auth.User = Depends(admin)):
    kinds = {"tvshows", "movies", "mixed", "homevideos", "musicvideos", None}
    return {"libraries": [{"id": v["Id"], "name": v["Name"], "type": v.get("CollectionType")}
                          for v in user.client().views(user.user_id) if v.get("CollectionType") in kinds]}


@app.get("/api/admin/libraries/{library_id}/items")
def admin_library_items(library_id: str, user: auth.User = Depends(admin)):
    by_jf = {s.jellyfin_id: s.slug for s in titles.list_titles() if s.jellyfin_id}
    return {"items": [_item_entry(i, by_jf) for i in user.client().children(library_id, user.user_id)]}


@app.get("/api/admin/titles/{slug}/matches")
def admin_title_matches(slug: str, user: auth.User = Depends(admin)):
    title, _ = _titles.get(slug)
    by_jf = {s.jellyfin_id: s.slug for s in titles.list_titles() if s.jellyfin_id}
    return {"items": [_item_entry(i, by_jf) for i in commission.find_match(user, title.name)]}


@app.post("/api/admin/titles")
def admin_commission(series_id: str = Body(..., embed=True), user: auth.User = Depends(admin)):
    try:
        title = commission.commission(user, series_id)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return {"slug": title.slug}


@app.post("/api/admin/titles/{slug}/link")
def admin_link(slug: str, series_id: str = Body(..., embed=True), user: auth.User = Depends(admin)):
    title, _ = _titles.get(slug)
    return commission.link(title, user, series_id)


@app.post("/api/admin/titles/{slug}/index")
def admin_reindex(slug: str, _=Depends(admin)):
    _titles.get(slug)
    return {"job": commission.enqueue(slug, "index")}


@app.put("/api/admin/titles/{slug}/settings")
def admin_title_settings(slug: str, body: dict = Body(...), _=Depends(admin)):
    title, _ = _titles.get(slug)
    titles.save_settings(slug, titles.Settings.from_dict({**title.settings.to_dict(), **body}))
    return titles.get_title(slug).settings.to_dict()


@app.delete("/api/admin/titles/{slug}")
def admin_delete(slug: str, _=Depends(admin)):
    _titles.get(slug)
    _titles.drop(slug)
    titles.delete_title(slug)
    return {"deleted": slug}


# --- workers -----------------------------------------------------------------------------------
# Workers index on their own hardware and talk to the web app only through these endpoints: claim a
# job, fetch what they need, upload each finished episode. The web app stays the only writer of the
# database and the titles' folders, so workers can run on other machines with no shared storage.

WORKER_TOKEN = os.environ.get("CLIPSHOW_WORKER_TOKEN", "")
_store_locks: dict[str, threading.Lock] = {}
_store_locks_guard = threading.Lock()


def worker(request: Request) -> None:
    if not WORKER_TOKEN:
        raise HTTPException(503, "Workers are turned off: set CLIPSHOW_WORKER_TOKEN on the web app and on each worker.")
    sent = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(sent.encode(), WORKER_TOKEN.encode()):
        raise HTTPException(401, "Wrong worker token.")


def _store_lock(slug: str) -> threading.Lock:
    with _store_locks_guard:
        return _store_locks.setdefault(slug, threading.Lock())


@app.post("/api/worker/claim")
def worker_claim(name: str = Body(..., embed=True), _=Depends(worker)):
    while True:
        job = commission.claim(name[:100])
        if job is None:
            return Response(status_code=204)
        title = titles.get_title(job["title"])
        if title is not None:
            break
        commission.update(job["id"], status="failed", error="The title was deleted.")
    return {"job": {"id": job["id"], "kind": job["kind"], "params": json.loads(job["params"] or "{}")},
            "title": {"slug": title.slug, "name": title.name, "media": title.media, "settings": title.settings.to_dict()}}


@app.post("/api/worker/jobs/{job_id}")
def worker_job_update(job_id: int, status: str | None = Body(None), progress: str | None = Body(None),
                      error: str | None = Body(None), _=Depends(worker)):
    if status not in (None, "running", "done", "failed"):
        raise HTTPException(400, "status must be running, done or failed")
    commission.update(job_id, status=status, progress=progress[:500] if progress else None, error=error[:2000] if error else None)
    return {"ok": True}


@app.get("/api/worker/titles/{slug}/episodes")
def worker_episodes(slug: str, _=Depends(worker)):
    _, conn = _titles.get(slug)
    return {"episodes": store.episode_states(conn)}


@app.get("/api/worker/titles/{slug}/analysis")
def worker_analysis(slug: str, path: str, file_signature: str, _=Depends(worker)):
    """Shot cuts and voice activity saved from an earlier run, so re-indexing skips that work."""
    _, conn = _titles.get(slug)
    cuts, speech = store.cached_analysis(store.get_episode(conn, path), file_signature)
    return {"cuts": cuts, "speech": speech}


@app.get("/api/worker/titles/{slug}/transcripts/{name}")
def worker_transcript(slug: str, name: str, _=Depends(worker)):
    title, _ = _titles.get(slug)
    f = (title.transcripts / name).resolve()
    if not name.endswith(".transcript.txt") or not f.is_relative_to(title.transcripts.resolve()) or not f.is_file():
        raise HTTPException(404, "No such transcript.")
    return FileResponse(f, media_type="text/plain; charset=utf-8")


@app.post("/api/worker/titles/{slug}/episodes")
def worker_upload_episode(slug: str, meta: UploadFile = File(...), vectors: UploadFile = File(...),
                          thumbs: UploadFile = File(...), _=Depends(worker)):
    """Stores one indexed episode: its moments and vectors, subtitles, and a tar of its thumbnails."""
    title, _ = _titles.get(slug)
    try:
        info = json.loads(meta.file.read())
        with np.load(io.BytesIO(vectors.file.read()), allow_pickle=False) as npz:
            vecs = {k: npz[k] for k in npz.files}
        result = indexer.EpisodeResult.from_meta(info, vecs)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(400, f"Malformed episode upload: {e}") from e
    if "/" in result.thumb_dir or result.thumb_dir in ("", ".", ".."):
        raise HTTPException(400, "Bad thumbnail folder name.")
    if any(not t.startswith(result.thumb_dir + "/") for t in result.thumbs):
        raise HTTPException(400, "Thumbnails must be inside the episode's thumbnail folder.")
    title.ensure_dirs()
    with tarfile.open(fileobj=thumbs.file, mode="r:*") as tar:
        members = tar.getmembers()
        if any(not m.name.startswith(result.thumb_dir + "/") or not m.isfile() for m in members if not m.isdir()):
            raise HTTPException(400, "Unexpected file in the thumbnail upload.")
        tar.extractall(title.thumbs, filter="data")
    with _store_lock(slug):  # one writer per title; a fresh connection so it never mixes with readers' state
        conn = store.connect(title.db_path)
        try:
            episode_id = indexer.store_result(conn, title, result, log=lambda msg: None)
        finally:
            conn.close()
    return {"episode_id": episode_id}

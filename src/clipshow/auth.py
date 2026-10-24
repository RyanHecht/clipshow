"""Global settings and Jellyfin sign-in sessions (both in the hub database)."""

import json
import os
import secrets
import uuid
from dataclasses import dataclass

from . import titles
from .jellyfin import Jellyfin

SESSION_COOKIE = "clipshow_session"
SESSION_DAYS = 30

# Which Jellyfin server to trust for sign-in comes only from the environment, fixed before startup: if
# it were settable from the website, whoever reached a fresh install first could point it at a Jellyfin
# server they control and make themselves admin.
JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "").rstrip("/")

# Defaults for settings admins can change in the admin page.
DEFAULTS = {
    "jellyfin_public_url": os.environ.get("JELLYFIN_PUBLIC_URL", ""),  # how browsers reach Jellyfin (default: JELLYFIN_URL)
    "signin": os.environ.get("CLIPSHOW_SIGNIN", "optional"),  # 'optional' | 'required'
    "path_maps": [],  # [[jellyfin prefix, local prefix], ...]
    "render_cache_mb": float(os.environ.get("CLIPSHOW_RENDER_CACHE_MB", "2048")),  # cap for rendered clips/GIFs/frames
    "prefetch": int(os.environ.get("CLIPSHOW_PREFETCH", "6")),  # clips rendered ahead for each search (0 = off)
    # Favorites are kept rendered forever, so these cap how much of the render cache they can hold (0 = off).
    "favorites_user": int(os.environ.get("CLIPSHOW_FAVORITES_USER", "500")),  # per Jellyfin account
    "favorites_browser": int(os.environ.get("CLIPSHOW_FAVORITES_BROWSER", "100")),  # per signed-out browser
    # Public share links: who may make them ('everyone' | 'signed_in' | 'off'), and the public address that
    # reaches the share listener (CLIPSHOW_SHARE_PORT), e.g. https://box.tailnet.ts.net:8443.
    "share_links": os.environ.get("CLIPSHOW_SHARE_LINKS", "signed_in"),
    "share_url": os.environ.get("CLIPSHOW_SHARE_URL", ""),
}


def get_setting(key: str):
    with titles.hub() as h:
        r = h.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return json.loads(r[0]) if r else DEFAULTS.get(key)


def set_setting(key: str, value) -> None:
    with titles.hub() as h:
        h.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                  (key, json.dumps(value)))


def all_settings() -> dict:
    return {k: get_setting(k) for k in DEFAULTS}


def device_id() -> str:
    """This server's stable identity as a Jellyfin client."""
    d = get_setting("device_id")
    if not d:
        d = uuid.uuid4().hex
        set_setting("device_id", d)
    return d


def jellyfin(token: str | None = None) -> Jellyfin:
    if not JELLYFIN_URL:
        raise RuntimeError("Jellyfin isn't configured: set JELLYFIN_URL")
    return Jellyfin(JELLYFIN_URL, token, device_id())


def public_jellyfin_url() -> str:
    return (get_setting("jellyfin_public_url") or JELLYFIN_URL).rstrip("/")


def map_path(jellyfin_path: str) -> str:
    """A path as Jellyfin sees it -> as this server sees it, using the longest matching prefix rule."""
    for src, dst in sorted(get_setting("path_maps") or [], key=lambda m: -len(m[0])):
        src = src.rstrip("/")
        if jellyfin_path == src or jellyfin_path.startswith(src + "/"):
            return dst.rstrip("/") + jellyfin_path[len(src):]
    return jellyfin_path


# --- sessions ----------------------------------------------------------------------------------

@dataclass
class User:
    session_id: str
    jf_token: str
    user_id: str
    name: str
    is_admin: bool

    def client(self) -> Jellyfin:
        return jellyfin(self.jf_token)


def create_session(auth_result: dict) -> User:
    u = auth_result["User"]
    sid = secrets.token_urlsafe(32)
    is_admin = bool(u.get("Policy", {}).get("IsAdministrator"))
    with titles.hub() as h:
        h.execute("INSERT INTO sessions (id, jf_token, user_id, user_name, is_admin) VALUES (?, ?, ?, ?, ?)",
                  (sid, auth_result["AccessToken"], u["Id"], u["Name"], int(is_admin)))
    if auth_result.get("ServerId"):
        set_setting("jellyfin_server_id", auth_result["ServerId"])
    return User(sid, auth_result["AccessToken"], u["Id"], u["Name"], is_admin)


def get_session(sid: str | None) -> User | None:
    if not sid:
        return None
    with titles.hub() as h:
        r = h.execute(f"SELECT * FROM sessions WHERE id = ? AND last_seen > datetime('now', '-{SESSION_DAYS} days')", (sid,)).fetchone()
        if r is None:
            return None
        h.execute("UPDATE sessions SET last_seen = datetime('now') WHERE id = ?", (sid,))
    return User(r["id"], r["jf_token"], r["user_id"], r["user_name"], bool(r["is_admin"]))


def end_session(sid: str) -> None:
    user = get_session(sid)
    if user:
        try:
            user.client().sign_out()
        except RuntimeError:
            pass
    with titles.hub() as h:
        h.execute("DELETE FROM sessions WHERE id = ?", (sid,))

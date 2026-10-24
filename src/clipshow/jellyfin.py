"""Minimal Jellyfin API client: sign-in (password or Quick Connect), series and episodes, remote playback."""

import uuid

import httpx

CLIENT = "Clip Show"
VERSION = "0.1.0"
TICKS_PER_SECOND = 10_000_000


class JellyfinError(Exception):
    pass


class Jellyfin:
    def __init__(self, base_url: str, token: str | None = None, device_id: str | None = None):
        self.base = base_url.rstrip("/")
        self.token = token
        self.device_id = device_id or uuid.uuid4().hex
        self.http = httpx.Client(timeout=20)

    def _auth_header(self) -> str:
        parts = [f'Client="{CLIENT}"', 'Device="Clip Show server"', f'DeviceId="{self.device_id}"', f'Version="{VERSION}"']
        if self.token:
            parts.append(f'Token="{self.token}"')
        return "MediaBrowser " + ", ".join(parts)

    def _req(self, method: str, path: str, **kw):
        headers = {"Authorization": self._auth_header(), **kw.pop("headers", {})}
        try:
            r = self.http.request(method, self.base + path, headers=headers, **kw)
        except httpx.HTTPError as e:
            raise JellyfinError(f"couldn't reach Jellyfin at {self.base}: {e}") from e
        if r.status_code == 401:
            raise JellyfinError("Jellyfin rejected the sign-in or the session has expired")
        if r.status_code >= 400:
            raise JellyfinError(f"Jellyfin answered {r.status_code} for {path}: {r.text[:200]}")
        return r.json() if r.content and "json" in r.headers.get("content-type", "") else None

    # --- public ---------------------------------------------------------------------------------
    def public_info(self) -> dict:
        return self._req("GET", "/System/Info/Public")

    def quick_connect_enabled(self) -> bool:
        return bool(self._req("GET", "/QuickConnect/Enabled"))

    # --- sign-in -------------------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> dict:
        """AuthenticationResult: {User, AccessToken, ServerId, ...}."""
        return self._req("POST", "/Users/AuthenticateByName", json={"Username": username, "Pw": password})

    def quick_connect_start(self) -> dict:
        """{Code, Secret, ...}: show Code to the user; poll quick_connect_state(Secret)."""
        return self._req("POST", "/QuickConnect/Initiate")

    def quick_connect_state(self, secret: str) -> dict:
        return self._req("GET", "/QuickConnect/Connect", params={"secret": secret})

    def quick_connect_finish(self, secret: str) -> dict:
        return self._req("POST", "/Users/AuthenticateWithQuickConnect", json={"Secret": secret})

    def sign_out(self) -> None:
        try:
            self._req("POST", "/Sessions/Logout")
        except JellyfinError:
            pass

    # --- signed in -----------------------------------------------------------------------------
    def me(self) -> dict:
        return self._req("GET", "/Users/Me")

    def libraries(self) -> list[dict]:
        """Library folders as Jellyfin sees them (admins only): [{Name, CollectionType, Locations, ItemId}]."""
        return self._req("GET", "/Library/VirtualFolders") or []

    def views(self, user_id: str) -> list[dict]:
        """The user's top-level libraries (TV Shows, Movies, ...)."""
        return (self._req("GET", "/UserViews", params={"userId": user_id}) or {}).get("Items", [])

    def children(self, parent_id: str, user_id: str, search: str | None = None) -> list[dict]:
        """Shows and movies inside a library (or matching a search across all of them)."""
        params = {"userId": user_id, "recursive": "true", "includeItemTypes": "Series,Movie", "sortBy": "SortName",
                  "fields": "Path,ProductionYear,RecursiveItemCount"}
        if parent_id:
            params["parentId"] = parent_id
        if search:
            params["searchTerm"] = search
        return (self._req("GET", "/Items", params=params) or {}).get("Items", [])

    def episodes(self, series_id: str, user_id: str) -> list[dict]:
        r = self._req("GET", f"/Shows/{series_id}/Episodes", params={"userId": user_id, "fields": "Path,MediaStreams"})
        return r.get("Items", [])

    def item(self, item_id: str, user_id: str) -> dict:
        return self._req("GET", f"/Items/{item_id}", params={"userId": user_id, "fields": "Path"})

    def sessions(self, user_id: str, active_within: int = 120) -> list[dict]:
        return self._req("GET", "/Sessions", params={"controllableByUserId": user_id, "activeWithinSeconds": active_within}) or []

    def play(self, session_id: str, item_id: str, start_seconds: float) -> None:
        self._req("POST", f"/Sessions/{session_id}/Playing", params={
            "playCommand": "PlayNow", "itemIds": item_id, "startPositionTicks": int(start_seconds * TICKS_PER_SECOND),
        })

    def image(self, item_id: str, kind: str, max_width: int = 1600) -> bytes | None:
        """Item artwork (public in Jellyfin): Primary (poster), Backdrop, Logo, Thumb."""
        try:
            r = self.http.get(f"{self.base}/Items/{item_id}/Images/{kind}", params={"maxWidth": max_width, "quality": 90})
        except httpx.HTTPError:
            return None
        return r.content if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") else None

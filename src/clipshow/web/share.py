"""Share links, served on their own: `public_app` answers /s/<token>.<ext> and /s/<token> and nothing else.

Run it on a separate port (CLIPSHOW_SHARE_PORT) and publish only that port, so a share link can be public
while the rest of Clip Show stays private. The main app serves the same routes too.
"""

import html
import re

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from .. import shares

TOKEN = re.compile(r"[A-Za-z0-9_-]{8,32}")
CACHE = {"Cache-Control": "public, max-age=31536000, immutable", "X-Robots-Tag": "noindex"}

router = APIRouter()


def _share(token: str):
    row = shares.get(token) if TOKEN.fullmatch(token) else None
    f = shares.path(row) if row else None
    if f is None or not f.exists():
        raise HTTPException(404, "This link doesn't exist or has been taken down.")
    return row, f


@router.api_route("/s/{token}/poster.png", methods=["GET", "HEAD"])
def shared_poster(token: str):
    """The still that link previews show for a video."""
    row, _ = _share(token)
    poster = shares.path(row, "poster")
    if poster is None or not poster.exists():
        raise HTTPException(404, "No preview image for this link.")
    return FileResponse(poster, media_type="image/png", headers=CACHE)


@router.api_route("/s/{token}.{ext}", methods=["GET", "HEAD"])
def shared_file(token: str, ext: str, request: Request):
    row, f = _share(token)
    if ext != shares.EXT[row["fmt"]]:
        raise HTTPException(404, "This link doesn't exist or has been taken down.")
    if request.method == "GET" and "range" not in request.headers:  # count a view once, not every video seek
        shares.viewed(token)
    return FileResponse(f, media_type=shares.MIME[row["fmt"]], headers=CACHE)


@router.get("/s/{token}")
def shared_page(token: str, request: Request):
    """A page around the file with link-preview tags, for apps that preview pages rather than files."""
    row, _ = _share(token)
    base = shares.base_url(str(request.base_url))
    file_url = f"{base}/s/{token}.{shares.EXT[row['fmt']]}"
    line = row["text"].strip()
    title = f"“{line}”" if line else row["source"]
    e = html.escape
    if row["fmt"] == "clip":
        image = f"{base}/s/{token}/poster.png" if row["poster"] else ""
        meta = (f'<meta property="og:type" content="video.other"><meta property="og:video" content="{e(file_url)}">'
                f'<meta property="og:video:type" content="video/mp4"><meta name="twitter:card" content="summary_large_image">')
        media = f'<video src="{e(file_url)}" controls autoplay muted loop playsinline></video>'
    else:
        image = file_url
        meta = '<meta property="og:type" content="website"><meta name="twitter:card" content="summary_large_image">'
        media = f'<img src="{e(file_url)}" alt="{e(line)}">'
    if image:
        meta += f'<meta property="og:image" content="{e(image)}"><meta name="twitter:image" content="{e(image)}">'
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title><meta name="robots" content="noindex">
<meta property="og:site_name" content="Clip Show"><meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(row['source'])}"><meta property="og:url" content="{e(f'{base}/s/{token}')}">
{meta}
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-content: center; gap: 12px; padding: 16px; box-sizing: border-box;
         background: #000; color: #eee; font: 16px/1.4 system-ui, sans-serif; text-align: center; }}
  img, video {{ display: block; max-width: min(960px, 100%); max-height: 80vh; margin: 0 auto; }}
  p {{ margin: 0; color: #aaa; font-size: 14px; }}
</style></head>
<body>{media}<p>{e(row['source'])}</p></body></html>"""
    return HTMLResponse(page, headers={"X-Robots-Tag": "noindex"})


public_app = FastAPI(title="Clip Show share links", docs_url=None, redoc_url=None, openapi_url=None)
public_app.include_router(router)


@public_app.get("/")
def _root():
    return Response(status_code=404)

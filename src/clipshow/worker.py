"""Indexing worker: python -m clipshow.worker

Claims jobs from the Clip Show web app over HTTP (CLIPSHOW_SERVER, authenticated with
CLIPSHOW_WORKER_TOKEN), indexes episodes on this machine's hardware, and uploads each finished episode.
It needs the media at the same path the web app sees it (e.g. both containers mount it at /media), but
no access to the web app's database or folders. Its own working files go in CLIPSHOW_SCRATCH.
"""

import argparse
import io
import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import httpx
import numpy as np

from . import indexer, ingest
from .titles import Settings, Title


SCRATCH = Path(os.environ.get("CLIPSHOW_SCRATCH", Path(tempfile.gettempdir()) / "clipshow-worker"))


class ScratchTitle(Title):
    """A title's settings with a private working folder: never the web app's own title folder, even when
    the worker runs next to it."""

    @property
    def root(self) -> Path:
        return SCRATCH / "titles" / self.slug


class Hub:
    """The web app's worker API."""

    def __init__(self, server: str, token: str):
        self.http = httpx.Client(base_url=server.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=600)

    def _check(self, r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:200]
            raise RuntimeError(f"{r.request.method} {r.request.url.path}: {r.status_code} {detail}")
        return r

    def claim(self, name: str) -> dict | None:
        r = self._check(self.http.post("/api/worker/claim", json={"name": name}))
        return None if r.status_code == 204 else r.json()

    def update(self, job_id: int, **fields) -> None:
        self._check(self.http.post(f"/api/worker/jobs/{job_id}", json=fields))

    def episodes(self, slug: str) -> dict[str, dict]:
        return self._check(self.http.get(f"/api/worker/titles/{slug}/episodes")).json()["episodes"]

    def analysis(self, slug: str, ep: ingest.EpisodeFile) -> tuple[list | None, list | None]:
        d = self._check(self.http.get(f"/api/worker/titles/{slug}/analysis",
                                      params={"path": str(ep.path), "file_signature": ep.file_signature})).json()
        speech = [tuple(x) for x in d["speech"]] if d["speech"] is not None else None
        return d["cuts"], speech

    def transcript(self, slug: str, name: str) -> str | None:
        r = self.http.get(f"/api/worker/titles/{slug}/transcripts/{name}")
        return r.text if r.status_code == 200 else None

    def upload(self, slug: str, result: indexer.EpisodeResult, thumbs_root: Path) -> None:
        vectors = io.BytesIO()
        np.savez(vectors, **result.vectors)
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w") as t:
            t.add(thumbs_root / result.thumb_dir, arcname=result.thumb_dir)
        files = {
            "meta": ("meta.json", json.dumps(result.meta()), "application/json"),
            "vectors": ("vectors.npz", vectors.getvalue(), "application/octet-stream"),
            "thumbs": ("thumbs.tar", tar.getvalue(), "application/x-tar"),
        }
        self._check(self.http.post(f"/api/worker/titles/{slug}/episodes", files=files))


def run_index(hub: Hub, job: dict, info: dict, workers: int, log) -> None:
    # A local stand-in for the title: its folder (under this worker's CLIPSHOW_DATA) is only scratch space.
    title = ScratchTitle(info["slug"], info["name"], info["media"], Settings.from_dict(info["settings"]))
    title.ensure_dirs()
    root = Path(title.media or "")
    if not root.exists():
        raise RuntimeError(f"This worker can't find {title.media}. Mount the media at the same path as the web app.")
    eps = ingest.find_episodes(root)
    todo_eps = indexer.todo_episodes(eps, hub.episodes(title.slug), force=bool(job["params"].get("force")))
    log(f"{len(eps)} videos, {len(todo_eps)} to index")

    todo = []
    for ep in todo_eps:
        name = f"{ep.path.stem}.transcript.txt"  # transcripts kept with the title (sidecars by the video are read directly)
        text = hub.transcript(title.slug, name)
        if text is not None:
            (title.transcripts / name).write_text(text, encoding="utf-8")
        todo.append((ep, *hub.analysis(title.slug, ep)))

    done = failed = skipped = 0
    t0 = time.monotonic()
    for n, (ep, prep) in enumerate(indexer.prepare_many(title, todo, True, workers), 1):
        if isinstance(prep, ingest.NoSubtitles):
            skipped += 1
        elif isinstance(prep, Exception):
            failed += 1
            log(f"{ep.label}: failed ({prep})", quiet=True)
        else:
            result = indexer.embed_episode(prep)
            hub.upload(title.slug, result, title.thumbs)
            shutil.rmtree(title.thumbs / result.thumb_dir, ignore_errors=True)
            done += 1
        left = (time.monotonic() - t0) / n * (len(todo) - n) / 60
        log(f"{n}/{len(todo)} episodes ({done} indexed, {skipped} without subtitles, {failed} failed), about {left:.0f} min left")
    if failed and not done:
        raise RuntimeError(f"all {failed} episodes failed; see the worker's log")


def main() -> None:
    p = argparse.ArgumentParser(prog="clipshow-worker")
    p.add_argument("--server", default=os.environ.get("CLIPSHOW_SERVER", "http://web:8000"), help="the Clip Show web app")
    p.add_argument("--workers", type=int, default=int(os.environ.get("CLIPSHOW_WORKERS", "4")), help="episodes prepared in parallel")
    p.add_argument("--once", action="store_true", help="run queued jobs, then exit")
    args = p.parse_args()
    token = os.environ.get("CLIPSHOW_WORKER_TOKEN", "")
    if not token:
        raise SystemExit("Set CLIPSHOW_WORKER_TOKEN to the same value as on the web app.")
    hub = Hub(args.server, token)
    name = f"{socket.gethostname()}:{os.getpid()}"
    print(f"worker {name} taking jobs from {args.server}", flush=True)
    unreachable = False
    while True:
        try:
            claimed = hub.claim(name)
            unreachable = False
        except (httpx.HTTPError, RuntimeError) as e:
            if not unreachable:
                print(f"can't reach the web app ({e}); retrying", flush=True)
                unreachable = True
            time.sleep(15)
            continue
        if claimed is None:
            if args.once:
                return
            time.sleep(5)
            continue
        job, info = claimed["job"], claimed["title"]
        print(f"job {job['id']}: {job['kind']} {info['slug']}", flush=True)

        def log(msg: str, quiet: bool = False, job_id=job["id"]) -> None:
            print(f"  {msg}", flush=True)
            if not quiet:
                hub.update(job_id, progress=msg)

        try:
            if job["kind"] != "index":
                raise RuntimeError(f"unknown job kind {job['kind']!r}")
            run_index(hub, job, info, args.workers, log)
            hub.update(job["id"], status="done", progress="finished")
        except Exception as e:  # noqa: BLE001 - record the failure and keep serving the queue
            traceback.print_exc()
            try:
                hub.update(job["id"], status="failed", error=str(e))
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()

"""Render moments as mp4 clips, GIFs, or captioned stills (cached on disk, size-capped; see cache.py)."""

import hashlib
import json
import os
import subprocess
import tempfile
import textwrap
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .cache import renders as cache
from .frames import grab_frame
from .ingest import load_cues
from .store import Moment

FORMATS = {"gif": "gif", "clip": "mp4", "still": "jpg"}
SUB_STYLE = "FontName=DejaVu Sans,Bold=1,FontSize=22,Outline=2,Shadow=0,MarginV=18"


def window(m: Moment, ds: float = 0.0, de: float = 0.0) -> tuple[float, float]:
    start = max(0.0, m.start + ds)
    return start, max(start + 0.3, m.end + de)


def _write_window_srt(m: Moment, start: float, end: float, out: Path, fix=None) -> bool:
    """Writes the cues overlapping [start, end] re-timed to the clip. Returns False if none."""
    if not m.sub_path or not Path(m.sub_path).exists():
        return False
    cues = [c for c in load_cues(Path(m.sub_path)) if c.end > start and c.start < end]
    if not cues:
        return False

    def ts(t: float) -> str:
        t = max(0.0, t)
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    out.write_text(
        "\n".join(f"{i}\n{ts(c.start - start)} --> {ts(c.end - start)}\n{fix(c.text) if fix else c.text}\n"
                  for i, c in enumerate(cues, 1)),
        encoding="utf-8",
    )
    return True


def _run_ffmpeg(args: list[str], cwd: Path) -> None:
    res = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()[-2000:]}")


@lru_cache(maxsize=256)
def _source_height(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "json", path],
                         capture_output=True, text=True)
    try:
        return int(json.loads(out.stdout)["streams"][0]["height"])
    except (ValueError, KeyError, IndexError):
        return 720


def make_clip(m: Moment, out: Path, start: float, end: float, height: int = 720, subs: bool = True, audio: bool = True,
              fix=None) -> Path:
    # Never upscale: a 480p episode stays 480p, which is also much faster to encode.
    height = min(height, _source_height(m.path)) // 2 * 2
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vf = f"scale=-2:{height}"
        if subs and _write_window_srt(m, start, end, tmp / "w.srt", fix):
            vf += f",subtitles=w.srt:force_style='{SUB_STYLE}'"
        audio_args = ["-c:a", "aac", "-b:a", "128k", "-ac", "2"] if audio else ["-an"]
        _run_ffmpeg(
            ["-ss", f"{start:.3f}", "-i", m.path, "-t", f"{end - start:.3f}", "-vf", vf, "-map", "0:v:0", "-map", "0:a:0?",
             "-c:v", "libx264", "-preset", "superfast", "-crf", "23", "-pix_fmt", "yuv420p", *audio_args,
             "-movflags", "+faststart", str(out.resolve())],
            tmp,
        )
    return out


def make_gif(m: Moment, out: Path, start: float, end: float, width: int = 480, fps: int = 12, subs: bool = True,
             fix=None) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vf = f"fps={fps},scale={width}:-2:flags=lanczos"
        if subs and _write_window_srt(m, start, end, tmp / "w.srt", fix):
            vf += f",subtitles=w.srt:force_style='{SUB_STYLE},FontSize=26'"
        vf += ",split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        _run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", m.path, "-filter_complex", vf, "-loop", "0", str(out.resolve())], tmp)
    return out


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for name in ("Impact", "Anton", "DejaVu Sans:bold"):
        res = subprocess.run(["fc-match", "--format=%{file}|%{family}", name], capture_output=True, text=True)
        if res.returncode == 0 and "|" in res.stdout:
            path, family = res.stdout.split("|", 1)
            if name.split(":")[0].lower() in family.lower():
                return path
    return None


@lru_cache(maxsize=8)
def _cues(sub_path: str):
    return load_cues(Path(sub_path))


def line_at(m: Moment, t: float) -> str:
    """The subtitle line on screen at absolute time t (empty if none)."""
    if not m.sub_path or not Path(m.sub_path).exists():
        return ""
    return " ".join(c.text for c in _cues(m.sub_path) if c.start <= t <= c.end + 0.3)


def make_still(m: Moment, out: Path, start: float, end: float, caption: str | None = None, t: float | None = None) -> Path:
    if t is None:
        t = start + (end - start) * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        frame = grab_frame(Path(m.path), t, Path(tmp) / "f.png", width=None)
        img = Image.open(frame).convert("RGB")
    if img.width > 1280:
        img = img.resize((1280, round(img.height * 1280 / img.width)), Image.LANCZOS)

    caption = caption if caption is not None else m.text
    if caption:
        draw = ImageDraw.Draw(img)
        size = max(18, img.width // 18)
        fp = _font_path()
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default(size)
        chars = max(10, int(img.width / (size * 0.55)))
        lines = textwrap.wrap(caption.upper() if fp and "impact" in fp.lower() else caption, width=chars)[:4]
        line_h = int(size * 1.15)
        y = img.height - line_h * len(lines) - size // 2
        for line in lines:
            w = draw.textlength(line, font=font)
            draw.text(((img.width - w) / 2, y), line, font=font, fill="white", stroke_width=max(2, size // 12), stroke_fill="black")
            y += line_h
    img.save(out, quality=90)
    return out


_inflight_lock = threading.Lock()
_inflight: dict[Path, threading.Lock] = {}


def _cached(out: Path, make: Callable[[Path], Path]) -> Path:
    """Returns the cached file, or makes it once even if several requests (or the prefetcher) ask at the same time.

    Files are written under a temporary name and renamed into place, so a half-written file is never served.
    """
    if out.exists():
        cache.hit(out)
        return out
    with _inflight_lock:
        lock = _inflight.setdefault(out, threading.Lock())
    with lock:
        try:
            if out.exists():  # someone else just finished it
                cache.hit(out)
                return out
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{os.getpid()}-{threading.get_ident()}-{out.name}")
            try:
                make(tmp)
                tmp.replace(out)
            finally:
                tmp.unlink(missing_ok=True)
            cache.added(out)
            return out
        finally:
            with _inflight_lock:
                _inflight.pop(out, None)


def render(m: Moment, fmt: str, out_dir: Path, ds: float = 0.0, de: float = 0.0, caption: str | None = None, fix=None) -> Path:
    """fix: optional text -> text applied to captions (e.g. sentence case for ALL-CAPS subtitles)."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {list(FORMATS)}")
    start, end = window(m, ds, de)
    # Window in the key too: moment ids can be reused after a re-index.
    key = f"{m.id}_{start:.2f}_{end:.2f}_{fmt}"
    if caption is not None:
        key += "_" + hashlib.sha1(caption.encode()).hexdigest()[:10]
    out = out_dir / f"{key}.{FORMATS[fmt]}"
    if fmt == "clip":
        return _cached(out, lambda p: make_clip(m, p, start, end, fix=fix))
    if fmt == "gif":
        return _cached(out, lambda p: make_gif(m, p, start, end, fix=fix))
    return _cached(out, lambda p: make_still(m, p, start, end, caption if caption is not None else (fix(m.text) if fix else None)))


def frame_png(m: Moment, t: float, out_dir: Path, caption: bool = True, fix=None) -> Path:
    """Full-resolution PNG of the frame at absolute time t, optionally captioned with the line on screen."""
    text = (line_at(m, t) or (m.text if m.start <= t <= m.end else "")) if caption else ""
    text = fix(text) if fix and text else text
    return _cached(out_dir / f"{m.id}_frame_{t:.2f}_{int(caption)}.png", lambda p: make_still(m, p, t, t, caption=text, t=t))

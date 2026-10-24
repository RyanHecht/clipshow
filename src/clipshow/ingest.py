"""Find episodes, probe them, and load their subtitles."""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pysrt

from . import config

EP_RE = re.compile(r"[Ss](\d{1,2})[ ._-]?[Ee](\d{1,3})(?:-?[Ee](\d{1,3}))?|(\d{1,2})x(\d{2,3})")
TAG_RE = re.compile(r"<[^>]+>|\{[^}]*\}")
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None  # from the subtitle itself or an aligned transcript
    notes: tuple[str, ...] = ()  # stage directions around this cue (from a transcript)


class NoSubtitles(Exception):
    pass


@dataclass
class EpisodeFile:
    path: Path
    label: str
    season: int | None
    episode: int | None

    @property
    def file_signature(self) -> str:
        st = self.path.stat()
        return f"{st.st_size}:{int(st.st_mtime)}"

    @property
    def signature(self) -> str:
        """Changes when the file or the segmentation rules change."""
        return f"{self.file_signature}:seg{config.SEGMENT_VERSION}"


def find_episodes(root: Path) -> list[EpisodeFile]:
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in config.VIDEO_EXTS)
    out = []
    for p in files:
        m = EP_RE.search(p.name)
        if m:
            s, e = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(5))
            season, episode = int(s), int(e)
            label = f"S{season:02d}E{episode:02d}" + (f"-E{int(m.group(3)):02d}" if m.group(3) else "")
        else:
            season = episode = None
            label = p.stem
        out.append(EpisodeFile(p.resolve(), label, season, episode))
    return out


QUALITY_RE = re.compile(
    r"\s+(?:SDTV|HDTV|WEB[- ]?DL|WEBRip|Bluray|BDRip|DVD(?:Rip)?|Remux|\d{3,4}p)\b.*$", re.I)


def episode_titles(ep: EpisodeFile) -> list[str]:
    """Episode titles from a "Show - S01E01-E02 - Title A + Title B 1080p" style filename ([] if none)."""
    m = EP_RE.search(ep.path.stem)
    if not m:
        return []
    rest = QUALITY_RE.sub("", ep.path.stem[m.end():]).strip(" -._")
    return [t.strip() for t in rest.split(" + ") if t.strip()]


def probe(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(res.stdout)


def duration(info: dict) -> float:
    return float(info["format"].get("duration", 0.0))


TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text", "eia_608"}  # eia_608: broadcast closed captions


def subtitle_file(ep: EpisodeFile, info: dict, subs_dir: Path) -> Path | None:
    """A sidecar .srt next to the video (prefers *.en.srt), else the first embedded text sub stream."""
    stem = ep.path.with_suffix("")
    for cand in (stem.with_name(stem.name + ".en.srt"), stem.with_name(stem.name + ".eng.srt"), ep.path.with_suffix(".srt")):
        if cand.exists():
            return cand
    sidecars = sorted(ep.path.parent.glob(f"{glob_escape(stem.name)}*.srt"))
    if sidecars:
        return sidecars[0]

    subs = [s for s in info["streams"] if s.get("codec_type") == "subtitle" and s.get("codec_name") in TEXT_SUB_CODECS]
    if not subs:
        return None
    eng = [s for s in subs if s.get("tags", {}).get("language", "").lower() in ("en", "eng")]
    stream = (eng or subs)[0]
    out = subs_dir / f"{ep.label}-{hashlib.md5(str(ep.path).encode()).hexdigest()[:8]}.srt"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(ep.path), "-map", f"0:{stream['index']}", "-f", "srt", str(out)],
            check=True,
        )
    return out


def glob_escape(s: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", s)


def load_cues(srt: Path) -> list[Cue]:
    subs = pysrt.open(str(srt), encoding="utf-8", error_handling=pysrt.ERROR_PASS)
    cues = []
    for s in subs:
        text = " ".join(TAG_RE.sub("", s.text).replace("\\h", " ").translate(QUOTES).split())
        if not text:
            continue
        cues.append(Cue(s.start.ordinal / 1000, s.end.ordinal / 1000, text))
    cues.sort(key=lambda c: c.start)
    return cues

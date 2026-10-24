"""Single-frame grabs with ffmpeg (for full-resolution stills) and per-moment thumbnail times."""

import subprocess
from pathlib import Path

from . import config


def grab_frame(video: Path, t: float, out: Path, width: int | None = config.THUMB_WIDTH) -> Path:
    vf = ["-vf", f"scale={width}:-2"] if width else []
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", *vf, "-q:v", "4", str(out)],
        check=True,
    )
    return out


def moment_frame_times(start: float, end: float) -> list[float]:
    return [start + (end - start) * p for p in config.FRAME_POSITIONS]

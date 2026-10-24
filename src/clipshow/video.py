"""One decoding pass per episode: shot cuts plus small frames to cut thumbnails from.

Decoding is the expensive part of indexing (a 44-minute 1080p episode is ~63k frames), so the video is
decoded once, by ffmpeg, already shrunk to thumbnail width: on the GPU via VAAPI when available,
otherwise multithreaded on the CPU. Shot cuts are detected on the small frames (the same HSV-change
measure as PySceneDetect's ContentDetector), and frames are kept at SAMPLE_FPS so thumbnails for any
moment can be taken afterwards from the nearest kept frame.
"""

import os
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config

SAMPLE_FPS = 4.0
CUT_THRESHOLD = 27.0  # ContentDetector's default
MIN_SHOT_FRAMES = 15
RENDER_NODE = os.environ.get("CLIPSHOW_VAAPI_DEVICE", "/dev/dri/renderD128")


@dataclass
class Analysis:
    cuts: list[float]
    sample_times: list[float]
    samples: list[bytes]  # JPEG-encoded frames at THUMB_WIDTH, one per sample time

    def thumb_at(self, t: float) -> bytes:
        i = bisect_left(self.sample_times, t)
        if i == len(self.sample_times) or (i > 0 and t - self.sample_times[i - 1] < self.sample_times[i] - t):
            i -= 1
        return self.samples[i]


def _probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    num, _, den = out[2].partition("/")
    return w, h, float(num) / float(den or 1)


@lru_cache(maxsize=1)
def vaapi_available() -> bool:
    if os.environ.get("CLIPSHOW_NO_VAAPI") or not Path(RENDER_NODE).exists():
        return False
    probe = subprocess.run(
        ["ffmpeg", "-v", "error", "-init_hw_device", f"vaapi=va:{RENDER_NODE}", "-f", "lavfi", "-i", "nullsrc=s=64x64",
         "-frames:v", "1", "-f", "null", "-"], capture_output=True,
    )
    return probe.returncode == 0


def _decoder(path: Path, w: int, h: int, gpu: bool) -> list[str]:
    if gpu:
        return ["ffmpeg", "-v", "error", "-hwaccel", "vaapi", "-hwaccel_device", RENDER_NODE, "-hwaccel_output_format", "vaapi",
                "-i", str(path), "-map", "0:v:0", "-vf", f"scale_vaapi=w={w}:h={h},hwdownload,format=nv12,format=bgr24",
                "-f", "rawvideo", "-"]
    return ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", f"scale={w}:{h}:flags=area,format=bgr24",
            "-f", "rawvideo", "-"]


def analyze(path: Path, detect_cuts: bool = True, gpu: bool | None = None) -> Analysis:
    src_w, src_h, fps = _probe(path)
    w = config.THUMB_WIDTH
    h = max(2, round(src_h * w / src_w / 2) * 2)
    frame_bytes = w * h * 3
    step = 1.0 / SAMPLE_FPS
    gpu = vaapi_available() if gpu is None else gpu
    proc = subprocess.Popen(_decoder(path, w, h, gpu), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 8)

    cuts: list[float] = []
    times: list[float] = []
    samples: list[bytes] = []
    prev_hsv = None
    last_cut = -MIN_SHOT_FRAMES
    next_sample = 0.0
    n = 0
    block = 256  # frames converted and compared per numpy call; a per-frame Python loop was the bottleneck
    while True:
        buf = proc.stdout.read(frame_bytes * block)
        k = len(buf) // frame_bytes
        if k == 0:
            break
        frames = np.frombuffer(buf[: k * frame_bytes], np.uint8).reshape(k, h, w, 3)
        if detect_cuts:
            hsv = cv2.cvtColor(frames.reshape(k * h, w, 3), cv2.COLOR_BGR2HSV).reshape(k, h, w, 3).astype(np.int16)
            seq = hsv if prev_hsv is None else np.concatenate([prev_hsv[None], hsv])
            scores = np.abs(np.diff(seq, axis=0)).mean(axis=(1, 2, 3))  # mean change in H, S and V
            first = n if prev_hsv is None else n - 1  # frame index each score's "before" frame belongs to
            for i, score in enumerate(scores):
                f = first + i + 1
                if score >= CUT_THRESHOLD and f - last_cut >= MIN_SHOT_FRAMES:
                    cuts.append(round(f / fps, 3))
                    last_cut = f
            prev_hsv = hsv[-1]
        for i in range(k):
            t = (n + i) / fps
            if t + 0.5 / fps >= next_sample:
                ok, jpg = cv2.imencode(".jpg", frames[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    times.append(t)
                    samples.append(jpg.tobytes())
                next_sample += step
        n += k
    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0 or n == 0:
        if gpu:  # some files/codecs the GPU can't decode: retry this one on the CPU
            return analyze(path, detect_cuts, gpu=False)
        raise RuntimeError(f"decoding failed: {err.strip()[-500:]}")
    return Analysis(cuts, times, samples)

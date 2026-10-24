"""Voice activity (Silero VAD, CPU): when people are actually speaking, which subtitle timing doesn't tell us."""

import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def _model():
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(2)  # runs inside parallel index workers
    return load_silero_vad()


def speech_segments(video: Path) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    wav = torch.from_numpy(np.frombuffer(pcm, np.int16).astype(np.float32) / 32768)
    segs = get_speech_timestamps(wav, _model(), sampling_rate=SAMPLE_RATE, return_seconds=True,
                                 min_silence_duration_ms=80, speech_pad_ms=30)
    return [(round(s["start"], 3), round(s["end"], 3)) for s in segs]

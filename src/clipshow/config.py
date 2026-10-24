"""Paths and tunables. CLIPSHOW_DATA sets the data root (hub database and every title's folder)."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CLIPSHOW_DATA", "data")).resolve()

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm"}

# Segmentation (seconds)
PAD_BEFORE = 0.3
SNAP_WINDOW = 0.5  # end on a cut at most this far past the planned end
MIN_MOMENT = 1.2
MAX_MOMENT = 12.0
MERGE_MAX_GAP = 1.5  # join cues split mid-sentence when separated by at most this
MERGE_MAX_LEN = 10.0
END_GUARD = 0.3
SPEECH_LAG = 0.25  # voices start ~this long after their subtitle appears (measured: +0.05..0.4s)
BACK_TO_BACK = 0.3  # next subtitle within this of the line's end: no pause to run into
VOICE_PAD = 0.15  # clip starts/ends this far outside the detected voice
PAUSE_MIN = 0.6  # silence after a line at least this long is a pause worth running into
TAIL_MAX = 2.5  # how far a line may run on into a following pause
LEAD_MAX = 1.0  # start on the shot's opening cut if it's at most this before the line
REACTION_SHOT_MAX = 3.0
SILENT_MIN_GAP = 1.5
SILENT_CHUNK = 4.0

# Theme-song removal: a run of >= SONG_MIN_CUES lyric cues near either end of an episode (the library's
# theme_within) is the opening theme or closing credits.
SONG_MIN_CUES = 5
SONG_MAX_GAP = 8.0

# Bump when moment boundaries change; episodes indexed under an older version are re-indexed.
SEGMENT_VERSION = 4

THUMB_WIDTH = 320
FRAME_POSITIONS = (0.15, 0.5, 0.85)

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
TEXT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VISUAL_MODEL = "google/siglip-base-patch16-224"

GEMINI_MODEL = os.environ.get("CLIPSHOW_GEMINI_MODEL", "google/gemini-3.8-flash")  # OpenRouter model id
GEMINI_REASONING = os.environ.get("CLIPSHOW_GEMINI_REASONING", "low")


def openrouter_key() -> str:
    """For the optional AI descriptions (`clipshow describe`): OPENROUTER_API_KEY from the environment."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY to use AI descriptions")
    return key

# Reciprocal-rank-fusion weights per signal
# visual and desc are off by default: visual added nothing measurable, and descriptions only cover part
# of the show so far. "postable" scales the reaction-GIF prior (0 = off).
DEFAULT_WEIGHTS = {"fts": 1.0, "text": 1.0, "notes": 1.0, "visual": 0.0, "desc": 0.0, "postable": 1.0}
RRF_K = 60
QUOTE_SIM = 0.88  # a spoken line this close in meaning to a quote-length query counts as that quote
QUOTE_MIN_WORDS = 4
CANDIDATES_PER_SIGNAL = 200

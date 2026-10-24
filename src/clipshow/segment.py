"""Turn subtitle cues and shot boundaries into searchable moments."""

import bisect
import re
from dataclasses import dataclass

from . import config
from .ingest import Cue
from .titles import Settings


@dataclass
class Span:
    start: float
    end: float
    kind: str
    text: str
    context: str
    speaker: str | None = None
    notes: str | None = None


def _song_runs(cues: list[Cue]) -> list[tuple[int, int]]:
    """Index ranges [i, j] of consecutive song-lyric cues (♪) with short gaps between them."""
    runs, i, last = [], None, 0
    for k, c in enumerate(cues):
        if "♪" not in c.text and not (i is not None and len(c.text) < 12):  # allow "Hey!" interjections inside a song
            if i is not None:
                runs.append((i, last))
                i = None
            continue
        if "♪" in c.text:
            if i is not None and c.start - cues[last].end > config.SONG_MAX_GAP:
                runs.append((i, last))
                i = None
            if i is None:
                i = k
            last = k
    if i is not None:
        runs.append((i, last))
    return runs


def excluded_ranges(cues: list[Cue], duration: float, settings: Settings | None = None) -> list[tuple[float, float]]:
    """Opening theme (and anything before it), closing credits song, and lines matching the library's skip patterns."""
    lib = settings or Settings()
    out = []
    if lib.skip_theme_songs:
        runs = [r for r in _song_runs(cues) if r[1] - r[0] + 1 >= config.SONG_MIN_CUES]
        if lib.theme_match:  # only the song with the theme's lyrics, not other music near the start/end
            pat = re.compile(lib.theme_match, re.I)
            runs = [r for r in runs if any(pat.search(c.text) for c in cues[r[0] : r[1] + 1])]
        if runs and cues[runs[0][0]].start < lib.theme_within:
            first = 0.0 if lib.drop_before_theme else cues[runs[0][0]].start - 0.5
            out.append((first, cues[runs[0][1]].end + 0.5))
        if runs and cues[runs[-1][1]].end > duration - lib.theme_within:
            out.append((cues[runs[-1][0]].start - 0.5, duration))
    if lib.skip_re:
        out += [(c.start - 0.3, c.end + 0.3) for c in cues if lib.skip_re.search(c.text)]
    return out


def _excluded(span: "Span", ranges: list[tuple[float, float]]) -> bool:
    length = max(span.end - span.start, 1e-6)
    return any(min(span.end, b) - max(span.start, a) > 0.5 * length for a, b in ranges)


SENTENCE_END_RE = re.compile(r"""[.?!…"')\]♪]$|--$""")
SPEAKER_RE = re.compile(r"^[A-Z][A-Z .'-]+:")


def merge_cues(cues: list[Cue]) -> list[Cue]:
    """Joins cues that split one sentence (e.g. "I'm going to watch the dog" + "and earn ten dollars.")."""
    out: list[Cue] = []
    for c in cues:
        prev = out[-1] if out else None
        if (
            prev is not None
            and not SENTENCE_END_RE.search(prev.text)
            and not SPEAKER_RE.match(c.text)
            and c.start - prev.end <= config.MERGE_MAX_GAP
            and c.end - prev.start <= config.MERGE_MAX_LEN
        ):
            speakers = [x for x in dict.fromkeys((prev.speaker or "").split(" & ") + (c.speaker or "").split(" & ")) if x]
            out[-1] = Cue(prev.start, max(prev.end, c.end), f"{prev.text} {c.text}",
                          " & ".join(speakers) or None, prev.notes + c.notes)
        else:
            out.append(c)
    return out


def _cuts_between(cuts: list[float], lo: float, hi: float) -> list[float]:
    return cuts[bisect.bisect_left(cuts, lo) : bisect.bisect_right(cuts, hi)]


def _voice(c: Cue, nxt: Cue | None, speech: list[tuple[float, float]]) -> tuple[float | None, float | None, float | None, float]:
    """(voice onset, voice end, next line's voice onset, end of the speech before this line's onset) from VAD.

    Any of the first three is None when the audio doesn't show a clear boundary (e.g. rapid overlapping exchanges).
    """
    starts = [seg[0] for seg in speech]
    on = None
    k = bisect.bisect_left(starts, c.start - 0.3)
    if k < len(speech) and speech[k][0] <= c.start + 0.8:
        on = speech[k][0]
    before_on = speech[k - 1][1] if k > 0 else 0.0

    nxt_on = None
    if nxt is not None:
        j = bisect.bisect_left(starts, max(nxt.start - 0.3, c.start + 0.2))
        if j < len(speech) and speech[j][0] <= nxt.start + 1.0:
            nxt_on = speech[j][0]
    # This line's voice ends at the end of the last segment before the next line's voice (or within ~1s of its subtitle).
    limit = nxt_on if nxt_on is not None else c.end + 1.0
    m = bisect.bisect_left(starts, limit) - 1
    off = speech[m][1] if m >= 0 and speech[m][1] > c.start else None
    return on, off, nxt_on, before_on


def dialogue_window(cues: list[Cue], i: int, cuts: list[float], duration: float,
                    speech: list[tuple[float, float]] | None = None) -> tuple[float, float]:
    """Deterministic in/out points for a line from voice activity, subtitle neighbours, and shot cuts.

    Subtitle times say when text is on screen, not when someone speaks (most cues run until the next
    one appears), so where VAD finds clear boundaries the clip runs from just before the voice starts
    to just after it stops, never into the next voice. A real pause after the line is where reactions
    are, so the clip runs on into it, preferably ending on a cut. A short reaction shot cutting in as
    the line ends is kept. Without clear voice boundaries it falls back to subtitle timing.
    """
    c = cues[i]
    nxt = cues[i + 1] if i + 1 < len(cues) else None
    prev_end = cues[i - 1].end if i > 0 else 0.0
    next_start = nxt.start if nxt else duration
    on, off, nxt_on, before_on = _voice(c, nxt, speech) if speech else (None, None, None, 0.0)

    # Start: just before the voice (or the subtitle), or on the cut opening the shot, never into the previous line.
    if on is not None:
        floor = min(on, max(0.0, before_on + 0.02))
        start = max(floor, on - config.VOICE_PAD)
        anchor = on
    else:
        floor = max(0.0, prev_end)
        start = max(floor, c.start - config.PAD_BEFORE)
        anchor = c.start + config.SPEECH_LAG
    lead = _cuts_between(cuts, max(floor, anchor - config.LEAD_MAX), anchor)
    if lead:
        start = lead[-1]

    # End.
    hard = (nxt_on - 0.05) if nxt_on is not None else min(duration, next_start + config.SPEECH_LAG - 0.05)
    line_end = off if off is not None else c.end
    gap = (nxt_on if nxt_on is not None else next_start + config.SPEECH_LAG) - line_end
    if off is None and next_start - c.end <= config.BACK_TO_BACK:
        end = hard  # no audio boundary, no subtitle gap: stop just before the next voice should begin
    elif gap < config.PAUSE_MIN:
        end = min(line_end + config.VOICE_PAD, hard)
    else:
        end = min(line_end + config.TAIL_MAX, hard)
        tail = _cuts_between(cuts, line_end + config.VOICE_PAD, end)
        if tail:
            end = tail[-1]
        else:
            after = _cuts_between(cuts, end, min(end + config.SNAP_WINDOW, hard))
            if after:
                end = after[0]
    # Reaction shot: a cut as the line ends into a short shot where nobody speaks.
    at_end = _cuts_between(cuts, line_end - 0.2, line_end + 0.8)
    if at_end:
        nxt_cut = _cuts_between(cuts, at_end[0] + 0.01, at_end[0] + config.REACTION_SHOT_MAX)
        if nxt_cut and nxt_cut[0] <= hard + 0.05:
            end = max(end, nxt_cut[0])

    end = max(end, line_end, start + config.MIN_MOMENT)
    return start, min(end, start + config.MAX_MOMENT, duration)


def build_moments(raw_cues: list[Cue], cuts: list[float], duration: float,
                  speech: list[tuple[float, float]] | None = None, settings: Settings | None = None) -> list[Span]:
    spans: list[Span] = []
    cues = merge_cues(raw_cues)

    for i, c in enumerate(cues):
        start, end = dialogue_window(cues, i, cuts, duration, speech)
        prev_line = cues[i - 1].text if i > 0 else ""
        next_line = cues[i + 1].text if i + 1 < len(cues) else ""
        spans.append(Span(start, end, "dialogue", c.text, " ".join(x for x in (prev_line, c.text, next_line) if x),
                          c.speaker, " ".join(c.notes) or None))

    # Silent moments: stretches outside every (padded) dialogue moment, split at shot cuts and capped in length.
    gaps = []
    t = 0.0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start - t >= config.SILENT_MIN_GAP:
            gaps.append((t, s.start))
        t = max(t, s.end)
    if duration - t >= config.SILENT_MIN_GAP:
        gaps.append((t, duration))

    for g0, g1 in gaps:
        prev_line = next((c.text for c in reversed(cues) if c.end <= g0 + 0.01), "")
        next_line = next((c.text for c in cues if c.start >= g1 - 0.01), "")
        context = " ".join(x for x in (prev_line, "[silence]", next_line) if x)
        bounds = [g0] + [x for x in cuts if g0 < x < g1] + [g1]
        for a, b in zip(bounds, bounds[1:]):
            s = a
            while b - s >= 1.0:
                e = min(b, s + config.SILENT_CHUNK)
                if b - e < 1.0:
                    e = b
                spans.append(Span(s, e, "silent", "", context))
                s = e

    # A silent moment right after a line inherits its stage directions: "He jumps around" happens in that pause.
    dialogue = sorted((s for s in spans if s.kind == "dialogue"), key=lambda s: s.end)
    ends = [s.end for s in dialogue]
    for s in spans:
        if s.kind == "silent":
            k = bisect.bisect_right(ends, s.start + 0.05) - 1
            if k >= 0 and dialogue[k].notes and s.start - dialogue[k].end <= 1.0:
                s.notes = dialogue[k].notes

    ranges = excluded_ranges(raw_cues, duration, settings)
    spans = [s for s in spans if not _excluded(s, ranges)]
    spans.sort(key=lambda s: s.start)
    return spans

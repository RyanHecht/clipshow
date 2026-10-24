"""Speaker-labelled transcripts: a plain sidecar format, and alignment onto subtitle cues.

Transcript format (UTF-8 text, one item per line; any source can produce it):

    # Episode Title            optional; marks where an episode starts (double episodes have two)
    Speaker: what they say     dialogue with a speaker
    what they say              dialogue, speaker unknown
    [stage direction]          what happens on screen
    ---                        optional scene break (ignored)

Transcripts have no timestamps, so alignment is by text: the transcript's words are matched to the
subtitles' words in order (like a diff). Each cue takes the speaker of the transcript line most of its
matched words came from, and stage directions attach to the cue of the line they follow.
"""

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Cue, EpisodeFile
from .titles import Settings, Title

SPEAKER_LINE_RE = re.compile(r"^([^:\[\]]{1,40}?):\s+(.+)$")
SUB_SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 .'&-]{0,30}):\s*")
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Line:
    kind: str  # 'line' | 'action' | 'break'
    text: str
    speaker: str | None = None


@dataclass
class CueInfo:
    speakers: list[str] = field(default_factory=list)  # in order; usually one
    actions: list[str] = field(default_factory=list)  # stage directions around this cue
    confidence: float = 0.0  # share of the cue's words matched to the transcript (1.0 for subtitle-labelled)


def parse(text: str, settings: Settings | None = None) -> list[Line]:
    settings = settings or Settings()
    out: list[Line] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s == "---":
            out.append(Line("break", ""))
            continue
        if s.startswith("[") and s.endswith("]"):
            out.append(Line("action", s[1:-1].strip()))
            continue
        m = SPEAKER_LINE_RE.match(s)
        if m:
            name = re.sub(r"\s*(?:\([^)]*\)|#\d+)$", "", m.group(1).strip()) or m.group(1).strip()  # "D.W. (V.O.)", "Kid #2"
            if name.isupper():  # "MUFFY" -> "Muffy", so one person isn't two speakers
                name = name.title()
            out.append(Line("line", m.group(2).strip(), settings.canonical(name)))
        else:
            out.append(Line("line", s))
    return out


def transcript_path(ep: EpisodeFile, transcripts_dir: Path | None = None) -> Path | None:
    """A sidecar next to the video, else one in the title's transcripts folder (where importers write)."""
    cands = [ep.path.with_suffix(".transcript.txt")]
    if transcripts_dir:
        cands.append(transcripts_dir / f"{ep.path.stem}.transcript.txt")
    for cand in cands:
        if cand.exists():
            return cand
    return None


def subtitle_speaker(text: str, settings: Settings | None = None) -> tuple[str | None, str]:
    """Splits an SDH speaker label ("ARTHUR: Hi.") off a cue's text."""
    m = SUB_SPEAKER_RE.match(text)
    if m and len(m.group(1).split()) <= 3:
        return (settings or Settings()).canonical(m.group(1).strip().title()), text[m.end():]
    return None, text


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("'", ""))


def align(cues: list[Cue], lines: list[Line], settings: Settings | None = None, min_confidence: float = 0.5) -> list[CueInfo]:
    """Speaker and stage directions for each cue (same order as cues)."""
    infos = [CueInfo() for _ in cues]

    sub_words, sub_owner = [], []
    for ci, c in enumerate(cues):
        sp, text = subtitle_speaker(c.text, settings)
        if sp:  # the subtitles say who it is: trust that over alignment
            infos[ci].speakers, infos[ci].confidence = [sp], 1.0
        for w in _words(text):
            sub_words.append(w)
            sub_owner.append(ci)

    dialogue = [i for i, ln in enumerate(lines) if ln.kind == "line"]
    tr_words, tr_owner = [], []
    for li in dialogue:
        for w in _words(lines[li].text):
            tr_words.append(w)
            tr_owner.append(li)

    votes: list[dict[int, int]] = [{} for _ in cues]
    sm = difflib.SequenceMatcher(None, sub_words, tr_words, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            ci, li = sub_owner[a + k], tr_owner[b + k]
            votes[ci][li] = votes[ci].get(li, 0) + 1

    cue_words = [0] * len(cues)
    for ci in sub_owner:
        cue_words[ci] += 1
    line_to_last_cue: dict[int, int] = {}
    for ci, v in enumerate(votes):
        if not v:
            continue
        for li in v:
            line_to_last_cue[li] = ci
        matched = sum(v.values())
        conf = matched / max(cue_words[ci], 1)
        if infos[ci].confidence >= 1.0 or conf < min_confidence:
            continue
        # Lines this cue covers, in transcript order, each with enough of the cue to count.
        speakers = []
        for li in sorted(v):
            sp = lines[li].speaker
            if sp and v[li] >= max(1, 0.25 * matched) and sp not in speakers:
                speakers.append(sp)
        infos[ci].speakers, infos[ci].confidence = speakers, conf

    # Stage directions: attach to the cue of the dialogue line before them (what the action reacts to);
    # at a scene start (after a break, or before any line), to the next line's first cue.
    last_line = None
    pending_start: list[str] = []
    for li, ln in enumerate(lines):
        if ln.kind == "break":
            last_line = None
        elif ln.kind == "line":
            if pending_start:
                first = next((ci for ci, v in enumerate(votes) if li in v), None)
                if first is not None:  # else carry them to the next line that aligned
                    infos[first].actions.extend(pending_start)
                    pending_start = []
            last_line = li
        elif last_line is not None and last_line in line_to_last_cue:
            infos[line_to_last_cue[last_line]].actions.append(ln.text)
        else:
            pending_start.append(ln.text)
    return infos


def annotate(cues: list[Cue], ep: EpisodeFile, title: Title) -> list[Cue]:
    """Cues with speaker and stage directions filled in from the subtitles' own labels and, when there is one, a transcript."""
    path = transcript_path(ep, title.transcripts)
    lines = parse(path.read_text(encoding="utf-8"), title.settings) if path else []
    infos = align(cues, lines, title.settings)
    return [Cue(c.start, c.end, c.text, " & ".join(i.speakers) or None, tuple(i.actions)) for c, i in zip(cues, infos)]

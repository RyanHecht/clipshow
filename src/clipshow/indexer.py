"""Index pipeline: ingest -> segment -> keyframes -> embeddings, per episode and incremental.

Two halves, so indexing can run on another machine than the web app:
  compute  prepare() (subtitles, one decoding pass, voice activity, moments, thumbnails) and embed():
           runs on a worker, several episodes at a time in separate processes.
  store    store_result(): writes an episode's moments, vectors, subtitles and thumbnails into a title.
           Only the web app does this (workers upload results to it), so it is the database's sole writer.
index_path() runs both in one process, for the CLI.
"""

import hashlib
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import embed, frames, ingest, segment, store, transcripts, vad, video
from .titles import Title


@dataclass
class Prepared:
    ep: ingest.EpisodeFile
    duration: float
    srt: str
    cuts: list[float]
    speech: list[tuple[float, float]]
    spans: list[segment.Span]
    thumbs: dict[int, list[str]]
    visual_frames: list[list[bytes]]
    n_cues: int
    seconds: float


@dataclass
class EpisodeResult:
    """Everything the web app needs to store an indexed episode (thumbnails travel separately)."""
    path: str
    label: str
    season: int | None
    episode: int | None
    duration: float
    signature: str
    file_signature: str
    srt_text: str
    cuts: list[float]
    speech: list[tuple[float, float]]
    spans: list[dict]
    thumb_dir: str
    thumbs: list[str]  # one relative thumbnail path per span
    n_cues: int
    seconds: float
    vectors: dict[str, np.ndarray] = field(default_factory=dict)  # 'text', 'visual' (per span), 'notes' (spans with notes)

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("vectors")
        return d

    @classmethod
    def from_meta(cls, meta: dict, vectors: dict[str, np.ndarray]) -> "EpisodeResult":
        return cls(**{**meta, "speech": [tuple(x) for x in meta["speech"]]}, vectors=vectors)


def thumb_dir(ep: ingest.EpisodeFile) -> str:
    return f"{ep.label}-{hashlib.md5(str(ep.path).encode()).hexdigest()[:8]}"


def prepare(title: Title, ep: ingest.EpisodeFile, shots: bool, cached_cuts: list[float] | None,
            cached_speech: list[tuple[float, float]] | None, gpu: bool = False) -> Prepared:
    """Runs in a worker process: everything for one episode that doesn't need the GPU or the database."""
    t0 = time.monotonic()
    info = ingest.probe(ep.path)
    dur = ingest.duration(info)
    srt = ingest.subtitle_file(ep, info, title.subs)
    if srt is None:
        raise ingest.NoSubtitles()
    cues = transcripts.annotate(ingest.load_cues(srt), ep, title)
    # One decoding pass gives both the shot cuts (unless cached) and the frames thumbnails are cut from.
    analysis = video.analyze(ep.path, detect_cuts=shots and cached_cuts is None, gpu=gpu)
    cuts = cached_cuts if cached_cuts is not None else (analysis.cuts if shots else [])
    speech = cached_speech if cached_speech is not None else vad.speech_segments(ep.path)
    spans = segment.build_moments(cues, cuts, dur, speech, title.settings)
    thumbs = write_thumbs(analysis, title.thumbs, thumb_dir(ep), spans)
    visual_frames = moment_frames(analysis, spans)
    return Prepared(ep, dur, str(srt), cuts, speech, spans, thumbs, visual_frames, len(cues), time.monotonic() - t0)


def write_thumbs(analysis: video.Analysis, thumbs_root: Path, episode_dir: str, spans: list[segment.Span]) -> dict[int, list[str]]:
    """{span index: [thumb path relative to thumbs_root]}: the moment's middle frame, from the frames kept while decoding."""
    out_dir = thumbs_root / episode_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = {}
    for n, s in enumerate(spans):
        rel = f"{episode_dir}/{n}.jpg"
        (thumbs_root / rel).write_bytes(analysis.thumb_at((s.start + s.end) / 2))
        thumbs[n] = [rel]
    return thumbs


def moment_frames(analysis: video.Analysis, spans: list[segment.Span]) -> list[list[bytes]]:
    """Several frames per moment (JPEG bytes) for the visual embedding; only the middle one is kept on disk."""
    return [[analysis.thumb_at(t) for t in frames.moment_frame_times(s.start, s.end)] for s in spans]


def embed_episode(prep: Prepared) -> EpisodeResult:
    """Embeddings for a prepared episode (GPU if there is one), packaged with everything needed to store it."""
    ep, spans = prep.ep, prep.spans
    # Dialogue moments embed their own line; silent ones embed the lines around them (what they react to).
    vectors = {
        "text": embed.embed_texts([s.text or s.context for s in spans]).astype(np.float16),
        "visual": embed.embed_moment_frames(prep.visual_frames).astype(np.float16),
    }
    notes = [s.notes for s in spans if s.notes]
    if notes:
        vectors["notes"] = embed.embed_texts(notes).astype(np.float16)
    return EpisodeResult(
        str(ep.path), ep.label, ep.season, ep.episode, prep.duration, ep.signature, ep.file_signature,
        Path(prep.srt).read_text(encoding="utf-8", errors="replace"), prep.cuts, prep.speech,
        [asdict(s) for s in spans], thumb_dir(ep), [prep.thumbs[n][0] for n in range(len(spans))],
        prep.n_cues, prep.seconds, vectors)


def store_result(conn, title: Title, res: EpisodeResult, log=print) -> int:
    """Writes an indexed episode into a title. Its thumbnails must already be under title.thumbs/res.thumb_dir."""
    spans = [segment.Span(**s) for s in res.spans]
    title.ensure_dirs()
    srt = title.subs / f"{res.thumb_dir}.srt"  # the title keeps its own copy, for captions on clips
    srt.write_text(res.srt_text, encoding="utf-8")

    episode_id = store.upsert_episode(conn, res.path, res.label, res.season, res.episode, res.duration, res.signature, str(srt))
    store.set_analysis(conn, episode_id, res.cuts, res.speech, res.file_signature)
    old_dirs = {Path(f).parent.name for r in conn.execute("SELECT frames FROM moments WHERE episode_id = ?", (episode_id,))
                for f in store.json.loads(r[0])}
    kept = store.saved_descriptions(conn, episode_id)
    store.clear_episode_moments(conn, episode_id)
    ids = [store.insert_moment(conn, episode_id, s.start, s.end, s.kind, s.text, s.context, s.speaker, s.notes) for s in spans]
    reused = store.restore_descriptions(conn, kept, ids, spans)
    for mid, thumb in zip(ids, res.thumbs):
        store.set_frames(conn, mid, [thumb])
    store.put_vectors(conn, "text", ids, res.vectors["text"])
    store.put_vectors(conn, "visual", ids, res.vectors["visual"])
    noted = [mid for mid, s in zip(ids, spans) if s.notes]
    if noted:
        store.put_vectors(conn, "notes", noted, res.vectors["notes"])
    store.mark_indexed(conn, episode_id)
    conn.commit()

    for d in old_dirs - {res.thumb_dir}:  # thumbnails from an older layout
        shutil.rmtree(title.thumbs / d, ignore_errors=True)
    silent = sum(s.kind == "silent" for s in spans)
    log(f"{res.label}: {res.duration / 60:.1f} min, {res.n_cues} cues, {len(res.cuts)} cuts -> {len(spans)} moments "
        f"({silent} silent), kept {reused} descriptions  [prep {res.seconds:.0f}s]")
    return episode_id


def todo_episodes(eps: list[ingest.EpisodeFile], known: dict[str, dict], force: bool = False) -> list[ingest.EpisodeFile]:
    """Episodes that aren't indexed yet, or changed since. known: {path: {"signature", "indexed"}} from the title."""
    return [ep for ep in eps if force or not (known.get(str(ep.path), {}).get("indexed")
                                              and known[str(ep.path)].get("signature") == ep.signature)]


def prepare_many(title: Title, todo: list[tuple], shots: bool, workers: int):
    """Yields (episode, Prepared or exception) as episodes finish, several at a time.

    todo: [(episode, cached cuts or None, cached speech or None)].
    """
    # spawn, not fork: the parent may already have the GPU initialised.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        # Alternate GPU and CPU decoding: they're separate hardware, so together they decode ~1.5x faster than either.
        gpu = video.vaapi_available()
        futs = {pool.submit(prepare, title, ep, shots, cuts if shots else None, speech, gpu and i % 2 == 0): ep
                for i, (ep, cuts, speech) in enumerate(todo)}
        for fut in as_completed(futs):
            try:
                yield futs[fut], fut.result()
            except Exception as e:  # noqa: BLE001 - reported per episode
                yield futs[fut], e


def index_path(title: Title, root: Path, shots: bool = True, force: bool = False, workers: int = 6, log=print) -> list[int]:
    """Indexes every episode under root into the title, all in this process (for the CLI).
    Returns the episode ids touched (new or re-indexed)."""
    title.ensure_dirs()
    conn = store.connect(title.db_path)
    eps = ingest.find_episodes(root)
    log(f"found {len(eps)} video file(s)")
    known = store.episode_states(conn)
    todo = []
    for ep in todo_episodes(eps, known, force):
        cuts, speech = store.cached_analysis(store.get_episode(conn, str(ep.path)), ep.file_signature)
        todo.append((ep, cuts, speech))
    log(f"{len(eps) - len(todo)} already indexed, {len(todo)} to do ({sum(t[1] is not None for t in todo)} with cached shot cuts)")

    touched: list[int] = []
    t_start = time.monotonic()
    for done, (ep, prep) in enumerate(prepare_many(title, todo, shots, workers), 1):
        if isinstance(prep, ingest.NoSubtitles):
            log(f"{ep.label}: no subtitles, skipping")
            continue
        if isinstance(prep, Exception):
            log(f"{ep.label}: FAILED ({prep})")
            continue
        try:
            touched.append(store_result(conn, title, embed_episode(prep), log))
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the run
            conn.rollback()
            log(f"{ep.label}: FAILED ({e})")
        rate = (time.monotonic() - t_start) / done
        log(f"  [{done}/{len(todo)}] ~{rate * (len(todo) - done) / 60:.0f} min left")
    return touched


def apply_annotations(title: Title, log=print) -> tuple[int, int]:
    """Attaches speakers and stage directions to already-indexed moments without re-indexing.

    Rebuilds each episode's moments from its saved shot cuts and voice activity (no video decoding)
    and copies speaker/notes onto the stored moment with the same window.
    """
    conn = store.connect(title.db_path)
    rows = conn.execute("SELECT * FROM episodes WHERE indexed_at IS NOT NULL AND cuts IS NOT NULL AND speech IS NOT NULL").fetchall()
    matched = total = 0
    for row in rows:
        ep = next(iter(ingest.find_episodes(Path(row["path"]))), None)
        if ep is None or not row["sub_path"] or not Path(row["sub_path"]).exists():
            continue
        cuts, speech = store.cached_analysis(row, ep.file_signature)
        if cuts is None:
            continue
        cues = transcripts.annotate(ingest.load_cues(Path(row["sub_path"])), ep, title)
        spans = segment.build_moments(cues, cuts, row["duration"], speech, title.settings)
        by_window = {(s.kind, round(s.start, 2), round(s.end, 2)): s for s in spans}
        noted = []
        for m in conn.execute("SELECT id, kind, start, end FROM moments WHERE episode_id = ?", (row["id"],)).fetchall():
            total += 1
            s = by_window.get((m["kind"], round(m["start"], 2), round(m["end"], 2)))
            if s is None:
                continue
            matched += 1
            store.set_annotations(conn, m["id"], s.speaker, s.notes)
            if s.notes:
                noted.append((m["id"], s.notes))
        conn.execute("DELETE FROM vectors WHERE kind = 'notes' AND moment_id IN (SELECT id FROM moments WHERE episode_id = ?)", (row["id"],))
        if noted:
            store.put_vectors(conn, "notes", [m for m, _ in noted], embed.embed_texts([n for _, n in noted]))
        conn.commit()
        log(f"{row['label']}: {sum(1 for s in spans if s.speaker)} moments with a speaker, {len(noted)} with stage directions")
    return matched, total

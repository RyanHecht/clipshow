"""Hybrid search: FTS5 + dialogue / stage-direction / visual / description embeddings, fused by weighted RRF.

Speaker names in the query ("Muffy smug") restrict results to that character's moments when speaker
data exists (from subtitle labels or aligned transcripts).
"""

import re
import threading

import numpy as np

from . import config, embed, postable, store
from .titles import Settings
from .store import Moment


def db_version(conn) -> tuple[int, int]:
    """Changes whenever the database does: data_version for other connections' commits (indexing,
    `transcripts apply`), total_changes for this connection's own writes. Far cheaper than counting rows."""
    return conn.execute("PRAGMA data_version").fetchone()[0], conn.total_changes


class VectorCache:
    """Vector matrices loaded from SQLite per database (one connection per title), reloaded when the database changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, str], tuple[tuple, np.ndarray, np.ndarray]] = {}

    def get(self, conn, kind: str) -> tuple[np.ndarray, np.ndarray]:
        version = db_version(conn)
        key = (id(conn), kind)
        with self._lock:
            hit = self._cache.get(key)
            if hit is None or hit[0] != version:
                ids, mat = store.load_vectors(conn, kind)
                self._cache[key] = (version, ids, mat)
            return self._cache[key][1], self._cache[key][2]


_vectors = VectorCache()


def _fts(conn, query: str, n: int, extra: str = "", speaker_clause: str = "") -> tuple[list[int], set[int]]:
    """(ranked ids, ids whose spoken line contains the query as an exact phrase)."""
    tokens = re.findall(r"[\w']+", query.lower())
    if not tokens:
        return [], set()
    extra_tokens = [t for t in re.findall(r"[\w']+", extra.lower()) if t not in tokens]
    phrase = '"' + " ".join(tokens).replace('"', "") + '"'
    all_words = " ".join(f'"{t}"' for t in tokens)
    # Exact phrase in the spoken line ranks first, then moments containing every word (line, neighbours,
    # or stage directions), by BM25. Requiring every word keeps descriptive queries like "eye roll" from
    # matching any line that merely says "eye".
    ids: list[int] = []
    phrase_ids: set[int] = set()
    matches = [f"text : {phrase}", f"{{text context notes}} : ({all_words})"]
    if extra_tokens:  # slang expansions ("lol" -> laughing): any of them, in the spoken line only
        matches.append("text : (" + " OR ".join(f'"{t}"' for t in extra_tokens) + ")")
    for i, match in enumerate(matches):
        if speaker_clause:
            match = f"({match}) AND {speaker_clause}"
        for (rid,) in conn.execute(
            "SELECT rowid FROM moments_fts WHERE moments_fts MATCH ? ORDER BY bm25(moments_fts, 4.0, 1.0, 0.5, 2.0) LIMIT ?",
            (match, n),
        ):
            if i == 0:
                phrase_ids.add(rid)
            if rid not in ids:
                ids.append(rid)
    return ids[:n], phrase_ids


# Role words that show up as transcript speakers but aren't characters anyone searches for by name.
GENERIC_SPEAKERS = {
    "all", "everyone", "kids", "class", "crowd", "audience", "announcer", "narrator", "woman", "man", "boy", "girl",
    "both", "voice", "customer", "others", "group", "students", "people", "children", "chorus", "singers", "together",
}


class SpeakerCache:
    """Per database: query-token variants of every searchable speaker name, and each speaker's moment ids.

    Built in one pass over the moments and rebuilt only when the database changes (see db_version), so
    "D.W. ugh" finds D.W.'s moments with a lookup instead of scanning every row per query.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict[int, tuple[tuple, dict, dict]] = {}  # per connection (= per title)

    def get(self, conn, settings: Settings) -> tuple[dict[tuple[str, ...], str], dict[str, np.ndarray]]:
        version = (*db_version(conn), id(settings))
        with self._lock:
            entry = self._entries.get(id(conn))
            if entry is None or entry[0] != version:
                entry = (version, *self._build(conn, settings))
                self._entries[id(conn)] = entry
            return entry[1], entry[2]

    @staticmethod
    def _build(conn, settings: Settings) -> tuple[dict[tuple[str, ...], str], dict[str, np.ndarray]]:
        ids_by_name: dict[str, list[int]] = {}
        for mid, sp in conn.execute("SELECT id, speaker FROM moments WHERE speaker IS NOT NULL"):
            for name in sp.split(" & "):
                ids_by_name.setdefault(name, []).append(mid)
        known = [name for name, ids in ids_by_name.items() if len(ids) >= 5 and name.lower() not in GENERIC_SPEAKERS]
        variants = {}  # token tuple -> canonical speaker name
        for name in known:
            for alias in [name, *settings.aliases.get(name, [])]:
                toks = tuple(re.findall(r"[a-z0-9]+", alias.lower()))
                if toks:
                    variants[toks] = name
                    if len(toks) > 1 and all(len(t) == 1 for t in toks):  # "D.W." also typed as "dw"
                        variants[("".join(toks),)] = name
        return variants, {name: np.array(ids_by_name[name], dtype=np.int64) for name in known}


_speakers = SpeakerCache()
_DEFAULT_SETTINGS = Settings()


def find_speakers(conn, query: str, settings: Settings | None = None) -> tuple[list[str], str]:
    """(speaker names mentioned in the query, the query without them)."""
    variants, _ = _speakers.get(conn, settings or _DEFAULT_SETTINGS)
    words = re.findall(r"[a-z0-9]+", query.lower())
    found, keep, i = [], [], 0
    while i < len(words):
        for size in (3, 2, 1):
            name = variants.get(tuple(words[i : i + size]))
            if name:
                if name not in found:
                    found.append(name)
                i += size
                break
        else:
            keep.append(words[i])
            i += 1
    return found, " ".join(keep) if found else query


def _dense(conn, kind: str, q: np.ndarray, n: int, allowed: np.ndarray | None = None,
           sims_out: dict[int, float] | None = None) -> list[int]:
    """Ids of the n nearest vectors (by cosine). If sims_out is given, it receives {id: similarity} for them."""
    ids, mat = _vectors.get(conn, kind)
    if len(ids) == 0:
        return []
    sims = mat @ q
    if allowed is not None:
        sims = np.where(np.isin(ids, allowed), sims, -np.inf)
        n = min(n, int(np.isfinite(sims).sum()))
        if n == 0:
            return []
    top = np.argpartition(-sims, min(n, len(sims) - 1))[:n]
    top = top[np.argsort(-sims[top])]
    if sims_out is not None:
        sims_out.update((int(ids[i]), float(sims[i])) for i in top)
    return [int(ids[i]) for i in top]


def search(conn, query: str, k: int = 12, weights: dict[str, float] | None = None, kinds: set[str] | None = None,
           settings: Settings | None = None) -> list[Moment]:
    w = {**config.DEFAULT_WEIGHTS, **(weights or {})}
    speakers, rest = find_speakers(conn, query, settings)
    if speakers:
        found = _search(conn, rest, k, w, kinds, speakers, settings)
        if found:
            return found
    return _search(conn, query, k, w, kinds, [], settings)


def _search(conn, query: str, k: int, w: dict[str, float], kinds: set[str] | None, speakers: list[str],
            settings: Settings | None = None) -> list[Moment]:
    n = config.CANDIDATES_PER_SIGNAL
    rankings: dict[str, list[int]] = {}
    allowed = None
    speaker_clause = ""
    if speakers:
        _, ids_by_name = _speakers.get(conn, settings or _DEFAULT_SETTINGS)
        allowed = np.unique(np.concatenate([ids_by_name.get(s, np.zeros(0, np.int64)) for s in speakers]))
        if len(allowed) == 0:
            return []
        speaker_clause = "speaker : (" + " OR ".join('"' + s.replace('"', "") + '"' for s in speakers) + ")"
    phrase_ids: set[int] = set()
    text_sims: dict[int, float] = {}
    if query.strip():
        expanded = postable.expand(query)
        if w["fts"] > 0:
            rankings["fts"], phrase_ids = _fts(conn, query, n, extra=expanded[len(query):], speaker_clause=speaker_clause)
        if w["text"] > 0 or w["desc"] > 0 or w["notes"] > 0:
            qt = embed.embed_query_text(expanded)
            for sig in ("text", "desc", "notes"):
                if w[sig] > 0:
                    rankings[sig] = _dense(conn, sig, qt, n, allowed, text_sims if sig == "text" else None)
        if w["visual"] > 0:
            rankings["visual"] = _dense(conn, "visual", embed.embed_query_visual(query), n, allowed)
    elif allowed is not None:
        # Just a name ("D.W."): their moments, best reactions first (ranked by the postable prior below).
        rankings["speaker"] = [int(i) for i in allowed]
        w = {**w, "speaker": 1.0}

    scores: dict[int, float] = {}
    hits: dict[int, dict[str, int]] = {}
    for sig, ids in rankings.items():
        for rank, mid in enumerate(ids):
            r = 0 if sig == "speaker" else rank  # a bare name has no ranking of its own
            scores[mid] = scores.get(mid, 0.0) + w[sig] / (config.RRF_K + r + 1)
            hits.setdefault(mid, {})[sig] = rank + 1

    ordered = sorted(scores, key=scores.get, reverse=True)
    pool = ordered if kinds or "speaker" in rankings else ordered[: k * 15]
    moments = store.get_moments(conn, pool)
    # A line that means almost exactly what a quote-length query says is a misremembered quote ("that sign
    # won't stop me" for "can't"): rank it like an exact phrase match.
    if len(re.findall(r"[\w']+", query)) >= config.QUOTE_MIN_WORDS:
        for mid, m in moments.items():
            if m.kind == "dialogue" and text_sims.get(mid, 0.0) >= config.QUOTE_SIM and mid not in phrase_ids:
                phrase_ids.add(mid)
                scores[mid] += w["fts"] / (config.RRF_K + 1)
                hits[mid]["quote"] = 1
    # Prefer moments that work as a standalone reaction GIF (short lines, exclamations, gasps and laughs).
    a = min(1.0, 0.7 * w["postable"])
    for mid, m in moments.items():
        if mid not in phrase_ids:  # a quote is what was asked for, however long
            scores[mid] *= (1 - a) + a * postable.score(m.text, m.kind)
    pool = sorted((mid for mid in pool if mid in moments), key=scores.get, reverse=True)
    out: list[Moment] = []
    for mid in pool:
        m = moments.get(mid)
        if m is None:
            continue
        if kinds and m.kind not in kinds:
            continue
        # Drop moments overlapping a better-ranked one in the same episode.
        if any(o.episode_id == m.episode_id and o.start < m.end and m.start < o.end for o in out):
            continue
        m.score, m.hits = scores[mid], hits[mid]
        out.append(m)
        if len(out) >= k:
            break
    extend_to_quoted_neighbours(conn, query, out)
    return out


# Function words only; reaction words like "oh", "no", "yes" stay meaningful.
STOPWORDS = {
    "a", "an", "the", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with", "is", "am", "are",
    "was", "were", "be", "been", "do", "does", "did", "so", "that", "this", "these", "those", "as", "by", "if",
    "then", "than", "just", "have", "has", "had", "will", "would", "s", "t", "ll", "m", "re", "ve", "d",
}
MAX_EXTENDED = 10.0


def _content_words(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", postable.SPEAKER_RE.sub("", text).lower()) if t not in STOPWORDS}


def extend_to_quoted_neighbours(conn, query: str, moments: list[Moment]) -> None:
    """Sets ds/de on moments whose quoted query continues into the line just before or after.

    Only for quote-like queries: a neighbour is pulled in (its full window) when it adds query words the
    moment's own line lacks and the lines together cover most of the query.
    """
    q = _content_words(query)
    if len(q) < 2:
        return
    for m in moments:
        if m.kind != "dialogue":
            continue
        own = _content_words(m.text) & q
        if not own or own == q:
            continue
        prev = conn.execute(
            "SELECT start, end, text FROM moments WHERE episode_id = ? AND kind = 'dialogue' AND start < ? ORDER BY start DESC LIMIT 1",
            (m.episode_id, m.start)).fetchone()
        nxt = conn.execute(
            "SELECT start, end, text FROM moments WHERE episode_id = ? AND kind = 'dialogue' AND start > ? ORDER BY start LIMIT 1",
            (m.episode_id, m.start)).fetchone()
        sides = []  # (new words, side, row)
        for side, row in (("prev", prev), ("next", nxt)):
            if row is not None:
                new = (_content_words(row["text"]) & q) - own
                if new:
                    sides.append((len(new), side, row))
        # Add the side bringing the most new query words first, then the other if it still fits.
        covered, start, end = set(own), m.start, m.end
        for _, side, row in sorted(sides, key=lambda x: -x[0]):
            s2, e2 = (min(start, row["start"]), end) if side == "prev" else (start, max(end, row["end"]))
            if e2 - s2 > MAX_EXTENDED:
                continue
            covered |= _content_words(row["text"]) & q
            start, end = s2, e2
        if (start, end) != (m.start, m.end) and len(covered) / len(q) >= 0.6:
            m.ds, m.de = start - m.start, end - m.end
    # A neighbour now inside an extended clip shouldn't also be listed on its own.
    ext = [(m.episode_id, m.start + m.ds, m.end + m.de, m.id) for m in moments if m.ds or m.de]
    moments[:] = [m for m in moments
                  if not any(e == m.episode_id and a <= m.start and m.end <= b and i != m.id for e, a, b, i in ext)]

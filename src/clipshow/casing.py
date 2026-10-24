"""Sentence case for ALL-CAPS subtitles (common in broadcast captions), using the title's own casing.

Lowercasing "SOOKIE, I NEED COFFEE." would lose the names, so each title's capitalisation is learned
from its mixed-case lines ("Sookie", "Stars Hollow", "I'm"), ignoring sentence-initial words, and then
applied to the shouting ones, with sentence starts capitalised. Display only: the index is untouched.
"""

import re
import threading
from collections import Counter, defaultdict

WORD = re.compile(r"[A-Za-z][A-Za-z']*")
SENTENCE_START = re.compile(r"(^|[.!?…]\s+|^\s*-\s*|\s-\s+|:\s+|[\[(]\s*)([a-z])")
ALWAYS = {"i": "I", "i'm": "I'm", "i'll": "I'll", "i've": "I've", "i'd": "I'd", "ok": "OK", "tv": "TV", "dj": "DJ"}
ALWAYS.update({w.lower(): w for w in (
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April June July August "
    "September October November December Christmas Thanksgiving Halloween Easter Hanukkah God English French "
    "Spanish American America Mr Mrs Ms Dr").split()})


def shouty(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 4 and sum(ch.isupper() for ch in letters) / len(letters) > 0.8


class Casing:
    def __init__(self, texts):
        seen: dict[str, Counter] = defaultdict(Counter)
        for text in texts:
            if not text or shouty(text):
                continue
            for m in WORD.finditer(text):
                before = text[: m.start()].rstrip()
                if not before or before[-1] in ".!?…:-([":  # sentence-initial: capitalised anyway, tells us nothing
                    continue
                seen[m.group().lower()][m.group()] += 1
        self.forms = dict(ALWAYS)
        for key, forms in seen.items():
            # Words in capitals mid-line are emphasis or speaker labels ("OH", "KIRK:"), not their usual spelling.
            forms = Counter({f: n for f, n in forms.items() if not (f.isupper() and len(f) > 1)})
            if not forms:
                continue
            form, n = forms.most_common(1)[0]
            if form != key and n >= 2 and n / sum(forms.values()) >= 0.6 and key not in ALWAYS:
                self.forms[key] = form

    def fix(self, text: str) -> str:
        if not text or not shouty(text):
            return text
        out = WORD.sub(lambda m: self.forms.get(m.group().lower(), m.group().lower()), text)
        return SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), out)


_cache: dict[int, tuple[tuple, Casing]] = {}
_lock = threading.Lock()


def for_db(conn) -> Casing:
    """The title's casing, learned from its subtitles once per database change."""
    version = (conn.execute("PRAGMA data_version").fetchone()[0], conn.total_changes)
    with _lock:
        hit = _cache.get(id(conn))
        if hit and hit[0] == version:
            return hit[1]
    casing = Casing(t for (t,) in conn.execute("SELECT text FROM moments WHERE kind = 'dialogue'"))
    with _lock:
        _cache[id(conn)] = (version, casing)
    return casing

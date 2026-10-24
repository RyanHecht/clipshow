"""How likely a moment is to work as a posted reaction GIF, judged from its subtitle text alone.

Reaction GIFs are short and self-contained: an exclamation, a one-liner, or a sound (gasp, laugh,
groan). Long plot dialogue rarely works out of context. This is a cheap prior for ranking, not a filter.
"""

import re

SPEAKER_RE = re.compile(r"^[A-Z][A-Z .'-]+:\s*")
CUE_RE = re.compile(r"^[\[(]\s*([^)\]]+?)\s*[)\]]$")
WORD_RE = re.compile(r"[A-Za-z']+")

# Sound cues that are a person reacting (stems; matched as substrings of the cue).
REACTION_SOUNDS = (
    "gasp", "sigh", "laugh", "scream", "groan", "cheer", "giggl", "yawn", "cry", "sob", "whimper", "shriek",
    "gulp", "chuckl", "snicker", "applau", "scoff", "humph", "squeal", "wail", "yell", "shout", "clears throat",
    "sniffl", "moan", "whistl", "grumbl", "mutter", "hyperventilat", "stammer", "sputter", "cackl", "howl",
)
INTERJECTIONS = {
    "oh", "ah", "aw", "aww", "wow", "whoa", "yes", "yeah", "yay", "no", "nope", "ugh", "hey", "what", "huh",
    "hmm", "ha", "haha", "oops", "phew", "gross", "ew", "eww", "cool", "great", "thanks", "sorry", "please",
    "never", "help", "awesome", "yikes", "uh", "oh-oh", "uh-oh", "hooray", "wait", "stop", "bye", "hi",
    "hello", "okay", "ok", "fine", "whatever", "really", "seriously", "duh", "boo", "shh", "ow", "ouch",
}


def score(text: str, kind: str) -> float:
    """0..1 prior that the moment works as a standalone reaction GIF."""
    if kind == "silent":
        return 0.45  # could be a great face or nothing; text can't tell
    t = SPEAKER_RE.sub("", text.strip())
    cue = CUE_RE.match(t)
    if cue:
        name = cue.group(1).lower()
        return 0.95 if any(s in name for s in REACTION_SOUNDS) else 0.15
    words = [w.lower() for w in WORD_RE.findall(t)]
    n = len(words)
    if n == 0:
        return 0.2
    p = 1.0 if n <= 3 else 0.85 if n <= 6 else 0.55 if n <= 9 else 0.3 if n <= 14 else 0.15
    if t.endswith("!") or "?!" in t:
        p += 0.1
    if words[0] in INTERJECTIONS:
        p += 0.1
    if "♪" in t:
        p -= 0.3  # song lyrics
    return max(0.05, min(1.0, p))


# Internet shorthand people type into GIF search, mapped to words the subtitles actually use.
SLANG = {
    "lol": "laughing laughs", "lmao": "laughing", "rofl": "laughing", "haha": "laughing laughs", "hehe": "giggling",
    "omg": "gasps oh my gosh", "wtf": "what", "smh": "sighs groans", "ugh": "groans sighs ugh", "meh": "sighs whatever",
    "yay": "cheering yay hooray", "woohoo": "cheering", "sad": "crying sobbing", "cry": "crying sobbing",
    "bored": "yawns", "tired": "yawns", "sleepy": "yawns", "shook": "gasps", "shocked": "gasps", "mind blown": "gasps whoa wow",
    "facepalm": "groans", "eye roll": "sighs whatever", "nope": "no nope", "yep": "yes yeah", "ty": "thank you",
    "thx": "thank you", "bye": "bye goodbye", "hi": "hi hello", "scared": "screams screaming", "angry": "yelling growls",
    "mad": "yelling", "excited": "cheering yay", "awkward": "clears throat", "gross": "gross ew yuck", "ew": "gross ew yuck",
}


def expand(query: str) -> str:
    """Adds subtitle-vocabulary equivalents for slang in the query (the original words are kept)."""
    q = query.lower()
    extra = [v for k, v in SLANG.items() if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", q)]
    return f"{query} {' '.join(extra)}" if extra else query

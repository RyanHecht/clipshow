"""A title's look, taken from its show's Jellyfin artwork: logo, backdrop, and a palette from the art.

The accent is the most vivid colour that covers a meaningful part of the backdrop (or poster). Text
colours are chosen by WCAG contrast, so any show's palette stays readable in light and dark mode.
"""

import colorsys
import io

from PIL import Image

from .jellyfin import Jellyfin
from .titles import Title

DEFAULT_ACCENT = (242, 182, 50)


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def _luminance(rgb) -> float:
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _hls(h, l, s):
    return tuple(c * 255 for c in colorsys.hls_to_rgb(h, l, s))


def _shade_for(bg, h, s, want: float = 4.5, dark_text: bool = True):
    """The lightness of hue h that reaches `want` contrast against bg (darker if dark_text, else lighter)."""
    steps = [i / 100 for i in range(60, -1, -2)] if dark_text else [i / 100 for i in range(40, 101, 2)]
    for l in steps:
        c = _hls(h, l, s)
        if contrast(c, bg) >= want:
            return c
    return (0, 0, 0) if dark_text else (255, 255, 255)


def accent_from(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((96, 96))
    pal = img.quantize(colors=10, method=Image.Quantize.MEDIANCUT)
    counts = sorted(pal.getcolors(), reverse=True)
    colours = pal.getpalette()
    total = sum(n for n, _ in counts)
    best, best_score = None, 0.0
    for n, idx in counts:
        rgb = tuple(colours[idx * 3: idx * 3 + 3])
        h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
        share = n / total
        if share < 0.02 or l < 0.2 or l > 0.85:
            continue
        score = s * (0.4 + min(share, 0.25)) * (1 - abs(l - 0.55))
        if score > best_score:
            best, best_score = rgb, score
    return best if best and best_score > 0.08 else None


def build(title: Title, jf: Jellyfin, series_id: str) -> dict:
    """Downloads the artwork into the title's folder and returns the theme (stored in the hub)."""
    out = title.root / "theme"
    out.mkdir(parents=True, exist_ok=True)
    have = {}
    for kind, width in (("Backdrop", 1600), ("Primary", 600), ("Logo", 800)):
        data = jf.image(series_id, kind, width)
        if data:
            (out / kind.lower()).write_bytes(data)
            have[kind.lower()] = True

    accent = None
    for kind in ("backdrop", "primary"):
        if have.get(kind):
            accent = accent_from((out / kind).read_bytes())
            if accent:
                break
    accent = accent or DEFAULT_ACCENT
    h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in accent))
    s_soft = min(s, 0.45)
    light_bg, dark_bg = _hls(h, 0.95, s_soft * 0.6), _hls(h, 0.09, s_soft * 0.5)
    return {
        "accent": _hex(accent),
        "accent_ink": "#000000" if contrast(accent, (0, 0, 0)) >= contrast(accent, (255, 255, 255)) else "#ffffff",
        "light": {"bg": _hex(light_bg), "text_accent": _hex(_shade_for(light_bg, h, max(s, 0.5)))},
        "dark": {"bg": _hex(dark_bg), "paper": _hex(_hls(h, 0.13, s_soft * 0.45)),
                 "text_accent": _hex(_shade_for(dark_bg, h, max(s, 0.5), dark_text=False))},
        "chrome": _hex(_hls(h, 0.08, s_soft * 0.4)),
        "images": sorted(have),
    }

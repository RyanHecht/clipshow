"""Size-capped cache for rendered clips, GIFs and frames, shared by every title.

A file's mtime records when it was last used (it's touched on every hit), so going over the cap
deletes the least recently used files first, down to 90% of the cap so eviction doesn't run on
every render. The cap is the `render_cache_mb` setting (CLIPSHOW_RENDER_CACHE_MB, default 2048).
Clips someone has starred or shared are never evicted (they still count toward the total).
"""

import os
import threading
from pathlib import Path

from . import config

LOW_WATER = 0.9


class RenderCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._total: int | None = None

    def _files(self) -> list[Path]:
        return [p for p in (config.DATA_ROOT / "titles").glob("*/renders/*") if p.is_file() and not p.name.startswith(".")]

    def limit_bytes(self) -> int:
        from .auth import get_setting

        return int(float(get_setting("render_cache_mb") or 2048) * 1024 * 1024)

    def used(self) -> int:
        with self._lock:
            if self._total is None:
                self._total = sum(p.stat().st_size for p in self._files())
            return self._total

    def hit(self, path: Path) -> None:
        try:
            os.utime(path)
        except OSError:
            pass

    def added(self, path: Path) -> None:
        size = path.stat().st_size
        self.used()
        with self._lock:
            self._total += size
            over = self._total > self.limit_bytes()
        if over:
            self.evict()

    def evict(self) -> int:
        """If over the cap, deletes least recently used renders until under 90% of it. Returns bytes freed."""
        limit = self.limit_bytes()
        target = int(limit * LOW_WATER)
        with self._lock:
            files = []
            for p in self._files():
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
            total = sum(size for _, size, _ in files)
            if total <= limit:
                self._total = total
                return 0
            from . import favorites, shares  # (imported here: favorites -> render -> cache)

            pinned = favorites.pinned_files() | shares.pinned_files()
            freed = 0
            for _, size, p in sorted(files):
                if p in pinned:
                    continue
                if total - freed <= target:
                    break
                try:
                    p.unlink()
                    freed += size
                except OSError:
                    pass
            self._total = total - freed
            return freed


renders = RenderCache()

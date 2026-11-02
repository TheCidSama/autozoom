"""Tiny dependency-free progress bar with fps and ETA."""

from __future__ import annotations

import sys
import time


def fmt_time(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class ProgressBar:
    def __init__(self, label: str, stream=None, width: int = 28):
        self.label = label
        self.stream = stream or sys.stderr
        self.width = width
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._last_pct = -10

    def __call__(self, done: int, total: int):
        now = time.perf_counter()
        final = done >= total
        if not final and now - self.last < 0.1:
            return
        self.last = now
        total = max(total, 1)
        frac = min(1.0, done / total)
        el = now - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        if self.tty:
            fill = int(frac * self.width)
            bar = "\u2588" * fill + "\u2591" * (self.width - fill)
            self.stream.write(f"\r{self.label:<10} {bar} {frac * 100:5.1f}%  {rate:6.1f} fps  ETA {fmt_time(eta)} ")
            self.stream.flush()
        else:
            pct = int(frac * 100)
            if pct >= self._last_pct + 10 or final:
                self._last_pct = pct
                self.stream.write(f"{self.label}: {pct}% ({rate:.1f} fps, ETA {fmt_time(eta)})\n")
                self.stream.flush()

    def close(self):
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()

"""The zoom plan: a small, hand-editable JSON document.

Coordinates are normalized (0..1, relative to the source frame) so a plan
works at any resolution. Example::

    {
      "version": 1,
      "duration": 22.0,
      "segments": [
        {"start": 2.9, "end": 5.9, "zoom": 2.0, "center_x": 0.88, "center_y": 0.15},
        {"start": 6.3, "end": 11.0, "zoom": 1.8, "center_x": 0.32, "center_y": 0.84,
         "path": [[7.0, 0.25, 0.84], [9.5, 0.36, 0.84]]}
      ]
    }

``path`` is optional: a list of ``[time, x, y]`` keyframes the camera follows
inside the segment (it overrides ``center_x/center_y`` while present).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

PLAN_VERSION = 1


@dataclass
class Segment:
    start: float
    end: float
    zoom: float = 2.0
    center_x: float = 0.5
    center_y: float = 0.5
    path: Optional[List[List[float]]] = None
    kind: str = "manual"

    def center_at(self, t: float) -> Tuple[float, float]:
        if not self.path:
            return self.center_x, self.center_y
        p = self.path
        if t <= p[0][0]:
            return p[0][1], p[0][2]
        if t >= p[-1][0]:
            return p[-1][1], p[-1][2]
        for a, b in zip(p, p[1:]):
            if a[0] <= t <= b[0]:
                k = (t - a[0]) / max(b[0] - a[0], 1e-9)
                return a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k
        return p[-1][1], p[-1][2]

    def to_dict(self) -> dict:
        d = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "zoom": round(self.zoom, 3),
            "center_x": round(self.center_x, 4),
            "center_y": round(self.center_y, 4),
            "kind": self.kind,
        }
        if self.path:
            d["path"] = self.path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        path = d.get("path")
        if path:
            path = sorted([[float(a), float(b), float(c)] for a, b, c in path])
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            zoom=float(d.get("zoom", 2.0)),
            center_x=float(d.get("center_x", 0.5)),
            center_y=float(d.get("center_y", 0.5)),
            path=path or None,
            kind=str(d.get("kind", "manual")),
        )


@dataclass
class Plan:
    segments: List[Segment] = field(default_factory=list)
    source: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    max_zoom: float = 2.0
    focus: List[List[float]] = field(default_factory=list)

    def validate(self) -> "Plan":
        segs = []
        for s in self.segments:
            if s.end <= s.start:
                continue
            s.zoom = max(1.0, min(float(s.zoom), 8.0))
            s.center_x = min(max(s.center_x, 0.0), 1.0)
            s.center_y = min(max(s.center_y, 0.0), 1.0)
            segs.append(s)
        segs.sort(key=lambda s: s.start)
        # resolve overlaps: a later segment starts when the previous ends
        for a, b in zip(segs, segs[1:]):
            if b.start < a.end:
                b.start = a.end
        self.segments = [s for s in segs if s.end > s.start]
        return self

    def segment_at(self, t: float) -> Optional[Segment]:
        for s in self.segments:
            if s.start <= t < s.end:
                return s
        return None

    def focus_at(self, t: float) -> Optional[Tuple[float, float]]:
        f = self.focus
        if not f:
            return None
        if t <= f[0][0]:
            return f[0][1], f[0][2]
        lo, hi = 0, len(f) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if f[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = f[lo], f[hi]
        if t >= b[0]:
            return b[1], b[2]
        k = (t - a[0]) / max(b[0] - a[0], 1e-9)
        return a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k

    def to_dict(self) -> dict:
        d = {
            "version": PLAN_VERSION,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "max_zoom": self.max_zoom,
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.focus:
            d["focus"] = self.focus
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        if isinstance(d, list):  # bare list of segments is accepted too
            d = {"segments": d}
        return cls(
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            source=d.get("source"),
            width=int(d.get("width") or 0),
            height=int(d.get("height") or 0),
            fps=float(d.get("fps") or 0),
            duration=float(d.get("duration") or 0),
            max_zoom=float(d.get("max_zoom") or 2.0),
            focus=[list(map(float, p)) for p in d.get("focus", [])],
        ).validate()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "Plan":
        with open(path) as f:
            return cls.from_dict(json.load(f))

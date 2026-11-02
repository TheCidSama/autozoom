"""Infer *where the action is* from pixels and turn it into a zoom plan.

Pipeline (all on a downscaled grayscale copy of the recording):

1. Frame differencing with a blur + threshold (kills compression noise),
   dilated so strokes of text merge into word-sized blobs.
2. Frames whose change covers a large part of the screen are flagged as
   *global* (scrolls, page loads, app switches) and never zoomed into.
3. A small-blob tracker follows the mouse cursor: a moving cursor leaves a
   pair of cursor-sized blobs (the spot it vacated and the spot it moved to).
   Cursor blobs are removed from the activity signal so plain mouse travel
   doesn't trigger zooms.
4. Remaining blobs are filtered for flicker (tiny components, and blobs that
   repeat in the same box like a blinking caret, a ticking clock or spinner).
5. "Click-like" events = localized change right after the cursor stops.
   "Typing" = bursts of small blobs at a high rate in one region.
6. Events are clustered in time into segments; each gets a zoom level from
   the extent of the activity, a center path that follows the action, a
   lead-in/tail, a minimum hold, and close segments are merged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .media import VideoReader, probe
from .plan import Plan, Segment

DEBUG = False


@dataclass
class AnalysisParams:
    analysis_width: int = 640
    analysis_fps: float = 15.0
    diff_threshold: int = 18          # 0-255 gray difference that counts as change
    global_frac: float = 0.16         # dilated changed area that counts as full-screen change
    global_span: float = 0.55         # ...or change bbox spanning this much of both axes with >5% area
    min_blob_frac: float = 0.00012    # ignore components smaller than this (fraction of frame)
    cursor_max_frac: float = 0.04     # max cursor blob side, fraction of analysis width
    flicker_window: float = 3.0       # seconds to look back for repeating blobs
    flicker_iou: float = 0.8
    max_zoom: float = 2.0
    min_zoom: float = 1.25            # don't bother zooming less than this
    context: float = 1.8              # zoomed view = activity extent * context
    lead: float = 0.5                 # start zooming this long before the action
    tail: float = 1.1                 # keep zoomed this long after the action
    min_hold: float = 1.6             # minimum segment length
    cluster_gap: float = 1.2          # activity further apart than this starts a new cluster
    merge_gap: float = 1.0            # segments closer than this are merged (camera pans instead)
    min_active: float = 0.5           # seconds of activity needed without a click
    min_union_frac: float = 0.0008    # union of changed boxes must cover this much
    min_persist_frac: float = 0.0012  # ...and leave a lasting change this big (unless clicked)
    click_window: float = 0.9         # look for a reaction this long after the cursor stops
    click_radius: float = 0.12        # ...within this distance (fraction of width)


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    area: float  # changed pixels (normalized by frame area)

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


@dataclass
class FrameInfo:
    t: float
    frac: float = 0.0
    is_global: bool = False
    cursor: Optional[Tuple[float, float]] = None  # normalized
    cursor_moving: bool = False
    boxes: List[Box] = field(default_factory=list)  # activity (cursor + flicker removed)
    small: int = 0  # number of small (typing-sized) boxes
    ptr: List[Box] = field(default_factory=list)  # blobs claimed by the cursor tracker


@dataclass
class Event:
    t: float
    box: Box
    weight: float
    kind: str = "change"  # change | click


def _same_spot(a: Box, b: Box, iou: float, dt: float = 1.0) -> bool:
    """Same place, same size: a blinking caret, a ticking clock, a spinner.
    Typing doesn't match because text advances by more than half a box
    between events that are >= 0.4 s apart."""
    if _iou(a, b) > iou:
        return True
    if dt < 0.4:
        return False
    return (abs(a.cx - b.cx) < 0.5 * max(a.w, b.w) and abs(a.cy - b.cy) < 0.5 * max(a.h, b.h)
            and 0.5 < (a.w * a.h + 1e-9) / (b.w * b.h + 1e-9) < 2.0)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-12)


class CursorTracker:
    """Follows the mouse pointer using small-blob motion.

    A moving pointer changes pixels where it was and where it now is. When we
    know where it was (``pos``), the blob overlapping ``pos`` is the vacated
    spot and the other nearby small blob is the new position. When we don't
    know, every small blob is a hypothesis; a hypothesis is confirmed when the
    next frame shows a blob vacating it.
    """

    def __init__(self, W: int, H: int, max_side: float):
        self.W, self.H = W, H
        self.max_side = max_side
        self.tol = max_side * 0.6
        self.jump = 0.18 * W
        self.pos: Optional[np.ndarray] = None
        self.vel = np.zeros(2)
        self.hyps: list = []
        self.still_frames = 0
        self.size: Optional[np.ndarray] = None  # learned cursor blob (w, h)

    def _cursor_like(self, stat) -> bool:
        if self.size is None:
            return True
        w, h = float(stat[2]), float(stat[3])
        return abs(w - self.size[0]) <= 0.35 * self.size[0] + 2 and abs(h - self.size[1]) <= 0.35 * self.size[1] + 2

    def _learn(self, stat):
        s = np.array([stat[2], stat[3]], float)
        self.size = s if self.size is None else self.size * 0.8 + s * 0.2

    def _near(self, p, stat) -> bool:
        x, y, w, h = stat[:4]
        dx = max(x - p[0], 0, p[0] - (x + w))
        dy = max(y - p[1], 0, p[1] - (y + h))
        return math.hypot(dx, dy) <= self.tol

    def update(self, small_stats, small_cents) -> Tuple[Optional[np.ndarray], bool, set]:
        """Returns (position, moving, indices of blobs attributed to the cursor)."""
        n = len(small_stats)
        p = self.pos
        hit = [i for i in range(n) if self._near(p, small_stats[i])] if p is not None else []
        acq = self._acquire(small_stats, small_cents)
        if hit:
            newp, used = self._follow(p, hit, small_stats, small_cents)
            # A tracked "move" that goes nowhere while a real pointer move is
            # visible elsewhere means we latched onto something flickering in
            # place (a clock, a caret). Jump to the real pointer instead.
            if acq is not None and np.linalg.norm(newp - p) < 0.5 * self.tol \
                    and np.linalg.norm(acq[0] - p) > 2 * self.tol:
                hit = []
        if not hit and acq is not None:
            newp, used, hp = acq
            self.vel = newp - hp
            self.pos = newp
            self.hyps = []
            self.still_frames = 0
            return newp.copy(), True, used
        self.hyps = [(c.astype(float), float(st[4])) for i, (c, st) in enumerate(zip(small_cents, small_stats))
                     if not hit or i not in used]
        if not hit:
            if p is None:
                return None, False, set()
            # pointer still (other blobs are activity)
            self.vel *= 0.0
            self.still_frames += 1
            return p.copy(), False, set()
        self.vel = newp - p
        self.pos = newp
        self.still_frames = 0
        return newp.copy(), True, used

    def _acquire(self, small_stats, small_cents):
        """Confirm a hypothesis from the previous frame: it was vacated *and* a
        similar-sized blob appeared a little further on (a real move, not a
        blinking caret)."""
        n = len(small_stats)
        for hp, harea in self.hyps:
            hh = [i for i in range(n) if self._near(hp, small_stats[i]) and self._cursor_like(small_stats[i])]
            if not hh:
                continue
            others = [
                i for i in range(n) if i not in hh
                and self.tol * 0.8 < np.linalg.norm(small_cents[i] - hp) < self.jump
                and 0.5 < small_stats[i][4] / max(harea, 1) < 2.0
                and self._cursor_like(small_stats[i])
            ]
            if not others:
                continue
            j = min(others, key=lambda i: np.linalg.norm(small_cents[i] - hp))
            return small_cents[j].astype(float), set(hh + [j]), hp
        return None

    def _follow(self, p, hit, small_stats, small_cents):
        n = len(small_stats)
        used = set(hit)
        pred = p + self.vel
        # the new position must look like the pointer we learned (size), so a
        # clock tick or caret blink near the pointer can't steal the track
        others = [i for i in range(n) if i not in hit and np.linalg.norm(small_cents[i] - p) < self.jump
                  and self._cursor_like(small_stats[i])]
        if others:
            j = min(others, key=lambda i: np.linalg.norm(small_cents[i] - pred))
            newp = small_cents[j].astype(float)
            used.add(j)
            if len(hit) == 1 and np.linalg.norm(newp - p) > self.tol * 1.5:
                self._learn(small_stats[j])
        else:
            c = small_cents[hit[0]].astype(float)
            newp = c + (c - p)
            x, y, w, h = small_stats[hit[0]][:4]
            newp = np.array([min(max(newp[0], x), x + w), min(max(newp[1], y), y + h)], float)
        return newp, used


def scan(path: str, params: AnalysisParams, progress: Optional[Callable[[int, int], None]] = None):
    """Decode the video downscaled and extract per-frame activity."""
    info = probe(path)
    aw = min(params.analysis_width, info.width)
    aw -= aw % 2
    ah = int(round(info.height * aw / info.width))
    ah -= ah % 2
    fps = min(params.analysis_fps, info.fps)
    reader = VideoReader(path, aw, ah, fps=fps, gray=True)
    total = max(1, int(info.duration * fps))
    A = float(aw * ah)
    min_area = params.min_blob_frac * A
    max_side = params.cursor_max_frac * aw
    tracker = CursorTracker(aw, ah, max_side)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frames: List[FrameInfo] = []
    snaps: List[Tuple[float, np.ndarray]] = []  # small grayscale snapshots for persistence checks
    snap_w = 256
    snap_h = max(2, int(round(ah * snap_w / aw)))
    recent: List[Tuple[float, Box]] = []  # for flicker detection
    prev = None
    for i, g in enumerate(reader):
        t = i / fps
        g = cv2.GaussianBlur(g, (3, 3), 0)
        if i % 4 == 0:
            snaps.append((t, cv2.resize(g, (snap_w, snap_h), interpolation=cv2.INTER_AREA)))
        fi = FrameInfo(t=t, cursor=None)
        if prev is None:
            prev = g
            frames.append(fi)
            continue
        d = cv2.absdiff(g, prev)
        prev = g
        _, m = cv2.threshold(d, params.diff_threshold, 255, cv2.THRESH_BINARY)
        raw = cv2.countNonZero(m)
        if raw == 0:
            p = tracker.pos
            tracker.still_frames += 1
            fi.cursor = (p[0] / aw, p[1] / ah) if p is not None else None
            frames.append(fi)
            if progress:
                progress(i + 1, total)
            continue
        m = cv2.dilate(m, kernel)
        frac = cv2.countNonZero(m) / A
        fi.frac = frac
        ys, xs = np.nonzero(m[:: 4, :: 4])
        span_x = (xs.max() - xs.min()) * 4 / aw if len(xs) else 0
        span_y = (ys.max() - ys.min()) * 4 / ah if len(ys) else 0
        if frac > params.global_frac or (frac > 0.05 and span_x > params.global_span and span_y > params.global_span):
            fi.is_global = True
            tracker.pos = None  # cursor position is unreliable across big changes
            tracker.hyps = []
            frames.append(fi)
            if progress:
                progress(i + 1, total)
            continue
        n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        small_idx = [
            k for k in range(1, n)
            if stats[k, 2] <= max_side and stats[k, 3] <= max_side * 1.5 and stats[k, 4] >= 6
        ]
        pos, moving, used = tracker.update([stats[k] for k in small_idx], [cents[k] for k in small_idx])
        cursor_blobs = {small_idx[j] for j in used}
        if pos is not None:
            fi.cursor = (float(pos[0] / aw), float(pos[1] / ah))
            fi.cursor_moving = moving
        # leftovers of a moving pointer the tracker didn't claim (merged/partial blobs)
        pf = frames[-1] if frames else None
        if moving or (pf is not None and pf.cursor_moving):
            pts = [np.array(c) * (aw, ah) for c in (fi.cursor, pf.cursor if pf else None) if c is not None]
            for k in small_idx:
                if k not in cursor_blobs and tracker._cursor_like(stats[k]) and any(
                        np.linalg.norm(cents[k] - q) < max_side for q in pts):
                    cursor_blobs.add(k)
        for k in range(1, n):
            if k in cursor_blobs:
                x, y, w, h, area = stats[k]
                fi.ptr.append(Box(x / aw, y / ah, (x + w) / aw, (y + h) / ah, area / A))
                continue
            x, y, w, h, area = stats[k]
            if area < min_area:
                continue
            b = Box(x / aw, y / ah, (x + w) / aw, (y + h) / ah, area / A)
            # flicker: same box seen repeatedly in the recent past (caret, clock, spinner)
            reps = sum(1 for (tt, rb) in recent if t - tt <= params.flicker_window and _same_spot(rb, b, params.flicker_iou, t - tt))
            recent.append((t, b))
            if reps >= 2:
                continue
            fi.boxes.append(b)
            if h <= max_side * 1.5:
                fi.small += 1
        if len(recent) > 400:
            recent = [r for r in recent if t - r[0] <= params.flicker_window]
        frames.append(fi)
        if progress:
            progress(i + 1, total)
    # the first/last frames of a scroll or page load are below the global
    # threshold but still scattered: treat a short margin around them as global
    margin = int(round(0.3 * fps))
    gl = [i for i, f in enumerate(frames) if f.is_global]
    for i in gl:
        for j in range(max(0, i - margin), min(len(frames), i + margin + 1)):
            frames[j].boxes = []
    _drop_pointer_trails(frames, params, ah / aw)
    return info, frames, fps, snaps


def _drop_pointer_trails(frames: List[FrameInfo], params: AnalysisParams, aspect: float):
    """Remove the trail a travelling pointer leaves when the tracker missed it.

    Cursor-sized blobs are linked frame-to-frame into chains (overlapping or
    within a short hop). A chain that travels several times its own size, fast,
    is a pointer (or a dragged icon) on the move, including the slow ease-in
    and ease-out at its ends. Typing advances under a glyph per sample and
    animations stay in place, so they survive. This doesn't depend on tracker
    state, so it gives the same answer whatever the source frame rate is."""
    def small(b):
        return b.w <= params.cursor_max_frac * 1.2 and b.h <= params.cursor_max_frac * 1.8 / aspect

    def dist(a, b):
        return math.hypot(a.cx - b.cx, (a.cy - b.cy) * aspect)

    def size(b):
        return max(b.w, b.h * aspect)

    chains: List[List[Tuple[int, int]]] = []   # (frame index, box index)
    open_: List[List[Tuple[int, int]]] = []
    allb = [f.boxes + f.ptr for f in frames]  # indices >= len(f.boxes) are pointer blobs
    for i, f in enumerate(frames):
        cand = [k for k, b in enumerate(allb[i]) if small(b)]
        taken: set = set()
        still = []
        for ch in open_:
            fi, ki = ch[-1]
            if fi != i - 1:
                chains.append(ch)
                continue
            last = allb[fi][ki]
            opts = [k for k in cand if k not in taken and dist(last, allb[i][k]) < 0.2
                    and 0.5 < (allb[i][k].w + 1e-6) / (last.w + 1e-6) < 2.0]
            if not opts:
                chains.append(ch)
                continue
            # follow the box furthest along (the pointer's new spot), keep its pair too
            k = max(opts, key=lambda k: dist(allb[ch[0][0]][ch[0][1]], allb[i][k]))
            for o in opts:
                if o == k or dist(allb[i][o], allb[i][k]) < 2.5 * size(allb[i][k]):
                    ch.append((i, o)) if o != k else None
                    taken.add(o)
            ch.append((i, k))
            taken.add(k)
            still.append(ch)
        open_ = still + [[(i, k)] for k in cand if k not in taken]
    chains.extend(open_)

    drop = [set() for _ in frames]
    for ch in chains:
        if len(ch) < 2:
            continue
        first = allb[ch[0][0]][ch[0][1]]
        last = allb[ch[-1][0]][ch[-1][1]]
        span = max(1, ch[-1][0] - ch[0][0])
        travel = dist(first, last)
        sz = max(size(first), 1e-4)
        if travel > 3 * sz and travel / span > 0.6 * sz:
            for fi, ki in ch:
                drop[fi].add(ki)
    for f, d in zip(frames, drop):
        if d:
            f.boxes = [b for k, b in enumerate(f.boxes) if k not in d]


def _events(frames: List[FrameInfo], params: AnalysisParams, fps: float, aspect: float = 0.5625) -> List[Event]:
    events: List[Event] = []
    for f in frames:
        for b in f.boxes:
            events.append(Event(f.t, b, math.sqrt(b.area) + 1e-4))
    # click-like: cursor stops after travelling, then something changes near it
    start_pos = None
    for i in range(1, len(frames)):
        f, pf = frames[i], frames[i - 1]
        if f.cursor_moving:
            if start_pos is None:
                start_pos = pf.cursor or f.cursor
            continue
        stop = pf.cursor_moving and f.cursor is not None and start_pos is not None
        travel = math.hypot(f.cursor[0] - start_pos[0], f.cursor[1] - start_pos[1]) if stop else 0.0
        start_pos = None
        if stop and travel >= 0.02:
            cx, cy = f.cursor
            j_end = min(len(frames), i + int(params.click_window * fps) + 1)
            for j in range(max(0, i - 1), j_end):
                if frames[j].is_global:
                    break
                near = [
                    b for b in frames[j].boxes
                    if not _is_cursor_ghost(b, cx, cy, params, aspect)
                    and (math.hypot(b.cx - cx, (b.cy - cy) * aspect) < params.click_radius
                         or (b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1))
                ]
                if near:
                    s = 0.035
                    events.append(Event(frames[j].t, Box(cx - s, cy - s * 1.6, cx + s, cy + s * 1.6, 0.0), 0.35, "click"))
                    break
    events.sort(key=lambda e: e.t)
    return events


def _is_cursor_ghost(b: Box, cx: float, cy: float, params: AnalysisParams, aspect: float) -> bool:
    """A cursor-sized blob sitting on the cursor is the cursor itself, not a reaction."""
    small = b.w <= params.cursor_max_frac * 1.2 and b.h <= params.cursor_max_frac * 1.8 / aspect
    return small and math.hypot(b.cx - cx, (b.cy - cy) * aspect) < 0.04


def _persistent_change(c: List[Event], snaps, params: AnalysisParams) -> float:
    """Fraction of the frame that differs between just before and just after
    a cluster, inside its bounding box. Typing, opening panels and charts leave
    a lasting change; a pointer gliding by doesn't."""
    t0, t1 = c[0].t - 0.3, c[-1].t + 0.3
    before = max((s for s in snaps if s[0] <= t0), key=lambda s: s[0], default=snaps[0])[1]
    after = min((s for s in snaps if s[0] >= t1), key=lambda s: s[0], default=snaps[-1])[1]
    h, w = before.shape
    x0 = max(0, int(min(e.box.x0 for e in c) * w) - 2)
    y0 = max(0, int(min(e.box.y0 for e in c) * h) - 2)
    x1 = min(w, int(math.ceil(max(e.box.x1 for e in c) * w)) + 2)
    y1 = min(h, int(math.ceil(max(e.box.y1 for e in c) * h)) + 2)
    d = cv2.absdiff(before[y0:y1, x0:x1], after[y0:y1, x0:x1])
    return float((d > params.diff_threshold).sum()) / (w * h)


def _union_frac(evs: List[Event], res=(128, 72)) -> float:
    canvas = np.zeros((res[1], res[0]), np.uint8)
    for e in evs:
        if e.kind != "change":
            continue
        b = e.box
        cv2.rectangle(canvas, (int(b.x0 * res[0]), int(b.y0 * res[1])), (int(math.ceil(b.x1 * res[0])), int(math.ceil(b.y1 * res[1]))), 1, -1)
    return float(canvas.mean())


def build_plan(info, frames: List[FrameInfo], fps: float, params: AnalysisParams, snaps=None) -> Plan:
    duration = info.duration or (frames[-1].t if frames else 0.0)
    global_times = [f.t for f in frames if f.is_global]
    aspect = info.height / max(1, info.width)
    events = _events(frames, params, fps, aspect)

    # 1. cluster events in space-time. Several clusters can be open at once so a
    #    stray blip elsewhere (a clock, a notification) can't glue unrelated
    #    actions together. Full-screen changes close every cluster.
    clusters: List[List[Event]] = []
    open_: List[List[Event]] = []
    gi = 0

    def recent_box(c, t):
        win = [o for o in c if t - o.t <= 1.5] or c[-1:]
        return (min(o.box.x0 for o in win), min(o.box.y0 for o in win),
                max(o.box.x1 for o in win), max(o.box.y1 for o in win))

    for e in events:
        crossed = False
        while gi < len(global_times) and global_times[gi] < e.t:
            crossed = True
            gi += 1
        if crossed:
            # close clusters whose last event precedes the global change
            keep = []
            for c in open_:
                (clusters if global_times[gi - 1] > c[-1].t else keep).append(c)
            open_ = keep
        keep = []
        for c in open_:
            (clusters if e.t - c[-1].t > params.cluster_gap else keep).append(c)
        open_ = keep
        best, bd = None, 1e9
        for c in open_:
            x0, y0, x1, y1 = recent_box(c, e.t)
            dx = max(x0 - e.box.cx, 0, e.box.cx - x1)
            dy = max(y0 - e.box.cy, 0, e.box.cy - y1) * aspect
            d = math.hypot(dx, dy)
            if d < bd:
                best, bd = c, d
        # tiny blips (clock digits, icons) only join if they're right next to the action
        lim = 0.03 if (e.kind == "change" and e.box.area < 4 * params.min_blob_frac) else 0.12
        if best is not None and all(o.kind == "change" and o.box.area < 4 * params.min_blob_frac for o in best):
            lim = 0.03  # a cluster of tiny blips (a clock) doesn't pull in real activity
        if best is not None and bd < lim:
            best.append(e)
        else:
            open_.append([e])
    clusters.extend(open_)
    clusters.sort(key=lambda c: c[0].t)

    # concurrent clusters that are close together are one action (e.g. a click
    # on "Refresh" and the chart that animates next to it)
    def bbox(c):
        return (min(o.box.x0 for o in c), min(o.box.y0 for o in c), max(o.box.x1 for o in c), max(o.box.y1 for o in c))

    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if b[0].t - a[-1].t > 0.5 or a[0].t - b[-1].t > 0.5:
                    continue
                if any(a[-1].t < g < b[0].t or b[-1].t < g < a[0].t for g in global_times):
                    continue
                ba, bb = bbox(a), bbox(b)
                dx = max(bb[0] - ba[2], ba[0] - bb[2], 0)
                dy = max(bb[1] - ba[3], ba[1] - bb[3], 0) * aspect
                tiny = lambda c: all(o.kind == "change" and o.box.area < 4 * params.min_blob_frac for o in c)
                if math.hypot(dx, dy) < 0.08 and not tiny(a) and not tiny(b):
                    clusters[i] = sorted(a + b, key=lambda e: e.t)
                    del clusters[j]
                    changed = True
                    break
            if changed:
                break

    segs: List[Segment] = []
    scores: List[float] = []
    dt = 1.0 / fps
    for c in clusters:
        has_click = any(e.kind == "click" for e in c)
        active = len({round(e.t * fps) for e in c if e.kind == "change"}) * dt
        if not has_click and active < params.min_active:
            continue
        if not has_click and _union_frac(c) < params.min_union_frac:
            continue
        if not has_click and snaps and _persistent_change(c, snaps, params) < params.min_persist_frac:
            continue  # something passed through (a pointer, a hover) but nothing changed
        # 2. zoom from the windowed extent of the activity
        exts = []
        for e in c:
            win = [o for o in c if abs(o.t - e.t) <= 0.75]
            x0 = min(o.box.x0 for o in win)
            x1 = max(o.box.x1 for o in win)
            y0 = min(o.box.y0 for o in win)
            y1 = max(o.box.y1 for o in win)
            exts.append((x1 - x0, y1 - y0))
        ew = float(np.percentile([a for a, _ in exts], 85))
        eh = float(np.percentile([b for _, b in exts], 85))
        z = min(params.max_zoom, 1.0 / max(ew * params.context, 1e-6), 1.0 / max(eh * params.context, 1e-6))
        if DEBUG:
            print("cluster", round(c[0].t, 2), round(c[-1].t, 2), len(c), "ext", round(ew, 3), round(eh, 3), "z", round(z, 2))
            lo = min(c, key=lambda o: o.box.y0); hi = max(c, key=lambda o: o.box.y1)
            print("   ", lo.t, lo.kind, lo.box, hi.t, hi.kind, hi.box)
        if z < params.min_zoom:
            continue
        # 3. center path following the action
        t0, t1 = c[0].t, c[-1].t
        start = max(0.0, t0 - params.lead)
        end = min(duration, t1 + params.tail)
        if end - start < params.min_hold:
            end = min(duration, start + params.min_hold)
        # never zoom across a full-screen change
        for gt in global_times:
            if t1 < gt < end:
                end = gt
                break
        for gt in reversed(global_times):
            if start < gt < t0:
                start = gt + dt
                break
        if end - start < 0.4:
            continue
        keys = []
        tk = t0
        while True:
            win = [o for o in c if abs(o.t - tk) <= 0.6]
            if win:
                ws = np.array([o.weight for o in win])
                xs = np.array([o.box.cx for o in win])
                ys = np.array([o.box.cy for o in win])
                x = float((xs * ws).sum() / ws.sum())
                y = float((ys * ws).sum() / ws.sum())
                keys.append([tk, x, y])  # unclamped: the camera clamps per layout
            if tk >= t1:
                break
            tk = min(t1, tk + 0.25)
        if not keys:
            continue
        arr = np.array(keys)
        if len(arr) >= 3:  # light smoothing of the path
            sm = arr.copy()
            for k in (1, 2):
                sm[:, k] = np.convolve(np.pad(arr[:, k], 1, mode="edge"), np.ones(3) / 3, mode="valid")
            arr = sm
        cx, cy = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
        spread = float(np.max(np.hypot(arr[:, 1] - cx, (arr[:, 2] - cy) * aspect)))
        path = None
        if spread > 0.04:
            path = [[round(float(a), 3), round(float(b), 4), round(float(d), 4)] for a, b, d in arr]
            cx, cy = float(arr[0, 1]), float(arr[0, 2])
        smalls = sum(1 for e in c if e.kind == "change" and e.box.h < 0.06)
        rate = smalls / max(t1 - t0, 0.5)
        kind = "click" if has_click else "activity"
        one_line = float(np.std([e.box.cy for e in c if e.kind == "change"] or [0])) < 0.02
        if rate >= 2.5 and (t1 - t0) >= 1.0 and one_line:
            kind = "typing"
        segs.append(Segment(start=round(start, 3), end=round(end, 3), zoom=round(z, 3),
                            center_x=round(cx, 4), center_y=round(cy, 4), path=path, kind=kind))
        scores.append(active + 1.0 * sum(1 for e in c if e.kind == "click"))

    # 4. resolve time overlaps between concurrent clusters: the stronger wins,
    #    the weaker is trimmed (or dropped if too little is left)
    order = sorted(range(len(segs)), key=lambda i: -scores[i])
    accepted: List[Segment] = []
    for i in order:
        s = segs[i]
        for a in accepted:
            if s.end <= a.start or s.start >= a.end:
                continue
            if s.start < a.start:
                s.end = min(s.end, a.start)
            else:
                s.start = max(s.start, a.end)
            if s.path:
                s.path = [p for p in s.path if s.start <= p[0] <= s.end] or None
        if s.end - s.start >= min(params.min_hold, 1.0):
            accepted.append(s)
    segs = accepted

    def keys(s: Segment):
        if s.path:
            return [[s.start, s.path[0][1], s.path[0][2]]] + s.path + [[s.end, s.path[-1][1], s.path[-1][2]]]
        return [[s.start, s.center_x, s.center_y], [s.end, s.center_x, s.center_y]]

    # 5. merge segments that are close in time (camera pans instead of zooming out/in)
    segs.sort(key=lambda s: s.start)
    merged: List[Segment] = []
    for s in segs:
        if merged:
            m = merged[-1]
            gap = s.start - m.end
            crosses_global = any(m.end - 0.05 <= gt <= s.start + 0.05 for gt in global_times)
            if gap < params.merge_gap and not crosses_global:
                z = min(m.zoom, s.zoom)
                pm = keys(m)
                ps = keys(s)
                # start panning shortly after the previous action, not at the very end of its tail
                pm[-1][0] = max(pm[-2][0], min(pm[-1][0], m.end - params.tail + 0.4))
                ps[0][0] = s.start + min(params.lead, (s.end - s.start) / 2)  # arrive as the action starts
                pts = sorted(pm + ps, key=lambda p: p[0])
                path = [list(p) for p in pts]
                path = [[round(a, 3), round(b, 4), round(c, 4)] for a, b, c in path]
                m.end = max(m.end, s.end)
                m.zoom = z
                m.path = path
                m.center_x, m.center_y = path[0][1], path[0][2]
                if m.kind != s.kind:
                    m.kind = "mixed"
                continue
        merged.append(s)

    focus = _focus_track(frames, events)
    return Plan(
        source=info.path,
        width=info.width,
        height=info.height,
        fps=info.fps,
        duration=round(duration, 3),
        max_zoom=params.max_zoom,
        segments=merged,
        focus=focus,
    )


def _focus_track(frames: List[FrameInfo], events: List[Event], step: float = 0.25):
    """A continuous point of attention (cursor, else latest activity). Used
    by crop layouts such as 9:16 to follow the action between zooms."""
    if not frames:
        return []
    out = []
    last = (0.5, 0.5)
    ei = 0
    t_next = 0.0
    for f in frames:
        while ei < len(events) and events[ei].t <= f.t:
            if events[ei].kind == "click" or events[ei].box.area > 0:
                last = (events[ei].box.cx, events[ei].box.cy)
            ei += 1
        if f.cursor is not None and f.cursor_moving:
            last = f.cursor
        if f.t >= t_next:
            out.append([round(f.t, 3), round(float(last[0]), 4), round(float(last[1]), 4)])
            t_next += step
    return out


def analyze(path: str, params: Optional[AnalysisParams] = None,
            progress: Optional[Callable[[int, int], None]] = None) -> Plan:
    params = params or AnalysisParams()
    info, frames, fps, snaps = scan(path, params, progress)
    return build_plan(info, frames, fps, params, snaps)

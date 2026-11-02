"""Per-frame virtual camera from a zoom plan.

The plan says *what* to look at; the camera decides *how to get there*. Each
axis (center x, center y, log zoom) is driven by a critically-damped spring,
integrated exactly per frame. That gives the Screen Studio feel: moves start
gently (zero initial velocity), never overshoot and settle without jitter.
After smoothing, the view rectangle is clamped inside the source frame so the
warp never reveals black edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .plan import Plan


@dataclass
class CameraParams:
    omega: float = 5.5          # spring stiffness (rad/s). ~0.9 s to settle a zoom
    omega_follow: float = 3.0   # softer spring for following the focus between zooms (crop layouts)


def spring_step(x: float, v: float, target: float, omega: float, dt: float) -> Tuple[float, float]:
    """Exact critically-damped spring update for a constant target over dt."""
    e = x - target
    w = math.exp(-omega * dt)
    tmp = (v + omega * e) * dt
    e_new = (e + tmp) * w
    v_new = (v - omega * tmp) * w
    return target + e_new, v_new


@dataclass
class Camera:
    """Arrays of per-frame camera state in *source pixels*."""

    cx: np.ndarray
    cy: np.ndarray
    zoom: np.ndarray
    base_w: float
    base_h: float
    src_w: int
    src_h: int

    def __len__(self):
        return len(self.cx)

    def view(self, i: int) -> Tuple[float, float, float, float]:
        """(x0, y0, w, h) of the source rectangle shown at frame i."""
        z = float(self.zoom[i])
        w, h = self.base_w / z, self.base_h / z
        return float(self.cx[i]) - w / 2, float(self.cy[i]) - h / 2, w, h

    def view_at(self, cx: float, cy: float, z: float) -> Tuple[float, float, float, float]:
        w, h = self.base_w / z, self.base_h / z
        return cx - w / 2, cy - h / 2, w, h


def _clamp(cx, cy, z, base_w, base_h, W, H):
    hw, hh = base_w / z / 2, base_h / z / 2
    return min(max(cx, hw), W - hw), min(max(cy, hh), H - hh)


def compute_camera(
    plan: Plan,
    n_frames: int,
    fps: float,
    src_w: int,
    src_h: int,
    base_w: float = None,
    base_h: float = None,
    follow_focus: bool = False,
    params: CameraParams = None,
) -> Camera:
    """Simulate the camera for ``n_frames`` frames.

    ``base_w/base_h`` is the source-pixel size of the view at zoom 1 (the whole
    frame for normal layouts, a narrower crop for e.g. 9:16 output).
    ``follow_focus`` makes the zoom-1 view track the plan's focus points
    (needed when the base view is a crop).
    """
    params = params or CameraParams()
    base_w = float(base_w or src_w)
    base_h = float(base_h or src_h)
    dt = 1.0 / fps
    cx = np.empty(n_frames)
    cy = np.empty(n_frames)
    zz = np.empty(n_frames)

    # Plan zooms are relative to the full source frame. When the zoom-1 view is
    # already a crop (e.g. 9:16 output), only zoom further if the plan asks for
    # a tighter view than the crop already is.
    crop_k = min(base_w / src_w, base_h / src_h)

    def target(t):
        seg = plan.segment_at(t)
        if seg is not None:
            x, y = seg.center_at(t)
            return x * src_w, y * src_h, max(1.0, seg.zoom * crop_k), True
        if follow_focus:
            f = plan.focus_at(t)
            if f is not None:
                return f[0] * src_w, f[1] * src_h, 1.0, False
        return src_w / 2, src_h / 2, 1.0, False

    tx, ty, tz, _ = target(0.0)
    x, y = _clamp(tx, ty, tz, base_w, base_h, src_w, src_h)
    lz = math.log(tz)
    vx = vy = vz = 0.0
    for i in range(n_frames):
        t = i * dt
        tx, ty, tz, in_seg = target(t)
        # aim at a reachable center for the *target* zoom so the camera
        # doesn't drift toward an unreachable point and then snap
        tx, ty = _clamp(tx, ty, tz, base_w, base_h, src_w, src_h)
        om = params.omega if (in_seg or not follow_focus) else params.omega_follow
        if i > 0:
            x, vx = spring_step(x, vx, tx, om, dt)
            y, vy = spring_step(y, vy, ty, om, dt)
            lz, vz = spring_step(lz, vz, math.log(tz), params.omega, dt)
        z = min(max(math.exp(lz), 1.0), 16.0)
        cxi, cyi = _clamp(x, y, z, base_w, base_h, src_w, src_h)
        cx[i], cy[i], zz[i] = cxi, cyi, z
    return Camera(cx, cy, zz, base_w, base_h, src_w, src_h)


def camera_motion(cam: Camera, i: int, out_w: float) -> float:
    """Approximate max on-screen pixel displacement between frame i-1 and i
    (for an output that is ``out_w`` pixels wide)."""
    if i <= 0:
        return 0.0
    a = cam.view(i - 1)
    b = cam.view(i)
    s = out_w / b[2]
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    best = 0.0
    for u, v in corners:
        # source point under output corner (u, v) in frame i-1, re-projected in frame i
        px = a[0] + u * a[2]
        py = a[1] + v * a[3]
        qx = (px - b[0]) * s - u * out_w
        qy = (py - b[1]) * s - v * out_w * b[3] / b[2]
        best = max(best, math.hypot(qx, qy))
    return best

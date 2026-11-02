"""Backgrounds, layout, rounded corners, drop shadow and window chrome.

Everything that does not change from frame to frame (background, shadow,
title bar) is baked once into a *base* image; per frame we only paste the
warped recording into the content rectangle and blend its rounded corners.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

# name -> (angle in degrees, color stops as hex). Rendered as a soft
# "mesh" gradient: a linear blend plus two large radial glows.
PRESETS = {
    "sunset": (135, ["#f97316", "#ec4899", "#8b5cf6"]),
    "ocean": (135, ["#22d3ee", "#3b82f6", "#1e3a8a"]),
    "midnight": (150, ["#1e1b4b", "#312e81", "#0f172a"]),
    "aurora": (120, ["#10b981", "#06b6d4", "#6366f1"]),
    "candy": (135, ["#f472b6", "#a78bfa", "#60a5fa"]),
    "peach": (135, ["#fde68a", "#fca5a5", "#f9a8d4"]),
    "forest": (140, ["#064e3b", "#059669", "#a3e635"]),
    "lavender": (135, ["#c4b5fd", "#e9d5ff", "#f0abfc"]),
    "slate": (160, ["#475569", "#1e293b", "#0f172a"]),
    "mono": (135, ["#f3f4f6", "#d1d5db", "#9ca3af"]),
}

HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


def hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def _srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


def gradient(w: int, h: int, angle: float, stops: List[str]) -> np.ndarray:
    """Float32 BGR (0..255) multi-stop gradient with soft glows, blended in
    linear light so midpoints don't turn muddy."""
    cols = np.array([hex_to_bgr(s) for s in stops], np.float32)
    lin = _srgb_to_lin(cols)
    # compute on a small grid, then upscale (gradients are smooth)
    gw, gh = max(2, w // 4), max(2, h // 4)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    xx = xx / (gw - 1) - 0.5
    yy = (yy / (gh - 1) - 0.5) * (h / w)
    a = math.radians(angle)
    d = xx * math.sin(a) - yy * math.cos(a)  # 135deg: top-left -> bottom-right
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    n = len(lin)
    pos = d * (n - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    f = (pos - i0)[..., None]
    f = f * f * (3 - 2 * f)
    img = lin[i0] * (1 - f) + lin[i0 + 1] * f
    # two radial glows using the first/last stops for a "mesh" look
    for (gx, gy, col, strength) in ((-0.32, -0.22 * h / w, lin[0], 0.45), (0.35, 0.25 * h / w, lin[-1], 0.35)):
        r2 = (xx - gx) ** 2 + (yy - gy) ** 2
        g = np.exp(-r2 / (2 * 0.22 ** 2))[..., None] * strength
        img = img * (1 - g) + col * g
    img = _lin_to_srgb(img).astype(np.float32)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def make_background(spec: str, w: int, h: int) -> np.ndarray:
    """Float32 BGR background from a preset name, '#hex' color or image path."""
    spec = (spec or "sunset").strip()
    key = spec.lower()
    if key in PRESETS:
        angle, stops = PRESETS[key]
        return gradient(w, h, angle, stops)
    if HEX_RE.match(spec):
        img = np.empty((h, w, 3), np.float32)
        img[:] = hex_to_bgr(spec)
        return img
    if os.path.exists(spec):
        src = cv2.imread(spec, cv2.IMREAD_COLOR)
        if src is None:
            raise ValueError(f"Could not read background image {spec!r}")
        sh, sw = src.shape[:2]
        s = max(w / sw, h / sh)
        rw, rh = max(w, int(math.ceil(sw * s))), max(h, int(math.ceil(sh * s)))
        src = cv2.resize(src, (rw, rh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        x0, y0 = (rw - w) // 2, (rh - h) // 2
        return src[y0:y0 + h, x0:x0 + w].astype(np.float32)
    raise ValueError(
        f"Unknown background {spec!r}: use a preset ({', '.join(PRESETS)}), a #hex color or an image path"
    )


ASPECTS = {"source": None, "16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1), "4:5": (4, 5), "9:16": (9, 16)}


@dataclass
class Style:
    background: str = "sunset"
    padding: str = "8%"          # "8%" of the output's short side, or pixels ("60")
    radius: float = 16.0         # corner radius in px at 1080p (scales with output)
    shadow: bool = True
    shadow_strength: float = 0.55
    frame: str = "none"          # none | light | dark  (macOS-style window title bar)
    aspect: str = "source"       # source | 16:9 | 4:3 | 1:1 | 4:5 | 9:16
    fit: str = "auto"            # auto | fit | crop


@dataclass
class Layout:
    out_w: int
    out_h: int
    win: Tuple[int, int, int, int]       # x, y, w, h (window incl. title bar)
    content: Tuple[int, int, int, int]   # x, y, w, h (where the video goes)
    bar_h: int
    radius: int
    base_w: float                         # source px visible at zoom 1
    base_h: float
    crop: bool
    unit: float                           # px per "1080p px"


def parse_padding(p, short_side: int) -> int:
    if p is None:
        return 0
    s = str(p).strip()
    if s.endswith("%"):
        return int(round(float(s[:-1]) / 100.0 * short_side))
    v = float(s)
    if 0 < v < 1:
        return int(round(v * short_side))
    return int(round(v))


def _even(v: float) -> int:
    v = int(round(v))
    return v - (v % 2)


def output_size(src_w: int, src_h: int, aspect: str, size: Tuple[int, int] = None) -> Tuple[int, int]:
    if size:
        return _even(size[0]), _even(size[1])
    a = ASPECTS.get(aspect)
    if aspect not in ASPECTS:
        m = re.match(r"^(\d+(?:\.\d+)?)[:x/](\d+(?:\.\d+)?)$", aspect or "")
        if not m:
            raise ValueError(f"Unknown aspect {aspect!r}")
        a = (float(m.group(1)), float(m.group(2)))
    if a is None:
        return _even(src_w), _even(src_h)
    short = min(src_w, src_h)
    ar = a[0] / a[1]
    if ar >= 1:
        return _even(short * ar), _even(short)
    return _even(short), _even(short / ar)


def compute_layout(src_w: int, src_h: int, style: Style, size: Tuple[int, int] = None) -> Layout:
    out_w, out_h = output_size(src_w, src_h, style.aspect, size)
    short = min(out_w, out_h)
    unit = short / 1080.0
    pad = parse_padding(style.padding, short)
    bar_h = int(round(38 * unit)) if style.frame in ("light", "dark") else 0
    aw, ah = max(16, out_w - 2 * pad), max(16, out_h - 2 * pad - bar_h)
    ra, rc = src_w / src_h, aw / ah
    crop = style.fit == "crop" or (style.fit == "auto" and max(ra / rc, rc / ra) > 1.5)
    if crop:
        cw, ch = aw, ah
        if cw / ch < ra:
            base_h = float(src_h)
            base_w = src_h * cw / ch
        else:
            base_w = float(src_w)
            base_h = src_w * ch / cw
    else:
        s = min(aw / src_w, ah / src_h)
        cw, ch = int(round(src_w * s)), int(round(src_h * s))
        base_w, base_h = float(src_w), float(src_h)
    wx = (out_w - cw) // 2
    wy = (out_h - ch - bar_h) // 2
    radius = int(round(max(0.0, style.radius) * unit))
    radius = min(radius, cw // 2, (ch + bar_h) // 2)
    return Layout(out_w, out_h, (wx, wy, cw, ch + bar_h), (wx, wy + bar_h, cw, ch), bar_h, radius,
                  base_w, base_h, crop, unit)


def rounded_mask(w: int, h: int, r: int, ss: int = 1) -> np.ndarray:
    """Anti-aliased rounded-rectangle alpha (float32 0..1) via signed distance."""
    if r <= 0:
        return np.ones((h, w), np.float32)
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    qx = np.abs(xs - w / 2) - (w / 2 - r)
    qy = np.abs(ys - h / 2) - (h / 2 - r)
    qx = qx[None, :]
    qy = qy[:, None]
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0)
    d = outside + inside - r
    return np.clip(0.5 - d, 0, 1).astype(np.float32)


@dataclass
class Canvas:
    layout: Layout
    base: np.ndarray                      # uint8 BGR, background + shadow + chrome
    patches: List[Tuple[slice, slice, np.ndarray, np.ndarray]] = field(default_factory=list)
    # (rows, cols) in canvas, alpha (h,w,1) float32, base patch float32


def build_canvas(layout: Layout, style: Style) -> Canvas:
    W, H = layout.out_w, layout.out_h
    bg = make_background(style.background, W, H)
    wx, wy, ww, wh = layout.win
    mask = np.zeros((H, W), np.float32)
    mask[wy:wy + wh, wx:wx + ww] = rounded_mask(ww, wh, layout.radius)
    u = layout.unit
    if style.shadow and style.shadow_strength > 0:
        # two layers: a wide soft key shadow and a tight contact shadow
        sh = np.zeros_like(mask)
        for blur, dy, op in ((36 * u, 18 * u, 0.55), (8 * u, 4 * u, 0.35)):
            m = np.zeros_like(mask)
            off = int(round(dy))
            m[off:, :] = mask[: H - off, :] if off > 0 else mask
            k = int(blur * 3) | 1
            m = cv2.GaussianBlur(m, (k, k), blur)
            sh = 1 - (1 - sh) * (1 - m * op * style.shadow_strength / 0.55)
        bg *= (1 - sh)[..., None]
    if layout.bar_h:
        dark = style.frame == "dark"
        bar = np.empty((layout.bar_h, ww, 3), np.float32)
        top = np.array(hex_to_bgr("#3a3a3c" if dark else "#f2f2f4"), np.float32)
        bot = np.array(hex_to_bgr("#2c2c2e" if dark else "#e4e4e7"), np.float32)
        k = np.linspace(0, 1, layout.bar_h, dtype=np.float32)[:, None, None]
        bar[:] = top * (1 - k) + bot * k
        bar_u8 = np.ascontiguousarray(bar.astype(np.uint8))
        cy = layout.bar_h / 2
        r = 6.5 * u
        for i, col in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            cx = 20 * u + i * 20 * u
            cv2.circle(bar_u8, (int(round(cx * 16)), int(round(cy * 16))), int(round(r * 16)), hex_to_bgr(col), -1, cv2.LINE_AA, shift=4)
        line = hex_to_bgr("#1c1c1e" if dark else "#d4d4d8")
        bar_u8[-1, :] = line
        a = mask[wy:wy + layout.bar_h, wx:wx + ww][..., None]
        region = bg[wy:wy + layout.bar_h, wx:wx + ww]
        region[:] = bar_u8.astype(np.float32) * a + region * (1 - a)
    # dither to avoid banding in smooth gradients after H.264
    rng = np.random.default_rng(1234)
    bg += rng.uniform(-0.5, 0.5, size=(H, W, 1)).astype(np.float32)
    base = np.clip(np.round(bg), 0, 255).astype(np.uint8)

    # corner patches of the content rect where the mask is fractional
    cx0, cy0, cw, ch = layout.content
    patches = []
    r = layout.radius + 2
    if layout.radius > 0:
        for ys in (slice(cy0, cy0 + min(r, ch)), slice(cy0 + max(0, ch - r), cy0 + ch)):
            for xs in (slice(cx0, cx0 + min(r, cw)), slice(cx0 + max(0, cw - r), cx0 + cw)):
                a = mask[ys, xs]
                if np.all(a >= 0.999):
                    continue
                patches.append((ys, xs, a[..., None].copy(), base[ys, xs].astype(np.float32)))
    return Canvas(layout, base, patches)

"""Stream frames through the camera and the styled canvas into an H.264 mp4."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera, CameraParams, camera_motion, compute_camera
from .media import MediaInfo, VideoReader, VideoWriter, probe
from .plan import Plan
from .style import Canvas, Style, build_canvas, compute_layout


@dataclass
class RenderOptions:
    fps: Optional[float] = None        # output fps (default: source fps)
    scale: float = 1.0                 # render scale (0.5 for fast previews)
    size: Optional[Tuple[int, int]] = None
    motion_blur: bool = True
    blur_samples: int = 5
    blur_threshold: float = 4.0        # px/frame of camera motion before blur kicks in
    max_blur_px: float = 5.0           # longest smear (output px) a moving pixel gets
    crf: int = 18
    preset: str = "medium"
    interpolation: str = "cubic"       # cubic | lanczos | linear
    camera: Optional[CameraParams] = None


@dataclass
class RenderStats:
    frames: int
    seconds: float
    output: str
    width: int
    height: int

    @property
    def fps(self) -> float:
        return self.frames / max(self.seconds, 1e-9)


_INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR}


def warp_matrix(view, cw: int, ch: int) -> np.ndarray:
    """Affine map from a source rect (x0, y0, w, h) onto a cw x ch image,
    using pixel-center conventions so sub-pixel camera motion stays smooth."""
    x0, y0, w, h = view
    sx, sy = cw / w, ch / h
    return np.array([[sx, 0, sx * (0.5 - x0) - 0.5], [0, sy, sy * (0.5 - y0) - 0.5]], np.float64)


class FrameRenderer:
    """Turns source frames into finished output frames for a given camera."""

    def __init__(self, canvas: Canvas, cam: Camera, opts: RenderOptions):
        self.canvas = canvas
        self.cam = cam
        self.opts = opts
        self.interp = _INTERP.get(opts.interpolation, cv2.INTER_CUBIC)
        L = canvas.layout
        self.cx, self.cy, self.cw, self.ch = L.content
        self._acc = None

    def _warp(self, src, view, dst, interp):
        M = warp_matrix(view, self.cw, self.ch)
        return cv2.warpAffine(src, M, (self.cw, self.ch), dst=dst, flags=interp, borderMode=cv2.BORDER_REPLICATE)

    def render(self, src: np.ndarray, i: int) -> np.ndarray:
        cam = self.cam
        i = min(i, len(cam) - 1)
        out = self.canvas.base.copy()
        region = out[self.cy:self.cy + self.ch, self.cx:self.cx + self.cw]
        motion = camera_motion(cam, i, self.cw) if self.opts.motion_blur else 0.0
        if motion > self.opts.blur_threshold and i > 0:
            # shutter: blend warps sampled between the previous and current camera
            # short shutter, and the smear itself is capped at ~max_blur_px so text
            # stays readable mid-move (just enough blur to kill strobing)
            shutter = min(0.35, self.opts.max_blur_px / motion)
            n = int(min(self.opts.blur_samples, max(2, math.ceil(motion * shutter / 1.25) + 1)))
            if self._acc is None:
                self._acc = np.zeros((self.ch, self.cw, 3), np.float32)
            acc = self._acc
            acc[:] = 0
            a = (cam.cx[i - 1], cam.cy[i - 1], math.log(cam.zoom[i - 1]))
            b = (cam.cx[i], cam.cy[i], math.log(cam.zoom[i]))
            for k in range(n):
                f = 1.0 - shutter * (1 - k / (n - 1))
                v = cam.view_at(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                                math.exp(a[2] + (b[2] - a[2]) * f))
                w = self._warp(src, v, None, cv2.INTER_LINEAR if k < n - 1 else self.interp)
                cv2.accumulate(w, acc)
            cv2.convertScaleAbs(acc, region, alpha=1.0 / n)
        else:
            warped = self._warp(src, cam.view(i), region, self.interp)
            if warped is not None and warped.ctypes.data != region.ctypes.data:
                region[:] = warped
        for ys, xs, alpha, basep in self.canvas.patches:
            p = out[ys, xs]
            p[:] = (p.astype(np.float32) * alpha + basep * (1 - alpha) + 0.5).astype(np.uint8)
        return out


def prepare(info: MediaInfo, plan: Plan, style: Style, opts: RenderOptions):
    fps = float(opts.fps or info.fps)
    src_w = int(round(info.width * opts.scale)) & ~1
    src_h = int(round(info.height * opts.scale)) & ~1
    size = None
    if opts.size:
        size = (opts.size[0] * opts.scale, opts.size[1] * opts.scale)
    # style sizes are relative to the output, so previews look the same, just smaller
    layout = compute_layout(src_w, src_h, style, size)
    n_frames = max(1, int(round(info.duration * fps)))
    cam = compute_camera(plan, n_frames, fps, src_w, src_h, layout.base_w, layout.base_h,
                         follow_focus=layout.crop, params=opts.camera)
    canvas = build_canvas(layout, style)
    return fps, src_w, src_h, layout, cam, canvas


def render(
    input_path: str,
    output_path: str,
    plan: Plan,
    style: Optional[Style] = None,
    opts: Optional[RenderOptions] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> RenderStats:
    style = style or Style()
    opts = opts or RenderOptions()
    info = probe(input_path)
    fps, src_w, src_h, layout, cam, canvas = prepare(info, plan, style, opts)
    renderer = FrameRenderer(canvas, cam, opts)
    reader = VideoReader(input_path, src_w, src_h, fps=fps,
                         scale_flags="area" if opts.scale < 1 else "bicubic")
    writer = VideoWriter(output_path, layout.out_w, layout.out_h, fps, audio_source=input_path,
                         audio_info=info, crf=opts.crf, preset=opts.preset)
    total = len(cam)
    t0 = time.perf_counter()
    i = 0
    try:
        for i, frame in enumerate(reader):
            if cancel and cancel():
                raise KeyboardInterrupt("render cancelled")
            writer.write(renderer.render(frame, i))
            if progress:
                progress(i + 1, total)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return RenderStats(i + 1, time.perf_counter() - t0, output_path, layout.out_w, layout.out_h)


def render_still(input_path: str, t: float, plan: Plan, style: Optional[Style] = None,
                 opts: Optional[RenderOptions] = None) -> np.ndarray:
    """Render one output frame at time ``t`` (used for docs and the editor)."""
    style = style or Style()
    opts = opts or RenderOptions()
    info = probe(input_path)
    fps, src_w, src_h, layout, cam, canvas = prepare(info, plan, style, opts)
    renderer = FrameRenderer(canvas, cam, opts)
    target = int(round(t * fps))
    reader = VideoReader(input_path, src_w, src_h, fps=fps)
    last = None
    for i, frame in enumerate(reader):
        last = (i, frame)
        if i >= target:
            break
    if last is None:
        raise ValueError("could not decode any frame")
    return renderer.render(last[1], last[0])

import math

import numpy as np
import pytest

from autozoom.camera import compute_camera, spring_step
from autozoom.plan import Plan, Segment

W, H, FPS = 1920, 1080, 30


def _plan():
    return Plan(segments=[
        Segment(1.0, 4.0, 2.0, 0.95, 0.05),                       # corner: must clamp
        Segment(4.3, 7.0, 1.8, 0.2, 0.8, path=[[4.5, 0.2, 0.8], [6.5, 0.6, 0.7]]),
    ], duration=10)


def test_spring_converges_without_overshoot():
    x, v = 0.0, 0.0
    xs = []
    for _ in range(120):
        x, v = spring_step(x, v, 1.0, 5.5, 1 / 30)
        xs.append(x)
    assert max(xs) <= 1.0 + 1e-9
    assert xs[-1] > 0.999
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))  # monotonic


@pytest.mark.parametrize("base", [(W, H), (607.5, 1080)])
def test_view_stays_inside_frame(base):
    cam = compute_camera(_plan(), 300, FPS, W, H, base[0], base[1], follow_focus=base[0] < W)
    for i in range(len(cam)):
        x0, y0, w, h = cam.view(i)
        assert x0 >= -1e-6 and y0 >= -1e-6
        assert x0 + w <= W + 1e-6 and y0 + h <= H + 1e-6


def test_camera_is_smooth():
    cam = compute_camera(_plan(), 300, FPS, W, H)
    pan = np.hypot(np.diff(cam.cx), np.diff(cam.cy))
    dz = np.abs(np.diff(np.log(cam.zoom)))
    assert pan.max() < 60          # px per frame at 1080p: no jumps
    assert dz.max() < 0.06         # zoom changes gradually
    # velocity changes gradually too (no sudden starts)
    assert np.abs(np.diff(pan)).max() < 15


def test_reaches_zoom_and_returns():
    cam = compute_camera(_plan(), 300, FPS, W, H)
    assert cam.zoom[0] == pytest.approx(1.0)
    assert cam.zoom[int(3.5 * FPS)] == pytest.approx(2.0, abs=0.03)
    assert cam.zoom[-1] == pytest.approx(1.0, abs=0.01)
    # zoom starts gently (ease-in): small first-frame change
    i = int(1.0 * FPS)
    assert cam.zoom[i + 1] - cam.zoom[i] < 0.03


def test_no_plan_means_static_full_frame():
    cam = compute_camera(Plan(), 50, FPS, W, H)
    assert np.allclose(cam.zoom, 1) and np.allclose(cam.cx, W / 2) and np.allclose(cam.cy, H / 2)

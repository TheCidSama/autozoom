import json
import math

from autozoom.analyze import analyze
from autozoom.plan import Plan, Segment


def test_finds_each_scripted_action(demo_video, demo_plan):
    _, demo = demo_video
    gt = demo.ground_truth()
    assert demo_plan.segments, "expected zoom segments"
    for name, t0, t1, (gx, gy) in gt["zoom"]:
        tm = (t0 + t1) / 2
        seg = demo_plan.segment_at(tm)
        assert seg is not None, f"no zoom during {name} at {tm}s: {demo_plan.segments}"
        assert seg.zoom >= 1.4, f"{name}: zoom too small ({seg.zoom})"
        # the zoomed view must contain the action
        cx, cy = seg.center_at(tm)
        half_w, half_h = 0.5 / seg.zoom, 0.5 / seg.zoom
        assert abs(cx - gx) <= half_w and abs(cy - gy) <= half_h, (name, (cx, cy), (gx, gy), seg.zoom)


def test_no_zoom_when_idle_or_full_screen_change(demo_video, demo_plan):
    _, demo = demo_video
    for t0, t1 in demo.ground_truth()["no_zoom"]:
        t = t0
        while t < t1:
            assert demo_plan.segment_at(t) is None, f"unexpected zoom at {t:.2f}s"
            t += 0.1


def test_idle_recording_has_no_zooms(idle_video):
    plan = analyze(idle_video[0])
    assert plan.segments == []


def test_segments_are_sane(demo_plan):
    segs = demo_plan.segments
    for s in segs:
        assert s.end - s.start >= 1.0
        assert 1.0 <= s.zoom <= 2.0
        assert 0 <= s.center_x <= 1 and 0 <= s.center_y <= 1
    for a, b in zip(segs, segs[1:]):
        assert b.start >= a.end


def test_max_zoom_respected(demo_video):
    plan = analyze(demo_video[0], params=None)
    from autozoom.analyze import AnalysisParams

    p2 = analyze(demo_video[0], AnalysisParams(max_zoom=1.5))
    assert all(s.zoom <= 1.5 for s in p2.segments)
    assert len(p2.segments) >= 1 and len(plan.segments) >= 1


def test_plan_json_roundtrip(tmp_path, demo_plan):
    p = tmp_path / "plan.json"
    demo_plan.save(str(p))
    data = json.loads(p.read_text())
    assert data["version"] == 1 and "segments" in data
    back = Plan.load(str(p))
    assert len(back.segments) == len(demo_plan.segments)
    for a, b in zip(back.segments, demo_plan.segments):
        assert math.isclose(a.start, b.start, abs_tol=1e-3) and math.isclose(a.zoom, b.zoom, abs_tol=1e-3)


def test_plan_validation_fixes_overlaps():
    p = Plan.from_dict({"segments": [
        {"start": 1, "end": 4, "zoom": 2, "center_x": 0.2, "center_y": 0.2},
        {"start": 3, "end": 6, "zoom": 1.5, "center_x": 1.4, "center_y": 0.5},
        {"start": 7, "end": 6},
    ]})
    assert len(p.segments) == 2
    assert p.segments[1].start == 4 and p.segments[1].center_x == 1.0


def test_segment_path_interpolation():
    s = Segment(0, 4, 2, 0.5, 0.5, path=[[1, 0.2, 0.2], [3, 0.6, 0.4]])
    assert s.center_at(0) == (0.2, 0.2)
    x, y = s.center_at(2)
    assert math.isclose(x, 0.4) and math.isclose(y, 0.3)
    assert s.center_at(5) == (0.6, 0.4)


def test_plan_does_not_depend_on_source_fps(tmp_path):
    """Regression: a 60 fps source used to let the cursor tracker latch onto the
    menu-bar clock, so the pointer's departure after the chart click got zoomed."""
    from make_demo import make_demo

    plans = {}
    for fps in (30, 60):
        path = tmp_path / f"demo{fps}.mov"
        make_demo(str(path), 640, 360, fps, audio=False)
        plans[fps] = analyze(str(path)).segments
    a, b = plans[30], plans[60]
    assert len(a) == len(b) == 2, (a, b)
    for s30, s60 in zip(a, b):
        assert abs(s30.start - s60.start) <= 0.3 and abs(s30.end - s60.end) <= 0.3, (s30, s60)
        assert abs(s30.center_x - s60.center_x) <= 0.05 and abs(s30.center_y - s60.center_y) <= 0.05
    for segs in (a, b):
        assert segs[1].end < 19.2, segs[1]   # cursor wanders off at 19.0 s: not a reason to stay zoomed

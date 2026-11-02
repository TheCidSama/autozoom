import numpy as np
import pytest

from autozoom.cli import main
from autozoom.media import probe
from autozoom.plan import Plan, Segment
from autozoom.render import RenderOptions, render, render_still
from autozoom.style import Style, compute_layout, make_background


def test_end_to_end_with_audio(tmp_path, demo_video, demo_plan):
    src, _ = demo_video
    out = tmp_path / "out.mp4"
    stats = render(src, str(out), demo_plan, Style(background="sunset"), RenderOptions(preset="ultrafast"))
    info = probe(str(out))
    assert (info.width, info.height) == (640, 360)
    assert info.video_codec == "h264"
    assert info.has_audio
    assert abs(info.duration - 22.0) < 0.2
    assert stats.frames == 660


def test_vertical_without_audio(tmp_path, idle_video):
    src, _ = idle_video
    out = tmp_path / "v.mp4"
    plan = Plan(segments=[Segment(2, 5, 2.0, 0.8, 0.3)])
    render(src, str(out), plan, Style(aspect="9:16", background="ocean"), RenderOptions(preset="ultrafast"))
    info = probe(str(out))
    assert (info.width, info.height) == (360, 640)
    assert not info.has_audio
    assert abs(info.duration - 10.0) < 0.2


def test_fps_override(tmp_path, idle_video):
    out = tmp_path / "f.mp4"
    render(idle_video[0], str(out), Plan(), Style(), RenderOptions(fps=15, preset="ultrafast"))
    info = probe(str(out))
    assert info.fps == pytest.approx(15, abs=0.1)
    assert abs(info.duration - 10.0) < 0.2


def test_frame_has_background_corners_and_no_black_edges(demo_video, demo_plan):
    src, _ = demo_video
    style = Style(background="#336699", radius=24, padding="10%")
    seg = demo_plan.segments[0]
    t = (seg.start + seg.end) / 2
    img = render_still(src, t, demo_plan, style)
    L = compute_layout(640, 360, style)
    x, y, w, h = L.content
    # outside the window: background color
    assert np.abs(img[2, 2].astype(int) - (153, 102, 51)).max() <= 3
    # very corner of the rounded window shows background, not video
    assert np.abs(img[y, x].astype(int) - (153, 102, 51)).max() <= 40
    # zoomed content has no black borders from warping
    inner = img[y + h // 4: y + 3 * h // 4, x + 2: x + w - 2]
    assert inner.min(axis=2).max() > 100
    assert not (img[y + 5: y + h - 5, x + 3].max(axis=1) < 10).any()


def test_backgrounds():
    for spec in ["sunset", "midnight", "#fff", "#123456"]:
        bg = make_background(spec, 64, 36)
        assert bg.shape == (36, 64, 3)
    with pytest.raises(ValueError):
        make_background("nope-not-a-thing", 10, 10)


def test_cli_save_plan_then_render(tmp_path, idle_video):
    src, _ = idle_video
    plan = tmp_path / "p.json"
    out = tmp_path / "c.mp4"
    assert main([src, "--analyze-only", "--save-plan", str(plan), "-q"]) == 0
    assert plan.exists()
    assert main([src, "--plan", str(plan), "-o", str(out), "--aspect", "1:1", "--preset", "ultrafast",
                 "--frame", "dark", "-q"]) == 0
    info = probe(str(out))
    assert (info.width, info.height) == (360, 360)


def test_cli_missing_file():
    assert main(["/nonexistent/file.mov", "-q"]) == 2

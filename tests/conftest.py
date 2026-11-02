import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from make_demo import Demo, make_demo  # noqa: E402


@pytest.fixture(scope="session")
def demo_video(tmp_path_factory):
    """22 s scripted demo at 640x360/30fps with a sine audio track."""
    path = tmp_path_factory.mktemp("vid") / "demo.mov"
    demo = make_demo(str(path), 640, 360, 30, audio=True, scenario="demo")
    return str(path), demo


@pytest.fixture(scope="session")
def idle_video(tmp_path_factory):
    """10 s of 'nothing happening': blinking caret, ticking clock, cursor wandering. No audio."""
    path = tmp_path_factory.mktemp("vid") / "idle.mp4"
    demo = make_demo(str(path), 640, 360, 30, audio=False, scenario="idle")
    return str(path), demo


@pytest.fixture(scope="session")
def demo_plan(demo_video):
    from autozoom.analyze import analyze

    return analyze(demo_video[0])

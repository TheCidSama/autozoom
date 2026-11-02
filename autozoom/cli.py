"""Command line interface.

    autozoom input.mov [-o out.mp4] [--zoom 1.8] [--background sunset] ...
    autozoom edit input.mov            # local web editor
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .style import ASPECTS, PRESETS, Style


def _size(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1920x1080")


def _bool_flag(p, name, default, help):
    g = p.add_mutually_exclusive_group()
    g.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help)
    g.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    p.set_defaults(**{name.replace("-", "_"): default})


def add_style_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("style")
    g.add_argument("--background", "-b", default="sunset",
                   help=f"preset ({', '.join(PRESETS)}), #hex color or image path (default: sunset)")
    g.add_argument("--padding", default="8%", help="space around the video: '8%%' of the short side or pixels (default: 8%%)")
    g.add_argument("--radius", type=float, default=16, help="corner radius in px at 1080p (default: 16)")
    _bool_flag(g, "shadow", True, "soft drop shadow (default: on)")
    g.add_argument("--frame", choices=["none", "light", "dark"], default="none", help="macOS-style window title bar")
    g.add_argument("--aspect", default="source", help=f"output aspect: {', '.join(ASPECTS)} (default: source)")
    g.add_argument("--fit", choices=["auto", "fit", "crop"], default="auto",
                   help="fit the whole recording, or crop to the output aspect and follow the action (auto: crop when aspects differ a lot, e.g. 9:16)")


def style_from_args(a) -> Style:
    return Style(background=a.background, padding=a.padding, radius=a.radius, shadow=a.shadow,
                 frame=a.frame, aspect=a.aspect, fit=a.fit)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autozoom",
        description="Turn any screen recording into a polished demo: automatic smooth zooms, "
                    "background, padding, rounded corners and shadow. Free and local.",
        epilog="Web editor: autozoom edit input.mov",
    )
    p.add_argument("input", help="screen recording (mov, mp4, webm, mkv, ...)")
    p.add_argument("-o", "--output", help="output mp4 (default: <input>.autozoom.mp4)")
    p.add_argument("--version", action="version", version=f"autozoom {__version__}")
    z = p.add_argument_group("zoom")
    z.add_argument("--zoom", type=float, default=2.0, help="maximum zoom (default: 2.0)")
    z.add_argument("--plan", help="use this zoom plan JSON instead of analyzing")
    z.add_argument("--save-plan", metavar="PATH", help="write the zoom plan JSON (edit it, then re-render with --plan)")
    z.add_argument("--analyze-only", action="store_true", help="only analyze (with --save-plan), don't render")
    z.add_argument("--no-zoom", action="store_true", help="no zooms, just the styled frame")
    add_style_args(p)
    o = p.add_argument_group("output")
    o.add_argument("--fps", type=float, help="output frame rate (default: same as input)")
    o.add_argument("--size", type=_size, help="exact output size, e.g. 1920x1080")
    o.add_argument("--preview", action="store_true", help="fast half-resolution draft (<input>.preview.mp4)")
    _bool_flag(o, "motion-blur", True, "motion blur during fast camera moves (default: on)")
    o.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better (default: 18)")
    o.add_argument("--preset", default="medium", help="x264 speed preset (default: medium)")
    o.add_argument("-q", "--quiet", action="store_true")
    return p


def edit_main(argv) -> int:
    p = argparse.ArgumentParser(prog="autozoom edit", description="Open the local web editor for a recording.")
    p.add_argument("input")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    p.add_argument("--zoom", type=float, default=2.0, help="maximum zoom for auto analysis")
    a = p.parse_args(argv)
    from .editor.server import serve

    return serve(a.input, host=a.host, port=a.port, open_browser=not a.no_browser, max_zoom=a.zoom)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "edit":
        return edit_main(argv[1:])
    a = build_parser().parse_args(argv)
    from .analyze import AnalysisParams, analyze
    from .plan import Plan
    from .progress import ProgressBar
    from .render import RenderOptions, render

    if not os.path.exists(a.input):
        print(f"autozoom: no such file: {a.input}", file=sys.stderr)
        return 2
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, file=sys.stderr))
    try:
        style = style_from_args(a)
        style.aspect = a.aspect
        if a.plan:
            plan = Plan.load(a.plan)
            log(f"Loaded plan {a.plan}: {len(plan.segments)} zoom segment(s)")
        elif a.no_zoom:
            plan = Plan()
        else:
            bar = None if a.quiet else ProgressBar("Analyzing")
            t0 = time.perf_counter()
            plan = analyze(a.input, AnalysisParams(max_zoom=a.zoom), progress=bar)
            if bar:
                bar.close()
            log(f"Found {len(plan.segments)} zoom segment(s) in {time.perf_counter() - t0:.1f}s")
            for s in plan.segments:
                log(f"  {s.start:6.2f}s - {s.end:6.2f}s  x{s.zoom:.2f} at ({s.center_x:.2f}, {s.center_y:.2f})  [{s.kind}]")
        if a.save_plan:
            plan.save(a.save_plan)
            log(f"Saved plan to {a.save_plan}")
        if a.analyze_only:
            return 0
        stem, _ = os.path.splitext(a.input)
        out = a.output or (stem + (".preview.mp4" if a.preview else ".autozoom.mp4"))
        if os.path.abspath(out) == os.path.abspath(a.input):
            print("autozoom: output would overwrite the input", file=sys.stderr)
            return 2
        opts = RenderOptions(fps=a.fps, size=a.size, motion_blur=a.motion_blur and not a.preview,
                             crf=a.crf, preset=a.preset)
        if a.preview:
            opts.scale, opts.preset, opts.crf, opts.interpolation = 0.5, "ultrafast", 23, "linear"
        bar = None if a.quiet else ProgressBar("Rendering")
        stats = render(a.input, out, plan, style, opts, progress=bar)
        if bar:
            bar.close()
        log(f"Wrote {out} ({stats.width}x{stats.height}, {stats.frames} frames, "
            f"{stats.seconds:.1f}s, {stats.fps:.1f} fps)")
        return 0
    except KeyboardInterrupt:
        print("\nautozoom: interrupted", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"autozoom: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

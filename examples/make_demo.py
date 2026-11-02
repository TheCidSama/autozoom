#!/usr/bin/env python3
"""Generate a synthetic but realistic-looking screen recording for autozoom.

It draws a fake desktop app (menu bar with a ticking clock, sidebar, header with
a blinking caret, cards, a form, a chart) and scripts a little product demo:

    0.0 - 2.0   idle (caret blinks, clock ticks: should NOT trigger a zoom)
    2.0 - 3.2   cursor glides to "New project" (top right)
    3.3         click -> dropdown menu opens, hover moves down the items
    4.8         click a menu item -> menu closes
    5.4 - 6.6   cursor glides to the form field (bottom left)
    6.7         click -> focus ring
    7.0 - 10.0  typing "Quarterly launch plan"
    11.0 - 12.5 full-screen scroll of the content (should NOT zoom)
    14.5 - 15.5 cursor glides to the chart card (right)
    15.7        click "Refresh" -> bars animate until 18.0
    19.0 - 19.8 cursor wanders away (cursor-only motion, no zoom)

Usage:
    python examples/make_demo.py demo.mov [--size 1920x1080] [--fps 30] [--no-audio]
    python examples/make_demo.py idle.mov --scenario idle

The same module is imported by the test-suite, which renders tiny versions.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from autozoom.media import ffmpeg_exe
except ImportError:  # running from a checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from autozoom.media import ffmpeg_exe

REF_W, REF_H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---- palette (BGR) ---------------------------------------------------------
MENUBAR = (246, 244, 243)
SIDEBAR = (48, 36, 31)
SIDEBAR_HI = (82, 64, 56)
MAIN_BG = (251, 248, 246)
CARD = (255, 255, 255)
BORDER = (232, 226, 222)
TEXT = (52, 41, 33)
MUTED = (150, 140, 132)
BAR = (228, 222, 218)
BLUE = (235, 99, 37)
BLUE_DARK = (190, 70, 20)
BLUE_LIGHT = (253, 236, 222)
GREEN = (120, 190, 60)

SCROLL_PX = 520          # content scroll distance (reference px)
CONTENT_TOP = 120        # where the scrollable content starts on screen
CONTENT_H = 2000

# Key reference-space positions
BTN = (1680, 50, 1880, 94)                    # "New project" button
MENU = (1630, 104, 1880, 304)
MENU_ITEMS = ["Blank project", "From template", "Import...", "Duplicate"]
FIELD = (340, 870, 900, 922)                  # form input (content coords + CONTENT_TOP before scroll)
SEARCH = (700, 50, 1120, 94)
CHART = (1220, 700, 1880, 1080)               # chart card in content coords (pre-scroll screen coords)
CHART_BTN = (1720, 724, 1850, 762)
NAV = ["Home", "Projects", "Analytics", "Team", "Settings"]


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x) if x < 1 else 1.0


def ease_io_cubic(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


class Painter:
    def __init__(self, width: int, height: int):
        self.W, self.H = width, height
        self.s = width / REF_W
        self.sy = height / REF_H

    def p(self, x, y):
        return int(round(x * self.s)), int(round(y * self.sy))

    def rect(self, img, box, color, thick=-1, radius=0):
        x0, y0 = self.p(box[0], box[1])
        x1, y1 = self.p(box[2], box[3])
        if radius <= 0 or thick != -1:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thick if thick == -1 else max(1, int(thick * self.s)), cv2.LINE_AA)
            return
        r = max(1, int(radius * self.s))
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)

    def text(self, img, s, x, y, scale=0.8, color=TEXT, thick=2):
        cv2.putText(img, s, self.p(x, y), FONT, scale * self.s, color, max(1, int(round(thick * self.s))), cv2.LINE_AA)


def build_static(P: Painter):
    """Static chrome (menu bar, sidebar, header) and the tall scrollable content."""
    W, H = P.W, P.H
    chrome = np.empty((H, W, 3), np.uint8)
    chrome[:] = MAIN_BG
    P.rect(chrome, (0, 0, REF_W, 32), MENUBAR)
    P.text(chrome, "Finder   File   Edit   View   Window   Help", 20, 23, 0.55, TEXT, 1)
    P.rect(chrome, (0, 32, 280, REF_H), SIDEBAR)
    P.text(chrome, "Acme Studio", 32, 82, 0.85, (250, 250, 250), 2)
    for i, name in enumerate(NAV):
        y = 140 + i * 58
        if name == "Projects":
            P.rect(chrome, (16, y - 30, 264, y + 16), SIDEBAR_HI, radius=10)
        cv2.circle(chrome, P.p(44, y - 7), max(1, int(8 * P.s)), (200, 190, 180), -1, cv2.LINE_AA)
        P.text(chrome, name, 66, y, 0.7, (235, 230, 225), 1)
    # header
    P.rect(chrome, (280, 32, REF_W, 112), CARD)
    P.rect(chrome, (280, 111, REF_W, 113), BORDER)
    P.text(chrome, "Projects", 320, 86, 1.1, TEXT, 2)
    P.rect(chrome, SEARCH, (244, 240, 238), radius=10)
    P.text(chrome, "Search projects...", SEARCH[0] + 20, SEARCH[1] + 30, 0.6, MUTED, 1)

    content = np.empty((int(round(CONTENT_H * P.sy)), W, 3), np.uint8)
    content[:] = MAIN_BG
    rng = np.random.default_rng(7)

    def cy(y):  # content-space y (reference) -> content image y
        return y - CONTENT_TOP

    # project cards grid
    for row in range(2):
        for col in range(3):
            x0 = 320 + col * 530
            y0 = cy(150 + row * 270)
            P.rect(content, (x0, y0, x0 + 500, y0 + 240), CARD, radius=14)
            P.rect(content, (x0 + 24, y0 + 24, x0 + 84, y0 + 84), (230 - col * 30, 200, 160 + row * 40), radius=12)
            P.text(content, f"Project {row * 3 + col + 1:02d}", x0 + 104, y0 + 52, 0.75, TEXT, 2)
            P.text(content, "Updated 2 days ago", x0 + 104, y0 + 80, 0.55, MUTED, 1)
            for k in range(3):
                w = int(rng.integers(220, 440))
                P.rect(content, (x0 + 24, y0 + 120 + k * 32, x0 + 24 + w, y0 + 134 + k * 32), BAR, radius=6)
    # form card
    P.rect(content, (320, cy(720), 1180, cy(1060)), CARD, radius=14)
    P.text(content, "New project", 340, cy(770), 0.85, TEXT, 2)
    P.text(content, "Project name", 340, cy(850), 0.6, MUTED, 1)
    P.rect(content, (FIELD[0], cy(FIELD[1]), FIELD[2], cy(FIELD[3])), (246, 243, 241), radius=8)
    P.rect(content, (340, cy(960), 520, cy(1010)), BLUE, radius=10)
    P.text(content, "Create", 392, cy(993), 0.7, (255, 255, 255), 2)
    # chart card (static parts)
    P.rect(content, (CHART[0], cy(CHART[1]), CHART[2], cy(CHART[3])), CARD, radius=14)
    P.text(content, "Weekly active users", CHART[0] + 24, cy(CHART[1] + 46), 0.75, TEXT, 2)
    P.rect(content, (CHART_BTN[0], cy(CHART_BTN[1]), CHART_BTN[2], cy(CHART_BTN[3])), BLUE_LIGHT, radius=8)
    P.text(content, "Refresh", CHART_BTN[0] + 20, cy(CHART_BTN[1] + 26), 0.55, BLUE, 2)
    # lots of text rows further down so the scroll changes most of the screen
    for i in range(26):
        y = cy(1110 + i * 34)
        w = int(rng.integers(500, 1500))
        P.rect(content, (320, y, 320 + w, y + 14), BAR, radius=6)
    return chrome, content


class Demo:
    """Scripted demo. ``frame(t)`` returns a BGR uint8 image."""

    duration = 22.0

    def __init__(self, width=1920, height=1080, scenario="demo"):
        self.P = Painter(width, height)
        self.scenario = scenario
        self.chrome, self.content = build_static(self.P)
        self.typed = "Quarterly launch plan"
        if scenario == "idle":
            self.duration = 10.0
        # cursor keyframes: (t_start, t_end, from, to)
        self.moves = [
            (2.0, 3.2, (960, 600), (1782, 74)),
            (3.7, 4.0, (1782, 74), (1760, 150)),
            (4.2, 4.6, (1760, 150), (1750, 196)),
            (5.4, 6.6, (1750, 196), (560, 896)),
            (14.5, 15.5, (560, 896), (1785, 743 - SCROLL_PX)),
            (19.0, 19.8, (1785, 743 - SCROLL_PX), (900, 640)),
        ]
        if scenario == "idle":
            self.moves = [(3.0, 4.0, (960, 600), (1000, 640)), (6.0, 7.5, (1000, 640), (700, 500))]

    # ---- timeline helpers -------------------------------------------------
    def cursor(self, t):
        pos = self.moves[0][2]
        for t0, t1, a, b in self.moves:
            if t >= t1:
                pos = b
            elif t >= t0:
                k = ease_io_cubic((t - t0) / (t1 - t0))
                pos = (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
                break
            else:
                break
        return pos

    def scroll(self, t):
        if self.scenario != "demo":
            return 0.0
        return SCROLL_PX * ease_io_cubic((t - 11.0) / 1.5)

    # ---- drawing ----------------------------------------------------------
    def frame(self, t: float) -> np.ndarray:
        P = self.P
        img = self.chrome.copy()
        top = int(round(CONTENT_TOP * P.sy))
        off = int(round(self.scroll(t) * P.sy))
        visible = P.H - top
        img[top:, P.p(280, 0)[0]:] = self.content[off: off + visible, P.p(280, 0)[0]:]
        sy = -self.scroll(t)  # reference-space offset applied to content items

        def cbox(b):
            return (b[0], b[1] + sy, b[2], b[3] + sy)

        # clock (ticks every second -> tiny change, should be ignored)
        sec = int(t)
        P.rect(img, (1700, 0, REF_W, 32), MENUBAR)
        P.text(img, f"Mon 9:41:{sec % 60:02d}", 1730, 23, 0.55, TEXT, 1)
        # header redraw (content blit covered nothing above top, header is chrome)
        demo = self.scenario == "demo"
        # search caret blink (when field not focused)
        focused = demo and t >= 6.7
        if not focused and int(t * 2) % 2 == 0:
            P.rect(img, (SEARCH[0] + 17, SEARCH[1] + 10, SEARCH[0] + 19, SEARCH[3] - 10), TEXT)
        # New project button
        pressed = demo and 3.3 <= t < 3.45
        P.rect(img, BTN, BLUE_DARK if pressed else BLUE, radius=10)
        P.text(img, "+ New project", BTN[0] + 18, BTN[1] + 30, 0.62, (255, 255, 255), 2)
        # dropdown menu
        if demo and 3.45 <= t < 4.9:
            P.rect(img, (MENU[0] + 4, MENU[1] + 6, MENU[2] + 4, MENU[3] + 8), (225, 220, 216), radius=12)
            P.rect(img, MENU, CARD, radius=12)
            cx, cyy = self.cursor(t)
            for i, name in enumerate(MENU_ITEMS):
                y0 = MENU[1] + 10 + i * 46
                if MENU[0] <= cx <= MENU[2] and y0 <= cyy < y0 + 46:
                    P.rect(img, (MENU[0] + 8, y0, MENU[2] - 8, y0 + 42), BLUE_LIGHT, radius=8)
                P.text(img, name, MENU[0] + 24, y0 + 29, 0.62, TEXT, 1)
        # form field focus + typing
        if demo:
            f = cbox(FIELD)
            if focused:
                P.rect(img, (f[0] - 3, f[1] - 3, f[2] + 3, f[3] + 3), BLUE, radius=10)
                P.rect(img, f, (255, 255, 255), radius=8)
            n = 0 if t < 7.0 else min(len(self.typed), int((t - 7.0) / 3.0 * len(self.typed)) + 1)
            txt = self.typed[:n]
            if n:
                P.text(img, txt, f[0] + 16, f[1] + 35, 0.75, TEXT, 2)
            if focused and (t < 10.0 or int(t * 2) % 2 == 0):
                (tw, _), _ = cv2.getTextSize(txt, FONT, 0.75, 2)
                x = f[0] + 18 + tw
                P.rect(img, (x, f[1] + 12, x + 2, f[3] - 12), TEXT)
            # chart bars
            base_y = CHART[3] - 40 + sy
            heights = [120, 180, 150, 230, 200, 260, 240]
            grow = 0.35 + 0.65 * ease((t - 15.8) / 2.2) if t >= 15.8 else 0.35
            for i, hgt in enumerate(heights):
                x0 = CHART[0] + 40 + i * 82
                hh = hgt * (grow if t >= 15.8 else 0.35 + 0.0 * i)
                if t >= 15.8:
                    # stagger the bars a little like a real chart animation
                    gi = 0.35 + 0.65 * ease((t - 15.8 - i * 0.12) / 1.6)
                    hh = hgt * gi
                if base_y - hh < CONTENT_TOP:
                    continue
                P.rect(img, (x0, base_y - hh, x0 + 52, base_y), GREEN if i == 5 else (230, 170, 120), radius=6)
            if 15.7 <= t < 15.85:
                b = cbox(CHART_BTN)
                P.rect(img, b, (240, 210, 190), radius=8)
                P.text(img, "Refresh", b[0] + 20, b[1] + 26, 0.55, BLUE, 2)
        # re-draw the menu bar over content where needed
        self._draw_cursor(img, *self.cursor(t))
        return img

    def _draw_cursor(self, img, x, y):
        P = self.P
        k = 1.25 * P.s
        pts = np.array([(0, 0), (0, 25), (6, 19.5), (10.5, 29), (14.5, 27.2), (10.2, 18.2), (18, 18)], np.float64)
        pts = pts * k + (x * P.s, y * P.sy)
        ip = np.round(pts * 16).astype(np.int32)
        cv2.fillPoly(img, [ip], (255, 255, 255), cv2.LINE_AA, shift=4)
        cv2.polylines(img, [ip], True, (0, 0, 0), max(1, int(round(1.4 * P.s))), cv2.LINE_AA, shift=4)

    def ground_truth(self):
        """Normalized regions/time-windows the analysis is expected to find."""
        if self.scenario != "demo":
            return {"zoom": [], "no_zoom": [(0.0, self.duration)]}
        return {
            "zoom": [
                ("menu", 3.4, 4.8, ((MENU[0] + MENU[2]) / 2 / REF_W, (BTN[1] + MENU[3]) / 2 / REF_H)),
                ("typing", 7.2, 9.8, ((FIELD[0] + 300) / REF_W, (FIELD[1] + FIELD[3]) / 2 / REF_H)),
                ("chart", 16.0, 17.6, ((CHART[0] + CHART[2]) / 2 / REF_W, (CHART[1] + CHART[3]) / 2 / REF_H - SCROLL_PX / REF_H)),
            ],
            "no_zoom": [(0.0, 2.0), (11.2, 12.4)],
        }


def make_demo(out: str, width=1920, height=1080, fps=30.0, audio=True, scenario="demo", crf=20, duration=None):
    demo = Demo(width, height, scenario)
    dur = duration or demo.duration
    n = int(round(dur * fps))
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={dur}"]
    cmd += ["-map", "0:v"]
    if audio:
        cmd += ["-map", "1:a", "-c:a", "aac", "-b:a", "128k", "-af", "volume=0.25"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-shortest", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            proc.stdin.write(demo.frame(i / fps).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg failed while writing demo")
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default="demo.mov")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scenario", choices=["demo", "idle"], default="demo")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    make_demo(a.output, w, h, a.fps, audio=not a.no_audio, scenario=a.scenario)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()

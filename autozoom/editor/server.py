"""Local web editor: ``autozoom edit input.mov``.

A stdlib ``http.server`` serving a vanilla HTML/JS app plus a tiny JSON API.
Nothing leaves your machine; the server binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from ..analyze import AnalysisParams, analyze
from ..media import VideoReader, ffmpeg_exe, probe
from ..plan import Plan
from ..render import RenderOptions, render
from ..style import ASPECTS, PRESETS, Style

STATIC = Path(__file__).resolve().parent / "static"


class Project:
    def __init__(self, input_path: str, max_zoom: float = 2.0, log=print):
        self.input = os.path.abspath(input_path)
        self.info = probe(self.input)
        stem, _ = os.path.splitext(self.input)
        self.plan_path = stem + ".autozoom.json"
        self.output_path = stem + ".autozoom.mp4"
        self.preview_path = stem + ".preview.mp4"
        self.tmp = tempfile.mkdtemp(prefix="autozoom-")
        self.proxy = os.path.join(self.tmp, "proxy.mp4")
        self.filmstrip = os.path.join(self.tmp, "filmstrip.jpg")
        self.log = log
        self.lock = threading.Lock()
        self.job = {"state": "idle"}
        self._cancel = False
        if os.path.exists(self.plan_path):
            self.plan = Plan.load(self.plan_path)
            log(f"Loaded plan {self.plan_path}")
        else:
            log("Analyzing recording...")
            self.plan = analyze(self.input, AnalysisParams(max_zoom=max_zoom))
            self.plan.save(self.plan_path)
            log(f"Found {len(self.plan.segments)} zoom segment(s); saved {self.plan_path}")
        self._make_proxy()
        self._make_filmstrip()

    def _make_proxy(self):
        """Browser-friendly H.264 proxy (<=720p, frequent keyframes for snappy seeking)."""
        self.log("Preparing preview proxy...")
        h = min(720, self.info.height)
        cmd = [ffmpeg_exe(), "-y", "-v", "error", "-nostdin", "-i", self.input, "-map", "0:v:0", "-map", "0:a:0?",
               "-vf", f"scale=-2:{h - h % 2}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", self.proxy]
        subprocess.run(cmd, check=True)

    def _make_filmstrip(self, n: int = 16, th: int = 72):
        d = max(self.info.duration, 0.1)
        tw = int(round(th * self.info.width / self.info.height)) & ~1
        thumbs = []
        for f in VideoReader(self.proxy, tw, th, fps=n / d):
            thumbs.append(f)
            if len(thumbs) >= n:
                break
        if not thumbs:
            thumbs = [np.zeros((th, tw, 3), np.uint8)]
        cv2.imwrite(self.filmstrip, np.hstack(thumbs), [cv2.IMWRITE_JPEG_QUALITY, 80])

    def project_json(self) -> dict:
        return {
            "name": os.path.basename(self.input),
            "input": self.input,
            "info": self.info.to_dict(),
            "plan": self.plan.to_dict(),
            "presets": {k: {"angle": a, "stops": s} for k, (a, s) in PRESETS.items()},
            "aspects": list(ASPECTS),
            "output": self.output_path,
            "plan_path": self.plan_path,
        }

    def set_plan(self, data: dict):
        plan = Plan.from_dict(data)
        # keep metadata the browser doesn't send
        plan.source, plan.width, plan.height = self.input, self.info.width, self.info.height
        plan.fps, plan.duration = self.info.fps, self.info.duration
        if not plan.focus:
            plan.focus = self.plan.focus
        self.plan = plan
        plan.save(self.plan_path)
        return plan

    def start_render(self, plan_data: dict, style_data: dict, preview: bool):
        with self.lock:
            if self.job.get("state") == "running":
                raise RuntimeError("a render is already running")
            plan = self.set_plan(plan_data) if plan_data else self.plan
            style = Style(
                background=str(style_data.get("background", "sunset")),
                padding=str(style_data.get("padding", "8%")),
                radius=float(style_data.get("radius", 16)),
                shadow=bool(style_data.get("shadow", True)),
                frame=str(style_data.get("frame", "none")),
                aspect=str(style_data.get("aspect", "source")),
            )
            opts = RenderOptions(motion_blur=bool(style_data.get("motion_blur", True)) and not preview)
            out = self.preview_path if preview else self.output_path
            if preview:
                opts.scale, opts.preset, opts.crf, opts.interpolation = 0.5, "ultrafast", 23, "linear"
            self._cancel = False
            self.job = {"state": "running", "done": 0, "total": 1, "fps": 0.0, "eta": None,
                        "output": out, "started": time.time(), "preview": preview}

        def progress(done, total):
            el = time.time() - self.job["started"]
            fps = done / el if el > 0 else 0
            self.job.update(done=done, total=total, fps=round(fps, 1),
                            eta=round((total - done) / fps, 1) if fps > 0 else None)

        def run():
            try:
                stats = render(self.input, out, plan, style, opts, progress=progress, cancel=lambda: self._cancel)
                self.job.update(state="done", done=stats.frames, total=stats.frames, fps=round(stats.fps, 1),
                                seconds=round(stats.seconds, 1), eta=0)
                self.log(f"Rendered {out} ({stats.fps:.1f} fps)")
            except BaseException as e:  # noqa: BLE001 - report everything to the UI
                self.job.update(state="cancelled" if self._cancel else "error", error=str(e))

        threading.Thread(target=run, daemon=True).start()

    def cleanup(self):
        self._cancel = True
        shutil.rmtree(self.tmp, ignore_errors=True)


def make_handler(project: Project):
    class Handler(BaseHTTPRequestHandler):
        server_version = "autozoom"

        def log_message(self, fmt, *args):  # quiet
            pass

        # ---- helpers -------------------------------------------------
        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 5_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _file(self, path: str, ctype: str = None):
            if not os.path.exists(path):
                return self._json({"error": "not found"}, 404)
            size = os.path.getsize(path)
            ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = HTTPStatus.OK
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        if m.group(2):
                            end = min(int(m.group(2)), size - 1)
                    elif m.group(2):
                        start = max(0, size - int(m.group(2)))
                    if start >= size:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ---- routes --------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._file(str(STATIC / "index.html"), "text/html; charset=utf-8")
            if path in ("/app.js", "/style.css"):
                ctype = "text/javascript" if path.endswith(".js") else "text/css"
                return self._file(str(STATIC / path[1:]), ctype + "; charset=utf-8")
            if path == "/video":
                return self._file(project.proxy, "video/mp4")
            if path == "/filmstrip.jpg":
                return self._file(project.filmstrip, "image/jpeg")
            if path == "/output":
                out = project.job.get("output")
                if project.job.get("state") != "done" or not out:
                    return self._json({"error": "no output yet"}, 404)
                return self._file(out, "video/mp4")
            if path == "/api/project":
                return self._json(project.project_json())
            if path == "/api/render":
                return self._json(project.job)
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                data = self._body()
                if path == "/api/plan":
                    plan = project.set_plan(data)
                    return self._json({"ok": True, "plan": plan.to_dict()})
                if path == "/api/analyze":
                    mz = float(data.get("max_zoom", 2.0))
                    project.plan = analyze(project.input, AnalysisParams(max_zoom=mz))
                    project.plan.save(project.plan_path)
                    return self._json({"ok": True, "plan": project.plan.to_dict()})
                if path == "/api/render":
                    project.start_render(data.get("plan"), data.get("style") or {}, bool(data.get("preview")))
                    return self._json({"ok": True, "job": project.job})
                if path == "/api/cancel":
                    project._cancel = True
                    return self._json({"ok": True})
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"error": "not found"}, 404)

    return Handler


def serve(input_path: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          max_zoom: float = 2.0) -> int:
    if not os.path.exists(input_path):
        print(f"autozoom: no such file: {input_path}", file=sys.stderr)
        return 2
    log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731
    project = Project(input_path, max_zoom=max_zoom, log=log)
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), make_handler(project))
            break
        except OSError:
            continue
    if httpd is None:
        log(f"autozoom: no free port in {port}-{port + 19}")
        project.cleanup()
        return 1
    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}/"
    log(f"autozoom editor running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("\nStopping editor.")
    finally:
        httpd.server_close()
        project.cleanup()
    return 0

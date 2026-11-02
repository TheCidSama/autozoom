"""Thin wrappers around the ffmpeg binary bundled with ``imageio-ffmpeg``.

Everything streams through pipes (raw BGR / gray frames), so memory use stays
constant no matter how long the recording is, and users never need a system
ffmpeg install.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


def ffmpeg_exe() -> str:
    env = os.environ.get("AUTOZOOM_FFMPEG")
    if env:
        return env
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    audio_codec: Optional[str] = None
    video_codec: Optional[str] = None

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "video_codec": self.video_codec,
        }


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})[\s,\[]")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")
_TBR_RE = re.compile(r"([\d.]+)k?\s*tbr")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?: Audio: (\w+)")
_ROT_RE = re.compile(r"rotation of (-?[\d.]+) degrees")


def _hms(m: re.Match) -> float:
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def probe(path: str) -> MediaInfo:
    """Read basic stream info by parsing ``ffmpeg -i`` (no ffprobe needed)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path],
        capture_output=True,
        text=True,
        errors="replace",
    )
    err = proc.stderr
    vm = _VIDEO_RE.search(err)
    if not vm:
        raise ValueError(f"No video stream found in {path!r}")
    vcodec, w, h = vm.group(1), int(vm.group(2)), int(vm.group(3))
    line = err[vm.start(): err.find("\n", vm.start())]
    fps = None
    fm = _FPS_RE.search(line)
    if fm:
        fps = float(fm.group(1))
    else:
        tm = _TBR_RE.search(line)
        if tm:
            fps = float(tm.group(1))
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    rm = _ROT_RE.search(err)
    if rm and abs(round(float(rm.group(1)))) % 180 == 90:
        w, h = h, w  # ffmpeg auto-rotates on decode
    am = _AUDIO_RE.search(err)
    dm = _DUR_RE.search(err)
    duration = _hms(dm) if dm else 0.0
    if duration <= 0:
        duration = _scan_duration(path)
    return MediaInfo(
        path=path,
        width=w,
        height=h,
        fps=fps,
        duration=duration,
        has_audio=am is not None,
        audio_codec=am.group(1) if am else None,
        video_codec=vcodec,
    )


def _scan_duration(path: str) -> float:
    """Fallback for containers without a duration header (e.g. some webm)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    times = _TIME_RE.findall(proc.stderr)
    if not times:
        return 0.0
    hh, mm, ss = times[-1]
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


class VideoReader:
    """Iterate decoded frames as numpy arrays, optionally resampled/rescaled.

    Decoding happens in the ffmpeg subprocess; a small prefetch thread keeps
    the pipe drained so decode overlaps with our processing.
    """

    def __init__(
        self,
        path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        gray: bool = False,
        prefetch: int = 6,
        scale_flags: str = "area",
    ):
        self.path = path
        info = probe(path)
        self.info = info
        self.width = int(width or info.width)
        self.height = int(height or info.height)
        self.channels = 1 if gray else 3
        vf = []
        if fps:
            vf.append(f"fps={fps:.6f}")
        if (self.width, self.height) != (info.width, info.height):
            vf.append(f"scale={self.width}:{self.height}:flags={scale_flags}")
        cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-an", "-sn"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
        self._cmd = cmd
        self._frame_bytes = self.width * self.height * self.channels
        self._prefetch = prefetch
        self._proc: Optional[subprocess.Popen] = None

    def __iter__(self) -> Iterator[np.ndarray]:
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self._frame_bytes * 2
        )
        q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=self._prefetch)
        stop = threading.Event()
        shape = (self.height, self.width) if self.channels == 1 else (self.height, self.width, 3)

        def pump():
            assert self._proc and self._proc.stdout
            try:
                while not stop.is_set():
                    buf = self._proc.stdout.read(self._frame_bytes)
                    if not buf or len(buf) < self._frame_bytes:
                        break
                    q.put(np.frombuffer(buf, np.uint8).reshape(shape))
            finally:
                q.put(None)

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            while True:
                frame = q.get()
                if frame is None:
                    break
                yield frame
        finally:
            stop.set()
            # drain so the pump thread can exit
            while th.is_alive():
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
                if self._proc.poll() is None:
                    self._proc.kill()
            self._proc.wait()
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._proc.stderr:
                self._proc.stderr.close()


# Audio codecs that can be stream-copied into an mp4 container as-is.
_MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}


class VideoWriter:
    """Encode BGR frames to H.264 mp4 (yuv420p, +faststart), muxing audio from
    ``audio_source`` when it has an audio track. Writes happen on a background
    thread so encoding overlaps with rendering."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: Optional[str] = None,
        audio_info: Optional[MediaInfo] = None,
        crf: int = 18,
        preset: str = "medium",
        queue_size: int = 6,
    ):
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p needs even dimensions")
        self.path = path
        cmd = [
            ffmpeg_exe(), "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-framerate", f"{fps:.6f}", "-i", "-",
        ]
        has_audio = bool(audio_source and audio_info and audio_info.has_audio)
        if has_audio:
            cmd += ["-i", audio_source]
        cmd += ["-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "1:a:0?"]
        cmd += [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
        ]
        if has_audio:
            if audio_info and audio_info.audio_codec in _MP4_SAFE_AUDIO:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            cmd += ["-shortest"]
        cmd += ["-movflags", "+faststart", path]
        self.cmd = cmd
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_size)
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        assert self._proc.stdin
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    break
                self._proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except BaseException as e:  # broken pipe etc.
            self._error = e
            # keep draining so producers never block forever
            while self._q.get() is not None:
                pass

    def write(self, frame: np.ndarray):
        if self._error:
            raise RuntimeError(f"ffmpeg encoder failed: {self._stderr()}") from self._error
        self._q.put(frame)

    def _stderr(self) -> str:
        try:
            return (self._proc.stderr.read() or b"").decode("utf-8", "replace")[-2000:]
        except Exception:
            return ""

    def close(self):
        self._q.put(None)
        self._thread.join()
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        rc = self._proc.wait()
        err = self._stderr()
        if self._proc.stderr:
            self._proc.stderr.close()
        if rc != 0 or self._error:
            raise RuntimeError(f"ffmpeg encoder exited with {rc}: {err}")

    def abort(self):
        try:
            self._proc.kill()
        finally:
            self._proc.wait()

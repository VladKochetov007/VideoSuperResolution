"""Video I/O. Frames go out as H.264/yuv420p in an mp4 (web-playable, unlike OpenCV's mp4v =
MPEG-4 Part 2 which browsers refuse) and the source audio is muxed back in.

OpenCV's VideoWriter cannot emit H.264 in most builds and never carries audio, so we pipe raw BGR
frames straight into ffmpeg and let libx264 + AAC do the encode in one pass — no double transcode.
"""

import subprocess
from pathlib import Path

import numpy as np


class FfmpegWriter:
    """Stream BGR uint8 frames to an H.264 mp4, optionally muxing audio from `audio_source`.

    Lazily starts ffmpeg on the first frame (needs the frame size). `audio_source` is mapped with a
    trailing '?' so a silent source simply produces a video-only file instead of failing.
    """

    def __init__(self, dst, fps: float, audio_source=None, crf: int = 18, preset: str = "veryfast"):
        self.dst = str(dst)
        self.fps = float(fps) if fps and fps > 0 else 24.0
        self.audio_source = str(audio_source) if audio_source is not None else None
        self.crf = crf
        self.preset = preset
        self.proc: subprocess.Popen | None = None

    def _start(self, w: int, h: int) -> None:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps}", "-i", "-"]
        if self.audio_source:
            cmd += ["-i", self.audio_source]
        cmd += ["-map", "0:v:0"]
        if self.audio_source:
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.dst]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.proc is None:
            self._start(w, h)
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None

    def __enter__(self) -> "FfmpegWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def mux_audio(video: Path, audio_source: Path, dst: Path) -> None:
    """Copy `video`'s stream and add `audio_source`'s audio (transcoded to AAC). No video re-encode.

    Used when the video is already H.264 (e.g. a fal result) and only needs its audio restored.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i",
         str(audio_source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
         "-shortest", "-movflags", "+faststart", str(dst)],
        check=True,
    )


def has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout
    return bool(out.strip())

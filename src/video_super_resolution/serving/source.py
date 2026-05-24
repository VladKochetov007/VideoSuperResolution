from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .unit import WorkUnit


class FrameSource(Protocol):
    def units(self, video_id: str, path: Path, model: str, out_h: int) -> Iterator[WorkUnit]: ...


class VideoFrameSource:
    # Per-frame models -> one unit per frame (streaming, O(1) RAM). Windowed models -> one unit
    # per overlapping window of `window` frames with `overlap` shared frames between neighbours.
    def __init__(self, window: int = 16, overlap: int = 2, max_frames: int = 0):
        self.window = window
        self.overlap = overlap
        self.max_frames = max_frames

    def units(self, video_id, path, model, out_h):
        cap = cv2.VideoCapture(str(path))
        if model == "seedvr2":
            yield from self._windows(cap, video_id, model, out_h)
        else:
            yield from self._frames(cap, video_id, model, out_h)
        cap.release()

    def _read_all(self, cap):
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            if self.max_frames and len(frames) >= self.max_frames:
                break
        return frames

    def _frames(self, cap, video_id, model, out_h):
        i = 0
        while True:
            ok, f = cap.read()
            if not ok or (self.max_frames and i >= self.max_frames):
                break
            yield WorkUnit(video_id=video_id, model=model, index=i, payload=f, out_h=out_h)
            i += 1

    def _windows(self, cap, video_id, model, out_h):
        frames = self._read_all(cap)
        stride = self.window - self.overlap
        start = 0
        while start < len(frames):
            stack = np.stack(frames[start:start + self.window])
            yield WorkUnit(video_id=video_id, model=model, index=start,
                           payload=stack, overlap=self.overlap, out_h=out_h)
            if start + self.window >= len(frames):
                break
            start += stride

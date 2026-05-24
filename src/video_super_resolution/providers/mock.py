from pathlib import Path

import cv2

from ..media import FfmpegWriter
from .base import UpscaleResult, probe


class MockUpscaler:
    """Zero-cost, offline stand-in: bicubic resize to the target height. No fal, no GPU, no weights.

    Satisfies the `Upscaler` protocol so the webui and pipeline exercise the real code path without
    spending credits. It is deliberately NOT in PROVIDERS (the billable fal registry). Output is
    H.264 with the source audio carried over.
    """

    name = "mock"

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float:
        return 0.0

    def run(self, input_path: Path, out_path: Path, out_h: int) -> UpscaleResult:
        in_w, in_h, fps, _ = probe(input_path)
        out_h += out_h % 2
        out_w = round(in_w * out_h / in_h)
        out_w += out_w % 2
        cap = cv2.VideoCapture(str(input_path))
        try:
            with FfmpegWriter(out_path, fps, audio_source=input_path) as writer:
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    writer.write(cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_CUBIC))
        finally:
            cap.release()
        return UpscaleResult(self.name, str(out_path), 0.0)

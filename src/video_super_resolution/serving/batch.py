from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .unit import WorkUnit


@runtime_checkable
class BatchUpscaler(Protocol):
    name: str

    # Returns upscaled payloads aligned 1:1 with `units` (element i = upscaled units[i].payload).
    # Aligned-order return keeps (video_id, index) routing unambiguous when a batch mixes videos.
    def batch_infer(self, units: list[WorkUnit]) -> list[np.ndarray]: ...


class BicubicBatchUpscaler:
    # CPU reference model (per-frame bicubic x2) for tests / the enqueue->drain->reassemble loop.
    name = "fake"

    def batch_infer(self, units: list[WorkUnit]) -> list[np.ndarray]:
        out = []
        for u in units:
            h, w = u.payload.shape[:2]
            out.append(cv2.resize(u.payload, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
        return out

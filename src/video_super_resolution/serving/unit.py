import msgspec
import numpy as np


class WorkUnit(msgspec.Struct, frozen=True):
    # model: units only batch with same model. payload: (H,W,3) frame (per-frame) or
    # (L,H,W,3) window stack (windowed). index: frame index, or window start frame.
    video_id: str
    model: str
    index: int
    payload: np.ndarray
    overlap: int = 0
    out_h: int = 1080

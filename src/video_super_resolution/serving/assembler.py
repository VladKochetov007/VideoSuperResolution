import numpy as np


class Assembler:
    # Per-video reorder buffer. Units finish in batch order, not video order, so parts are keyed by
    # index and ordered on emit(). Windowed models overlap-add adjacent windows with linear weights
    # (crossfade, not a hard switch); the normalization recovers full weight at the sequence edges.
    def __init__(self, video_id: str, total_units: int, overlap: int, windowed: bool):
        self.video_id = video_id
        self.total = total_units
        self.overlap = overlap
        self.windowed = windowed
        self.parts: dict[int, np.ndarray] = {}

    def put(self, index: int, out: np.ndarray) -> None:
        if index not in self.parts:
            self.parts[index] = out

    def done(self) -> bool:
        return len(self.parts) >= self.total

    def emit(self) -> list[np.ndarray]:
        keys = sorted(self.parts)
        if not self.windowed:
            return [self.parts[k] for k in keys]

        total_len = max(k + len(self.parts[k]) for k in keys)
        h, w = self.parts[keys[0]].shape[1:3]
        acc = np.zeros((total_len, h, w, 3), np.float32)
        wsum = np.zeros((total_len, 1, 1, 1), np.float32)
        o = self.overlap
        for k in keys:
            win = self.parts[k].astype(np.float32)
            length = len(win)
            wt = np.ones(length, np.float32)
            if o > 0 and length >= o:
                ramp = np.linspace(1.0 / (o + 1), 1.0, o, dtype=np.float32)
                wt[:o] = ramp
                wt[-o:] = ramp[::-1]
            acc[k:k + length] += win * wt[:, None, None, None]
            wsum[k:k + length] += wt[:, None, None, None]
        out = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
        return list(out)

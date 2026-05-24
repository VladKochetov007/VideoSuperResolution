import math


class CapacityModel:
    # batch_size(model) from a measured VRAM fit: peak_vram(batch) ~= a*batch + b, where `a` is
    # per-unit activation bytes and `b` is params+workspace (get (a,b) from a one-off batch sweep,
    # e.g. scripts/bench_throughput.py). headroom guards fragmentation + transient peaks.
    def __init__(self, fit: dict[str, tuple[float, float]], vram_bytes: float, headroom: float = 0.10,
                 max_batch: int = 0):
        self.fit = fit
        self.vram = vram_bytes
        self.headroom = headroom
        self.max_batch = max_batch  # user cap; 0 = VRAM-bound only

    def batch_size(self, model: str) -> int:
        a, b = self.fit[model]
        budget = self.vram * (1.0 - self.headroom) - b
        n = math.floor(budget / a) if a > 0 else 1
        n = max(1, int(n))
        return min(n, self.max_batch) if self.max_batch > 0 else n

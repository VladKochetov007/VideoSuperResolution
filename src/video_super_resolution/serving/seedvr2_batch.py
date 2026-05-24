"""SeedVR2 as a batched, windowed BatchUpscaler for the Scheduler. RUNS UNDER torch+CUDA.

Mirrors RealESRGANBatch's contract but for the windowed diffusion model: each WorkUnit carries an
(L,H,W,3) frame window (built by VideoFrameSource for model=="seedvr2"), and batch_infer upscales
each window. SeedVR2-3B/7B is a large-GPU model, so the heavy diffusion forward is supplied through
an injectable backend at deployment rather than vendored here; the VRAM preflight and weight
resolution are real and run anywhere.

On a small GPU the preflight raises in __init__ (before any work). On the big server, pass a
`backend` (the official SeedVR2 pipeline / a ComfyUI runner) and it drives the cached weights.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..local.weights import SEEDVR2_MIN_VRAM_GB, ensure_seedvr2_weights, seedvr2_is_cached
from .unit import WorkUnit

# backend(units, weights_path) -> upscaled (L,H',W',3) uint8 windows aligned 1:1 with units
SeedVR2Backend = Callable[[list[WorkUnit], Path], list[np.ndarray]]


class SeedVR2Batch:
    name = "seedvr2"

    def __init__(self, variant: str = "3B", min_vram_gb: float = 0.0, download: bool = False,
                 backend: SeedVR2Backend | None = None, cache_dir: Path | None = None):
        self.variant = variant
        self.min_vram_gb = min_vram_gb or SEEDVR2_MIN_VRAM_GB[variant]
        self.backend = backend
        self._preflight_vram()
        self.weights = self._resolve_weights(download, cache_dir)

    def _preflight_vram(self) -> None:
        import torch

        free = torch.cuda.mem_get_info()[0] / 1024 ** 3 if torch.cuda.is_available() else 0.0
        if free < self.min_vram_gb:
            raise RuntimeError(
                f"insufficient VRAM for SeedVR2-{self.variant}: need ~{self.min_vram_gb:.0f} GB, "
                f"have {free:.1f} GB free. This model targets the large GPU.")

    def _resolve_weights(self, download: bool, cache_dir: Path | None) -> Path | None:
        if seedvr2_is_cached(self.variant, cache_dir):
            return ensure_seedvr2_weights(self.variant, cache_dir)  # cached -> no network
        if download:
            return ensure_seedvr2_weights(self.variant, cache_dir)
        return None  # backend may locate its own weights; otherwise it raises at infer time

    def batch_infer(self, units: list[WorkUnit]) -> list[np.ndarray]:
        if self.backend is None:
            raise NotImplementedError(
                "SeedVR2 inference backend not configured. Plug the diffusion pipeline in via the "
                "`backend` argument (weights, VRAM preflight, windowing and batching are handled).")
        return self.backend(units, self.weights)

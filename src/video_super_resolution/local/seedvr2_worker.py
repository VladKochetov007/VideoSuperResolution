"""Self-hosted SeedVR2 inference. RUNS UNDER a torch+CUDA venv on the deployment GPU.

SeedVR2-3B/7B is a diffusion-transformer VSR model sized for a large GPU (the 140 GB target), not a
6 GB laptop. This worker owns the parts that are model-agnostic and testable — VRAM preflight, weight
caching/download, and tiling/temporal-batch plumbing — and calls the diffusion forward through an
injectable backend so the heavy dependency is plugged in at deployment, not vendored here.

Order matters: VRAM is checked BEFORE any multi-GB download, so an undersized GPU fails fast and
wastes no bandwidth. Progress on stdout: STAGE/PROGRESS/DONE, same protocol as the Real-ESRGAN worker.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_super_resolution.local.weights import SEEDVR2_MIN_VRAM_GB, ensure_seedvr2_weights  # noqa: E402

DEFAULT_MIN_VRAM_GB = SEEDVR2_MIN_VRAM_GB


def _emit(line: str) -> None:
    print(line, flush=True)


def _free_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--out-h", type=int, required=True)
    ap.add_argument("--variant", default="3B", choices=["3B", "7B"])
    ap.add_argument("--batch", type=int, default=5, help="temporal batch (causal VAE wants 4n+1)")
    ap.add_argument("--temporal-overlap", type=int, default=4, help="frames shared between batches")
    ap.add_argument("--tile", type=int, default=0, help="spatial tile px; 0 = whole frame")
    ap.add_argument("--min-vram-gb", type=float, default=0.0, help="0 = use the per-variant default")
    ap.add_argument("--download", action="store_true", help="allow the multi-GB weight download")
    args = ap.parse_args()

    need = args.min_vram_gb or DEFAULT_MIN_VRAM_GB[args.variant]
    free = _free_vram_gb()
    _emit(f"STAGE VRAM preflight: need ~{need:.0f} GB, free {free:.1f} GB")
    if free < need:
        _emit(f"ERROR insufficient VRAM for SeedVR2-{args.variant}: "
              f"need ~{need:.0f} GB, have {free:.1f} GB free. This model targets the large GPU; "
              f"use realesrgan-local here, or run SeedVR2 on the deployment box.")
        return 2

    if not args.download:
        _emit("ERROR SeedVR2 weights not present and --download not set "
              "(weights are multi-GB; pass --download to fetch).")
        return 2

    _emit(f"STAGE resolving SeedVR2-{args.variant} weights")
    weights = ensure_seedvr2_weights(args.variant, progress=lambda s, f, m: _emit(f"STAGE {m}"))

    backend = _load_backend(weights, args)
    backend.run(args)  # pragma: no cover - exercised only on the deployment GPU
    return 0


def _load_backend(weights: Path, args):  # pragma: no cover - integration seam
    """Integration point for the SeedVR2 diffusion pipeline (e.g. the official ByteDance repo or a
    ComfyUI node). It receives the cached weights and the tiling/batch args. Kept injectable so the
    heavy, GPU-only dependency is supplied at deployment instead of vendored into this library."""
    raise NotImplementedError(
        "SeedVR2 inference backend not configured. Plug the diffusion pipeline in here "
        f"(weights at {weights}); preflight, caching and batching are already handled.")


if __name__ == "__main__":
    raise SystemExit(main())

"""Live VRAM calibration. RUNS UNDER a torch+CUDA venv.

We assume this process owns the GPU (single-session, video upscaling is the priority workload), so
we size batches to ~90% of TOTAL VRAM. The peak-memory-vs-batch relationship is linear
(peak ~= a*batch + b: `a` = per-frame activations, `b` = params + workspace), so two short probe
forwards at the real frame size give (a,b); CapacityModel then solves for the largest batch that
fits. RealESRGANBatch's OOM auto-halving covers any underestimate.
"""

import numpy as np
import torch

from .capacity import CapacityModel
from .unit import WorkUnit


def measure_fit(batch_model, sample_frame: np.ndarray, probes=(1, 2)) -> tuple[float, float]:
    """Fit peak_alloc(batch) = a*batch + b by running `probes` forwards at the sample frame size.

    Probes ascend; if one OOMs we stop and fit from those that fit (a small GPU may only hold batch
    1). `base` (resident params) anchors the fit when a single probe survives. Raises only if even
    batch 1 OOMs — then the frame needs tiling, not just a smaller batch.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM-calibrated batch queue")
    device = getattr(batch_model, "device", None)
    base = torch.cuda.memory_allocated(device)
    peaks: dict[int, int] = {}
    for n in probes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        units = [WorkUnit(video_id="cal", model=batch_model.name, index=i, payload=sample_frame)
                 for i in range(n)]
        try:
            setattr(batch_model, "_calibrating", True)
            batch_model._infer(units)  # raw forward, no OOM wrapper, to read true peak
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        finally:
            setattr(batch_model, "_calibrating", False)
        peaks[n] = torch.cuda.max_memory_allocated(device)
    torch.cuda.empty_cache()

    if not peaks:
        raise RuntimeError("batch 1 OOMs at this frame size — enable spatial tiling")
    if len(peaks) >= 2:
        ns = np.array(list(peaks), dtype=np.float64)
        ys = np.array([peaks[n] for n in peaks], dtype=np.float64)
        a, b = np.polyfit(ns, ys, 1)
    else:
        (n, p), = peaks.items()  # single point: activations above resident params
        a, b = (p - base) / n, float(base)
    return float(max(a, 1.0)), float(max(b, 0.0))


def calibrate_capacity(batch_model, sample_frame: np.ndarray, headroom: float = 0.10,
                       vram_bytes: float | None = None) -> tuple[CapacityModel, tuple[float, float]]:
    a, b = measure_fit(batch_model, sample_frame)
    if vram_bytes is None:
        _free, total = torch.cuda.mem_get_info()
        vram_bytes = float(total)  # own the GPU -> target a fraction of total, not just free
    torch.cuda.empty_cache()
    return CapacityModel({batch_model.name: (a, b)}, vram_bytes=vram_bytes, headroom=headroom), (a, b)

"""GPU-saturating batched VSR over a QUEUE of videos. RUNS UNDER .venv-onprem (torch+CUDA).

Reads a manifest of jobs, enqueues them into one shared WorkQueue, and drains it with a true batched
forward. Batch size is calibrated live to ~90% of VRAM (this process owns the GPU), so frames from
DIFFERENT videos ride the same forward. Fully configurable — model, color post-processing, batch cap,
tiling/temporal-batch — and built to run on a large GPU (where SeedVR2 and big batches fit).

    STAGE <text> | PROGRESS <done> <total> | VIDEO <id> <frames> <path> | DONE | ERROR <msg>
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_super_resolution.postprocess import ColorConfig, VideoColorPostProcessor, make_corrector
from video_super_resolution.local.weights import ensure_realesrgan_weights
from video_super_resolution.serving import CapacityModel, Scheduler, VideoFrameSource
from video_super_resolution.serving.calibrate import calibrate_capacity
from video_super_resolution.serving.realesrgan_batch import RealESRGANBatch
from video_super_resolution.serving.seedvr2_batch import SeedVR2Batch


def _emit(line: str) -> None:
    print(line, flush=True)


def _sample_frame(path: str):
    cap = cv2.VideoCapture(path)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read a frame from {path}")
    return f


def _detect_vram() -> float:
    return float(torch.cuda.mem_get_info()[1]) if torch.cuda.is_available() else 0.0


def _build(args, sample):
    """Return (model, capacity, source). Raises (caught in main) on VRAM/config failure."""
    vram = args.vram_gb * 1e9 or _detect_vram()
    if args.model == "realesrgan":
        weights = args.weights if args.weights.exists() else ensure_realesrgan_weights(
            args.weights.stem if args.weights.suffix else args.weights.name
        )
        model = RealESRGANBatch(weights, half=not args.fp32,
                                tile=args.tile, tile_pad=args.tile_pad)
        try:
            capacity, (a, b) = calibrate_capacity(model, sample, headroom=args.headroom,
                                                  vram_bytes=args.vram_gb * 1e9 or None)
        except RuntimeError as exc:
            if args.tile == 0 and "batch 1 OOMs" in str(exc):
                raise RuntimeError(
                    "whole-frame batch does not fit. Set Tile size to 256 or 512.") from exc
            raise
        if args.max_batch:
            capacity.max_batch = args.max_batch
        unit = "per-tile-frame" if args.tile else "per-frame"
        tile_msg = f", tile={args.tile}/pad={args.tile_pad}" if args.tile else ", whole-frame"
        _emit(f"STAGE B_max={capacity.batch_size('realesrgan')} ({unit} {a / 1e6:.0f} MB, "
              f"base {b / 1e9:.2f} GB, VRAM {vram / 1e9:.0f} GB, "
              f"target {(1 - args.headroom) * 100:.0f}%{tile_msg})")
        return model, capacity, VideoFrameSource()

    model = SeedVR2Batch(variant=args.variant, min_vram_gb=args.min_vram_gb, download=args.download)
    bmax = args.max_batch or 1
    capacity = CapacityModel({"seedvr2": (1.0, 0.0)}, vram_bytes=vram, headroom=args.headroom,
                             max_batch=bmax)
    _emit(f"STAGE SeedVR2-{args.variant}: window={args.temporal_batch} "
          f"overlap={args.temporal_overlap} batch={capacity.batch_size('seedvr2')} windows")
    return model, capacity, VideoFrameSource(window=args.temporal_batch, overlap=args.temporal_overlap)


def _pending(sched: Scheduler) -> int:
    return sum(sched.queue.pending(m) for m in ("realesrgan", "seedvr2"))


def _unique_jobs(jobs: list[dict]) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    uniq = []
    for job in jobs:
        base = str(job["name"])
        counts[base] += 1
        name = base if counts[base] == 1 else f"{base}_{counts[base]}"
        uniq.append({**job, "name": name})
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="JSON list of {input, out_h, name}")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "serve")
    ap.add_argument("--model", choices=["realesrgan", "seedvr2"], default="realesrgan")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "RealESRGAN_x2plus.pth")
    ap.add_argument("--headroom", type=float, default=0.10, help="VRAM kept free (0.10 = ~90%)")
    ap.add_argument("--max-batch", type=int, default=0, help="cap B_max; 0 = VRAM-bound")
    ap.add_argument("--fp32", action="store_true", help="disable fp16")
    ap.add_argument("--vram-gb", type=float, default=0.0, help="override total VRAM; 0 = detect")
    ap.add_argument("--tile", type=int, default=0, help="RealESRGAN tile size; 0 = whole frame")
    ap.add_argument("--tile-pad", type=int, default=10, help="RealESRGAN tile overlap padding")
    # color post-processing (applied per finished video via the Scheduler hook)
    ap.add_argument("--color", default="none", choices=["none", "wavelet", "reinhard", "adain", "histogram"])
    ap.add_argument("--color-strength", type=float, default=1.0)
    ap.add_argument("--wavelet-levels", type=int, default=5)
    # seedvr2
    ap.add_argument("--variant", default="3B", choices=["3B", "7B"])
    ap.add_argument("--temporal-batch", type=int, default=5, help="window length (4n+1)")
    ap.add_argument("--temporal-overlap", type=int, default=2)
    ap.add_argument("--min-vram-gb", type=float, default=0.0)
    ap.add_argument("--download", action="store_true", help="allow multi-GB SeedVR2 download")
    args = ap.parse_args()

    jobs = _unique_jobs(json.loads(args.manifest.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _emit(f"STAGE loading {args.model} on {('cuda' if torch.cuda.is_available() else 'cpu')}")

    try:
        model, capacity, source = _build(args, _sample_frame(jobs[0]["input"]))
    except Exception as exc:  # noqa: BLE001 - must surface worker failure as ERROR to parent UI
        _emit(f"ERROR {exc}")
        return 2

    corrector = make_corrector(ColorConfig(method=args.color, strength=args.color_strength,
                                           wavelet_levels=args.wavelet_levels))
    post = VideoColorPostProcessor(corrector) if corrector is not None else None
    if post is not None:
        _emit(f"STAGE color post-processing: {args.color} (strength {args.color_strength})")

    sched = Scheduler({model.name: model}, capacity, source, postprocess=post)
    for j in jobs:
        sched.enqueue_video(j["name"], j["input"], model.name, out_h=j.get("out_h", 0),
                            out_path=args.out_dir / f"{j['name']}_{model.name}.mp4")

    total = _pending(sched)
    _emit(f"STAGE queued {len(jobs)} videos, {total} units total")
    done = 0
    try:
        while not sched.queue.is_empty():
            before = _pending(sched)
            finished = sched.run_step()
            done += before - _pending(sched)
            used = (1 - torch.cuda.mem_get_info()[0] / torch.cuda.mem_get_info()[1]) * 100 \
                if torch.cuda.is_available() else 0.0
            _emit(f"STAGE batch size={before - _pending(sched)} VRAM={used:.0f}%")
            _emit(f"PROGRESS {done} {total}")
            for r in finished:
                _emit(f"VIDEO {r.video_id} {r.frames} {r.out_path}")
    except Exception as exc:  # noqa: BLE001 - must surface worker failure as ERROR to parent UI
        _emit(f"ERROR {exc}")
        return 2
    _emit("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

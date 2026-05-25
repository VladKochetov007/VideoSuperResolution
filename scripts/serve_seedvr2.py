"""Self-hosted SeedVR2 over a QUEUE of videos. RUNS UNDER .venv-onprem (torch+CUDA + node deps).

Unlike Real-ESRGAN (per-frame/tile, cross-video batched by the Scheduler), SeedVR2 is a windowed
diffusion model whose own pipeline does the temporal batching and overlap blending. So this worker
loads the model ONCE and pushes each whole clip through it, then applies our color post-processing
and restores the source audio. Heavy inference is delegated to the ComfyUI-SeedVR2 core via
`SeedVR2ComfyRunner`; GGUF + block-swap let the 3B model fit a small GPU.

    STAGE <text> | PROGRESS <done> <total> | VIDEO <id> <frames> <path> | DONE | ERROR <msg>
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_super_resolution.postprocess import ColorConfig, VideoColorPostProcessor, make_corrector
from video_super_resolution.serving.seedvr2_comfy import SeedVR2ComfyRunner


def _emit(line: str) -> None:
    print(line, flush=True)


# The SeedVR2 core prints its own phase/batch logs straight to stdout. Those lines don't carry the
# STAGE/PROGRESS prefixes the parent parses, so a long single-clip run looks frozen. This filter
# re-emits the meaningful phase/batch lines as STAGE (and drops the per-block swap spam) so the UI
# shows live progress.
_PROGRESS_MARKERS = ("Phase ", "Upscaling batch", "Decoding batch", "Encoding batch",
                     "Materializing", "Downloading", "Output saved")


class _NodeProgressFilter:
    """stdout proxy: re-emits node phase/batch lines as STAGE, drops the rest. Delegates everything
    else (fileno, isatty, encoding, ...) to the real stream so tqdm and the node behave normally."""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            text = line.strip()
            if any(m in text for m in _PROGRESS_MARKERS):
                clean = text.split("] ", 1)[-1]  # drop the node's "[HH:MM:SS]" timestamp
                self._real.write(f"STAGE {clean}\n")
                self._real.flush()
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _read_frames(path: str):
    import numpy as np

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames), fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="JSON list of {input, out_h, name}")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "serve_seedvr2")
    ap.add_argument("--dit-model", default="seedvr2_ema_3b-Q4_K_M.gguf")
    ap.add_argument("--batch-size", type=int, default=1, help="temporal window (4n+1); raise on big GPU")
    ap.add_argument("--blocks-to-swap", type=int, default=32, help="0 = no block-swap (large GPU)")
    ap.add_argument("--no-swap-io", action="store_true", help="keep I/O components on GPU")
    ap.add_argument("--offload", default="cpu", choices=["cpu", "none"], help="DiT/VAE offload device")
    ap.add_argument("--attention-mode", default="sdpa", help="sdpa | flash_attn | sage (host-dependent)")
    ap.add_argument("--vae-tile", type=int, default=256, help="VAE tile px; raise/0 on big GPU")
    ap.add_argument("--vae-tile-overlap", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo-dir", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--color", default="none", choices=["none", "wavelet", "reinhard", "adain", "histogram"])
    ap.add_argument("--color-strength", type=float, default=1.0)
    ap.add_argument("--wavelet-levels", type=int, default=5)
    args = ap.parse_args()

    jobs = json.loads(args.manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _emit(f"STAGE loading seedvr2 ({args.dit_model}, blocks_to_swap={args.blocks_to_swap}, "
          f"attention={args.attention_mode})")

    try:
        runner = SeedVR2ComfyRunner(
            repo_dir=args.repo_dir, model_dir=args.model_dir, dit_model=args.dit_model,
            batch_size=args.batch_size, blocks_to_swap=args.blocks_to_swap,
            swap_io_components=not args.no_swap_io, offload=args.offload,
            attention_mode=args.attention_mode, vae_tile=args.vae_tile,
            vae_tile_overlap=args.vae_tile_overlap, color_correction="none", seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - surface setup failure to the parent UI
        _emit(f"ERROR {exc}")
        return 2

    corrector = make_corrector(ColorConfig(method=args.color, strength=args.color_strength,
                                           wavelet_levels=args.wavelet_levels))
    post = VideoColorPostProcessor(corrector) if corrector is not None else None

    from video_super_resolution.media import FfmpegWriter

    total = len(jobs)
    _emit(f"STAGE queued {total} videos")
    for i, job in enumerate(jobs, 1):
        try:
            src = job["input"]
            out_h = int(job.get("out_h") or 1080)
            _emit(f"STAGE upscaling {job['name']} -> {out_h}px")
            frames, fps = _read_frames(src)
            import contextlib
            with contextlib.redirect_stdout(_NodeProgressFilter(sys.stdout)):
                up = runner.upscale(frames, out_h)
            out_list = list(up)
            if post is not None:
                _emit(f"STAGE color post-processing {job['name']}: {args.color}")
                out_list = post(out_list, Path(src))
            dst = args.out_dir / f"{job['name']}_seedvr2.mp4"
            with FfmpegWriter(dst, fps, audio_source=src) as writer:
                for f in out_list:
                    writer.write(f)
            _emit(f"VIDEO {job['name']} {len(out_list)} {dst}")
            _emit(f"PROGRESS {i} {total}")
        except Exception as exc:  # noqa: BLE001 - surface per-clip failure, keep protocol
            _emit(f"ERROR {exc}")
            return 2
    _emit("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

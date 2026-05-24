"""Self-hosted Real-ESRGAN inference. RUNS UNDER .venv-onprem (needs torch+CUDA, basicsr,
realesrgan). Invoked as a subprocess by video_super_resolution.local.provider so the Streamlit venv stays clean.

Emits machine-readable progress on stdout for the parent to parse:
    STAGE <text>
    PROGRESS <done> <total>
    DONE <out_w> <out_h> <frames> <seconds>
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make video_super_resolution importable

import cv2  # noqa: E402

from video_super_resolution.local.weights import ensure_realesrgan_weights  # noqa: E402
from video_super_resolution.media import FfmpegWriter  # noqa: E402


def _emit(line: str) -> None:
    print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--out-h", type=int, required=True)
    ap.add_argument("--model", default="RealESRGAN_x2plus")
    ap.add_argument("--native-scale", type=int, default=2, help="model's native scale (x2plus=2)")
    ap.add_argument("--tile", type=int, default=256, help="0 disables tiling")
    ap.add_argument("--tile-pad", type=int, default=10)
    ap.add_argument("--half", action="store_true", help="fp16 (saves VRAM)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _emit(f"STAGE resolving weights ({args.model})")
    weights = ensure_realesrgan_weights(args.model, progress=lambda s, f, m: _emit(f"STAGE {m}"))

    _emit(f"STAGE loading model on {dev}")
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,
                  scale=args.native_scale)
    up = RealESRGANer(scale=args.native_scale, model_path=str(weights), model=net,
                      tile=args.tile, tile_pad=args.tile_pad, half=args.half, device=dev)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.max_frames:
        total = min(total, args.max_frames) if total else args.max_frames
    outscale = args.out_h / in_h

    _emit(f"STAGE upscaling {total or '?'} frames x{outscale:.3f} (tile={args.tile})")
    t0 = time.time()
    ow = oh = n = 0
    with FfmpegWriter(args.output, fps, audio_source=args.input) as writer:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            out, _ = up.enhance(frame, outscale=outscale)
            oh, ow = out.shape[:2]
            writer.write(out)
            n += 1
            if total:
                _emit(f"PROGRESS {n} {total}")
    cap.release()
    _emit(f"DONE {ow} {oh} {n} {time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

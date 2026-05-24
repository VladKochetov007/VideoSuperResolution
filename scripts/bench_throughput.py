import argparse
import time
from pathlib import Path

import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "RealESRGAN_x2plus.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--in-w", type=int, default=960)
    ap.add_argument("--in-h", type=int, default=540)
    ap.add_argument("--tile", type=int, default=0, help="0 = no tiling (use on a big GPU)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    half = dev == "cuda"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    up = RealESRGANer(scale=2, model_path=str(WEIGHTS), model=model, tile=args.tile,
                      tile_pad=10, half=half, device=dev)

    img = (np.random.rand(args.in_h, args.in_w, 3) * 255).astype(np.uint8)
    for _ in range(3):  # warmup (CUDA init, allocator)
        up.enhance(img, outscale=2)
    if dev == "cuda":
        torch.cuda.synchronize()

    t = time.time()
    for _ in range(args.frames):
        up.enhance(img, outscale=2)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t
    fps = args.frames / dt

    print(f"Real-ESRGAN x2 | {dev} half={half} | {args.in_w}x{args.in_h} -> "
          f"{args.in_w * 2}x{args.in_h * 2} | tile={args.tile}")
    print(f"  {args.frames} frames in {dt:.2f}s = {fps:.2f} fps (pure inference)")
    print(f"  => {fps / 24:.2f}x realtime(24fps) | {24 / fps:.2f} GPU-min per video-min")


if __name__ == "__main__":
    main()

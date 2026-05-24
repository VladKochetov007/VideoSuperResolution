import argparse
import time
from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "eval" / "face" / "2.00x" / "input.mp4"
OUT = ROOT / "data" / "eval" / "outputs" / "face_2.00x_onprem_realesrgan.mp4"
WEIGHTS = ROOT / "weights" / "RealESRGAN_x2plus.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    up = RealESRGANer(scale=2, model_path=str(WEIGHTS), model=model, tile=256, tile_pad=10,
                      half=False, device=dev)

    cap = cv2.VideoCapture(str(IN))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if args.max_frames and len(frames) >= args.max_frames:
            break
    cap.release()
    print(f"device {dev} | {len(frames)} frames @ {fps:.0f}fps | NO network at inference")

    t0 = time.time()
    outs = [up.enhance(f, outscale=2)[0] for f in frames]
    dt = time.time() - t0
    h, w = outs[0].shape[:2]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for o in outs:
        vw.write(o)
    vw.release()
    print(f"self-hosted Real-ESRGAN x2 -> {w}x{h}, {len(frames)} frames in {dt:.1f}s "
          f"({len(frames) / dt:.2f} fps) on {dev} -> {OUT.name}")


if __name__ == "__main__":
    main()

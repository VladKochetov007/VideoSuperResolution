import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLIPS = [
    {"name": "text", "axis": "text", "master": "sandbox/eval/_masters/text_master.mp4",
     "jobs": [(720, 1080), (720, 1440)]},
    {"name": "face", "axis": "faces", "master": "sandbox/eval/_masters/face_master.mp4",
     "jobs": [(534, 800), (400, 800)]},
    {"name": "texture", "axis": "fine-texture", "master": "sandbox/eval/_masters/texture_master.mp4",
     "jobs": [(534, 800), (400, 800)]},
]


def probe_wh(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def even(x: float) -> int:
    return int(round(x)) - (int(round(x)) % 2)


def scale_to_h(src: Path, dst: Path, h: int, mw: int, mh: int, crf: int) -> tuple[int, int]:
    w, hh = even(h * mw / mh), even(h)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale={w}:{hh}:flags=lanczos", "-an", "-c:v", "libx264", "-crf", str(crf), str(dst)],
        check=True,
    )
    return w, hh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-crf", type=int, default=26, help="realistic codec degradation on input")
    ap.add_argument("--gt-crf", type=int, default=12, help="near-lossless GT")
    args = ap.parse_args()

    jobs = []
    for clip in CLIPS:
        master = ROOT / clip["master"]
        mw, mh = probe_wh(master)
        for in_h, gt_h in clip["jobs"]:
            label = f"{gt_h / in_h:.2f}x"
            d = ROOT / "data" / "eval" / clip["name"] / label
            d.mkdir(parents=True, exist_ok=True)
            gw, gh = scale_to_h(master, d / "gt.mp4", gt_h, mw, mh, args.gt_crf)
            iw, ih = scale_to_h(master, d / "input.mp4", in_h, mw, mh, args.input_crf)
            jobs.append({"clip": clip["name"], "axis": clip["axis"], "scale": label,
                         "input": str((d / "input.mp4").relative_to(ROOT)),
                         "gt": str((d / "gt.mp4").relative_to(ROOT)),
                         "in_w": iw, "in_h": ih, "gt_w": gw, "gt_h": gh})
            print(f"{clip['name']:8s} {label}: input {iw}x{ih} -> gt {gw}x{gh}")

    (ROOT / "data" / "eval" / "jobs.json").write_text(json.dumps(jobs, indent=2))
    print(f"\nwrote {len(jobs)} jobs to data/eval/jobs.json")


if __name__ == "__main__":
    main()

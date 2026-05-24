import argparse
import json
import shlex
import subprocess
import urllib.request
from pathlib import Path

# name, url, axis, track, start_s, dur_s. Demo URLs (Google sample bucket) are reliably
# downloadable; swap for 4K masters for the real run.
MANIFEST = [
    {"name": "tears_of_steel", "axis": "face+vfx", "track": "A", "start": 120, "dur": 8,
     "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"},
    {"name": "big_buck_bunny", "axis": "texture+animation", "track": "A", "start": 30, "dur": 8,
     "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"},
    {"name": "sintel", "axis": "fast-motion", "track": "A", "start": 40, "dur": 8,
     "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4"},
]


def run(cmd: list[str]) -> None:
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  cached {dest.name}")
        return True
    try:
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001 - log and skip, never abort the batch
        print(f"  !! failed {url}: {exc}")
        return False


def prep_clip(entry: dict, data_dir: Path, master_h: int, scale: float, crf: int) -> None:
    name = entry["name"]
    raw = data_dir / "raw" / f"{name}.mp4"
    if not download(entry["url"], raw):
        return
    src_w, src_h, fps = probe(raw)
    trim = ["-ss", str(entry["start"]), "-t", str(entry["dur"])]

    if entry["track"] == "B":
        out = data_dir / "input" / f"{name}_B_input.mp4"
        run(["ffmpeg", "-y", *trim, "-i", str(raw), "-c:v", "libx264", "-an", str(out)])
        print(f"  -> Track B input {out.name} (no ground truth)\n")
        return

    h = min(master_h, src_h)
    w = h * 16 // 9
    lr_h, lr_w = int(h / scale), int(h / scale) * 16 // 9
    master = data_dir / "master" / f"{name}_HR_{h}p.mp4"
    clean = data_dir / "input" / f"{name}_A_input_clean_{lr_h}p.mp4"
    realistic = data_dir / "input" / f"{name}_A_input_realistic_{lr_h}p.mp4"

    run(["ffmpeg", "-y", *trim, "-i", str(raw),
         "-vf", f"scale={w}:{h}:flags=lanczos", "-c:v", "libx264", "-crf", "12", "-an", str(master)])
    run(["ffmpeg", "-y", "-i", str(master),
         "-vf", f"scale={lr_w}:{lr_h}:flags=bicubic", "-c:v", "libx264", "-crf", "12", "-an",
         str(clean)])
    run(["ffmpeg", "-y", "-i", str(master),
         "-vf", f"scale={lr_w}:{lr_h}:flags=bicubic", "-c:v", "libx264", "-crf", str(crf), "-an",
         str(realistic)])
    print(f"  -> HR master {master.name}  |  LR inputs: clean + realistic (crf {crf})\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--manifest", type=Path, help="JSON override for the clip manifest")
    ap.add_argument("--master-height", type=int, default=1080, help="HR ground-truth height")
    ap.add_argument("--scale", type=float, default=1.5, help="HR/LR ratio (1.5=720->1080, 2=720->1440)")
    ap.add_argument("--crf", type=int, default=26, help="x264 CRF for the realistic LR input")
    args = ap.parse_args()

    for sub in ("raw", "master", "input"):
        (args.data_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()) if args.manifest else MANIFEST

    for entry in manifest:
        print(f"[{entry['name']}] axis={entry['axis']} track={entry['track']}")
        prep_clip(entry, args.data_dir, args.master_height, args.scale, args.crf)


if __name__ == "__main__":
    main()

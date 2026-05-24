import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "eval" / "results.json"
JOBS = ROOT / "data" / "eval" / "jobs.json"
OUT = ROOT / "outputs"
PANEL_H = 480
ORDER = {"realesrgan": 1, "topaz": 2, "seedvr2": 3}
FONT = "/tmp/video_sr_f.ttf"


def ensure_font() -> None:
    if not Path(FONT).exists():
        src = subprocess.run(["fc-match", "-f", "%{file}", "sans"],
                             capture_output=True, text=True).stdout.strip()
        shutil.copy(src, FONT)


def label(text: str) -> str:
    return (f"drawtext=fontfile={FONT}:text='{text}':x=12:y=12:fontsize=24:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8")


def main() -> None:
    ensure_font()
    OUT.mkdir(exist_ok=True)
    results = json.loads(RESULTS.read_text())
    jobs = {(j["clip"], j["scale"]): j for j in json.loads(JOBS.read_text())}

    groups: dict = {}
    for r in results:
        groups.setdefault((r["clip"], r["scale"]), []).append(r)

    for (clip, scale), rs in groups.items():
        j = jobs[(clip, scale)]
        panels = [("input 720p", ROOT / j["input"])]
        panels += [(r["model"], ROOT / r["output"]) for r in sorted(rs, key=lambda x: ORDER.get(x["model"], 9))]
        panels.append(("GT", ROOT / j["gt"]))

        inputs: list[str] = []
        for _, p in panels:
            inputs += ["-i", str(p)]
        fc = [f"[{i}:v]scale=-2:{PANEL_H},setsar=1,{label(name)}[v{i}]"
              for i, (name, _) in enumerate(panels)]
        fc.append("".join(f"[v{i}]" for i in range(len(panels))) + f"hstack=inputs={len(panels)}[out]")

        dst = OUT / f"sidebyside_{clip}_{scale}.mp4"
        print(f"build {dst.name} ({len(panels)} panels)")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
             "-filter_complex", ";".join(fc), "-map", "[out]",
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)],
            check=True,
        )
    print("done")


if __name__ == "__main__":
    main()

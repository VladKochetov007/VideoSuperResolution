"""Measure color drift of each upscaler output vs its source, and how much each correction method
removes. Reads data/eval/results.json (no fal spend, no GPU). Writes outputs/color_delta_e.{csv,md}
and one before/after crop for the worst-drift clip.

    .venv/bin/python scripts/color_eval.py
"""

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import cv2
import numpy as np

from video_super_resolution.postprocess import ColorConfig, FrameColorCorrector, color_drift

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "eval" / "results.json"
JOBS = ROOT / "data" / "eval" / "jobs.json"
OUT = ROOT / "outputs"
METHODS = ["wavelet", "reinhard", "adain", "histogram"]


def read_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    fs: list[np.ndarray] = []
    while len(fs) < n:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    cap.release()
    return fs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    results = json.loads(RESULTS.read_text())
    jobs = {(j["clip"], j["scale"]): j for j in json.loads(JOBS.read_text())}
    correctors = {m: FrameColorCorrector(ColorConfig(method=m)) for m in METHODS}

    rows: list[dict] = []
    worst = None  # (drift, result, job)
    for r in results:
        job = jobs.get((r["clip"], r["scale"]))
        if job is None:
            continue
        out = read_frames(ROOT / r["output"], args.frames)
        src = read_frames(ROOT / job["input"], args.frames)
        if len(out) < 1 or len(src) < 1:
            print(f"  !! skip {r['clip']} {r['scale']} {r['model']}: unreadable")
            continue
        pairs = list(zip(out, src))
        before = float(np.mean([color_drift(o, s) for o, s in pairs]))
        row = {"clip": r["clip"], "scale": r["scale"], "model": r["model"], "drift_raw": before}
        for m, corr in correctors.items():
            after = float(np.mean([color_drift(corr(o, s), s) for o, s in pairs]))
            row[m] = after
        rows.append(row)
        print(f"{r['clip']:8s} {r['scale']} {r['model']:11s} ΔE raw {before:5.2f} -> "
              + " ".join(f"{m} {row[m]:4.2f}" for m in METHODS))
        if worst is None or before > worst[0]:
            worst = (before, r, job)

    cols = ["clip", "scale", "model", "drift_raw"] + METHODS
    with (OUT / "color_delta_e.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 3) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    _write_md(rows)
    if worst is not None:
        _crop(worst[1], worst[2], args.frames)
    print(f"\nwrote {OUT/'color_delta_e.csv'}, color_delta_e.md")


def _write_md(rows: list[dict]) -> None:
    lines = ["# Color drift (ΔE2000 vs source, low-frequency) — raw output and after correction", "",
             "Lower = closer to the source's color. `drift_raw` is the uncorrected upscaler output.",
             "Measured on the blurred image pair so added detail is not counted as color drift.", "",
             "| clip | scale | model | raw | wavelet | reinhard | adain | histogram |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: -x["drift_raw"]):
        best = min(r[m] for m in METHODS)
        cells = [r["clip"], r["scale"], r["model"], f"{r['drift_raw']:.2f}"]
        for m in METHODS:
            cells.append(f"**{r[m]:.2f}**" if r[m] == best else f"{r[m]:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    (OUT / "color_delta_e.md").write_text("\n".join(lines) + "\n")


def _crop(r: dict, job: dict, frames: int) -> None:
    """Before/after panel for the worst-drift clip: raw | wavelet | source. Mid-frame center crop."""
    out = read_frames(ROOT / r["output"], frames)
    src = read_frames(ROOT / job["input"], frames)
    mid = len(out) // 2
    o, s = out[mid], src[mid]
    s = cv2.resize(s, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_CUBIC)
    fixed = FrameColorCorrector(ColorConfig(method="wavelet"))(o, s)
    h, w = o.shape[:2]
    c = min(360, h, w)
    y, x = h // 2 - c // 2, w // 2 - c // 2
    crop = lambda im: im[y:y + c, x:x + c]
    dst = OUT / f"color_fix_{r['clip']}_{r['scale']}_{r['model']}.png"
    cv2.imwrite(str(dst), np.hstack([crop(o), crop(fixed), crop(s)]))
    print(f"  wrote {dst.name} (raw | wavelet | source)")


if __name__ == "__main__":
    main()

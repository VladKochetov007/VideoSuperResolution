import argparse
import csv
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import cv2
import numpy as np
import pyiqa
import torch

from video_super_resolution.metrics.temporal import warping_error

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "eval" / "results.json"
JOBS = ROOT / "data" / "eval" / "jobs.json"
OUT = ROOT / "outputs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def to_t(frames: list[np.ndarray]) -> torch.Tensor:
    a = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(0, 3, 1, 2)


def resize_to(frames: list[np.ndarray], w: int, h: int) -> list[np.ndarray]:
    return [cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
            if (f.shape[1], f.shape[0]) != (w, h) else f for f in frames]


class Metrics:
    def __init__(self) -> None:
        self.psnr = pyiqa.create_metric("psnr", device=DEV)
        self.ssim = pyiqa.create_metric("ssim", device=DEV)
        self.lpips = pyiqa.create_metric("lpips", device=DEV)
        self.musiq = pyiqa.create_metric("musiq", device=DEV)

    def _fr(self, metric, test: torch.Tensor, ref: torch.Tensor) -> float:
        return float(np.mean([float(metric(test[i:i + 1].to(DEV), ref[i:i + 1].to(DEV)))
                              for i in range(test.shape[0])]))

    def _nr(self, metric, test: torch.Tensor) -> float:
        return float(np.mean([float(metric(test[i:i + 1].to(DEV))) for i in range(test.shape[0])]))

    def score(self, out_frames: list[np.ndarray], gt_frames: list[np.ndarray]) -> dict:
        gh, gw = gt_frames[0].shape[:2]
        of = resize_to(out_frames, gw, gh)
        ot, gt = to_t(of), to_t(gt_frames)
        return {"PSNR": self._fr(self.psnr, ot, gt), "SSIM": self._fr(self.ssim, ot, gt),
                "LPIPS": self._fr(self.lpips, ot, gt), "MUSIQ": self._nr(self.musiq, ot),
                "E_warp": warping_error(of)["E_warp_short"]}


def montage(input_p: Path, out_p: Path, gt_p: Path, dst: Path, gw: int, gh: int) -> None:
    """Mid-frame 320-crop: bicubic(input) | model | GT."""
    def mid(p: Path, w: int, h: int) -> np.ndarray:
        fs = read_frames(p, 999)
        f = fs[len(fs) // 2]
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    bic, mod, gt = mid(input_p, gw, gh), mid(out_p, gw, gh), mid(gt_p, gw, gh)
    s = min(360, gh)
    y, x = gh // 2 - s // 2, gw // 2 - s // 2
    crop = lambda im: im[y:y + s, x:x + s]
    cv2.imwrite(str(dst), np.hstack([crop(bic), crop(mod), crop(gt)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    results = json.loads(RESULTS.read_text())
    jobs = {(j["clip"], j["scale"]): j for j in json.loads(JOBS.read_text())}
    M = Metrics()
    rows: list[dict] = []
    baselined: set = set()

    for r in results:
        gt = read_frames(ROOT / r["gt"], args.frames)
        out = read_frames(ROOT / r["output"], args.frames)
        if len(out) < 2 or len(gt) < 2:
            print(f"  !! skip {r['clip']} {r['scale']} {r['model']}: "
                  f"<2 readable frames (out={len(out)}, gt={len(gt)})")
            continue
        m = M.score(out, gt) | {"clip": r["clip"], "scale": r["scale"], "model": r["model"],
                                "cost_usd": r["cost_usd"]}
        rows.append(m)
        print(f"{r['clip']:8s} {r['scale']} {r['model']:11s} "
              f"LPIPS {m['LPIPS']:.3f} MUSIQ {m['MUSIQ']:.1f} E_warp {m['E_warp']:.4f}")

        key = (r["clip"], r["scale"])
        if key not in baselined:
            baselined.add(key)
            j = jobs[key]
            inp = read_frames(ROOT / j["input"], args.frames)
            if len(inp) >= 2:
                rows.append(M.score(inp, gt) | {"clip": r["clip"], "scale": r["scale"],
                                                "model": "bicubic", "cost_usd": 0.0})
            gt_t = to_t(gt)
            rows.append({"clip": r["clip"], "scale": r["scale"], "model": "GT(ref)",
                         "PSNR": float("nan"), "SSIM": 1.0, "LPIPS": 0.0,
                         "MUSIQ": M._nr(M.musiq, gt_t),
                         "E_warp": warping_error(gt)["E_warp_short"], "cost_usd": 0.0})
            try:
                montage(ROOT / j["input"], ROOT / r["output"], ROOT / r["gt"],
                        OUT / f"montage_{r['clip']}_{r['scale']}_{r['model']}.png", j["gt_w"], j["gt_h"])
            except Exception as exc:  # noqa: BLE001 - montage is cosmetic, never fail the report
                print(f"  !! montage skipped for {r['clip']} {r['scale']}: {exc}")

    cols = ["clip", "scale", "model", "PSNR", "SSIM", "LPIPS", "MUSIQ", "E_warp", "cost_usd"]
    with (OUT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    _write_markdown(rows, cols)
    _plot_pareto(rows)
    print(f"\nwrote {OUT/'metrics.csv'}, metrics.md, pareto.png, montages")


def _write_markdown(rows: list[dict], cols: list[str]) -> None:
    lines = ["# VSR comparison — metrics", "",
             "FR (vs GT): PSNR/SSIM/LPIPS. NR: MUSIQ. Temporal: E_warp (read vs GT row).",
             "Direction: PSNR/SSIM/MUSIQ higher better; LPIPS/E_warp lower; E_warp ideal ~= GT.", ""]
    by_key: dict = {}
    for r in rows:
        by_key.setdefault((r["clip"], r["scale"]), []).append(r)
    order = {"bicubic": 0, "realesrgan": 1, "topaz": 2, "seedvr2": 3, "GT(ref)": 9}
    for (clip, scale), rs in by_key.items():
        lines += [f"## {clip} — {scale}", "", "| " + " | ".join(cols[2:]) + " |",
                  "|" + "---|" * len(cols[2:])]
        for r in sorted(rs, key=lambda x: order.get(x["model"], 5)):
            cells = [r["model"]] + [(f"{r[c]:.4f}" if isinstance(r.get(c), float) and c != "cost_usd"
                                     else f"{r[c]:.3f}" if c == "cost_usd" and isinstance(r.get(c), float)
                                     else str(r.get(c, ""))) for c in cols[3:]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "metrics.md").write_text("\n".join(lines))


def _plot_pareto(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r for r in rows if r["model"] not in ("GT(ref)",)]
    colors = {"bicubic": "gray", "realesrgan": "tab:orange", "topaz": "tab:blue", "seedvr2": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (ykey, ylab, inv) in zip(axes, [("LPIPS", "LPIPS (lower better)", True),
                                            ("MUSIQ", "MUSIQ (higher better)", False)]):
        for r in models:
            ax.scatter(r["cost_usd"], r[ykey], c=colors.get(r["model"], "k"), s=60,
                       label=r["model"])
            ax.annotate(f"{r['clip'][:4]}/{r['scale']}", (r["cost_usd"], r[ykey]),
                        fontsize=6, alpha=0.6)
        ax.set_xlabel("cost USD per clip")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in colors.items()]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle("Quality vs cost (Pareto) — each point a clip/scale")
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=120)


if __name__ == "__main__":
    main()

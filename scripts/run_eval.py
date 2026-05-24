"""Run upscalers over the eval jobs with a hard budget guard. SPENDS fal credits.

Reads data/eval/jobs.json. For each (job, model): estimate cost, refuse if cumulative would
exceed --budget, skip if already in results.json (idempotent), call the provider, VERIFY the
output, persist after every success. A single job failure (fal outage, bad output, timeout) is
logged and the batch CONTINUES; failures are not recorded so a re-run retries only them. Staged
via --scale so you can run 1.5x first, prune losers, then run 2x.

    .venv/bin/python scripts/run_eval.py --dry-run
    .venv/bin/python scripts/run_eval.py --scale 1.50x
"""

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from video_super_resolution.providers import PROVIDERS
from video_super_resolution.providers.base import UpscaleResult, probe

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data" / "eval" / "jobs.json"
RESULTS = ROOT / "data" / "eval" / "results.json"


def execute_job(prov, input_path: Path, out_path: Path, gt_h: int) -> UpscaleResult:
    """Run one upscale and verify the output is real (exists, decodable, right height).

    Raises on any failure so the caller can record-and-continue. Verification catches the silent
    failure mode where fal returns a URL but the file is truncated / wrong resolution / 0 frames.
    """
    res = prov.run(input_path, out_path, gt_h)
    if not out_path.exists():
        raise RuntimeError("provider returned but output file is missing")
    ow, oh, _fps, frames = probe(out_path)
    if frames < 2:
        raise RuntimeError(f"output has too few frames ({frames})")
    if abs(oh - gt_h) > max(8, 0.05 * gt_h):
        raise RuntimeError(f"output height {oh} far from target {gt_h}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["topaz", "seedvr2", "realesrgan"])
    ap.add_argument("--scale", help="filter to one scale label, e.g. 1.50x")
    ap.add_argument("--clips", nargs="+", help="filter to these clip names")
    ap.add_argument("--budget", type=float, default=15.0, help="hard cap on cumulative USD")
    ap.add_argument("--dry-run", action="store_true", help="print the cost plan, spend nothing")
    args = ap.parse_args()

    jobs = json.loads(JOBS.read_text())
    if args.scale:
        jobs = [j for j in jobs if j["scale"] == args.scale]
    if args.clips:
        jobs = [j for j in jobs if j["clip"] in args.clips]

    providers = {m: PROVIDERS[m]() for m in args.models}
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {(r["clip"], r["scale"], r["model"]) for r in results}
    spent = sum(r["cost_usd"] for r in results)

    plan = []
    est_total = 0.0
    for j in jobs:
        try:
            iw, ih, fps, frames = probe(ROOT / j["input"])
        except Exception as exc:  # noqa: BLE001 - unreadable input, skip this clip
            print(f"  !! skip {j['clip']} {j['scale']}: unreadable input ({exc})")
            continue
        for m, prov in providers.items():
            if (j["clip"], j["scale"], m) in done:
                continue
            est = prov.estimate_cost(iw, ih, frames, fps, j["gt_h"])
            plan.append((j, m, prov, est))
            est_total += est
            print(f"  plan {j['clip']:8s} {j['scale']} {m:11s} ~${est:.3f}")

    print(f"\nalready spent ${spent:.2f} | planned {len(plan)} runs | est +${est_total:.2f} "
          f"| budget ${args.budget:.2f}")
    if spent + est_total > args.budget:
        sys.exit(f"REFUSING: would exceed budget (${spent + est_total:.2f} > ${args.budget:.2f})")
    if args.dry_run or not plan:
        return

    out_dir = ROOT / "data" / "eval" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for j, m, prov, est in plan:
        out = out_dir / f"{j['clip']}_{j['scale']}_{m}.mp4"
        print(f"\nrun {j['clip']} {j['scale']} {m} (~${est:.3f}) -> {out.name}")
        try:
            res = execute_job(prov, ROOT / j["input"], out, j["gt_h"])
        except Exception as exc:  # noqa: BLE001 - one job's failure must not abort the batch
            print(f"  !! FAILED {j['clip']} {j['scale']} {m}: {exc}")
            failures.append((j["clip"], j["scale"], m, str(exc)))
            continue
        results.append({"clip": j["clip"], "axis": j["axis"], "scale": j["scale"], "model": m,
                        "output": str(out.relative_to(ROOT)), "gt": j["gt"],
                        "gt_w": j["gt_w"], "gt_h": j["gt_h"], "cost_usd": res.cost_usd})
        RESULTS.write_text(json.dumps(results, indent=2))
        print(f"  done, cost ${res.cost_usd:.3f}, cumulative ${sum(r['cost_usd'] for r in results):.2f}")

    if failures:
        print(f"\n{len(failures)} job(s) FAILED (re-run to retry — completed jobs are skipped):")
        for clip, scale, m, err in failures:
            print(f"  - {clip} {scale} {m}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Self-hosted upscaler providers. These satisfy the `Upscaler` protocol but run inference in a
separate torch+CUDA venv (.venv-onprem) via subprocess, so the Streamlit/eval venv needs no GPU
stack. Progress is parsed from the worker's stdout and forwarded to an optional callback.
"""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from ..providers.base import UpscaleResult, probe
from .weights import realesrgan_is_cached

ROOT = Path(__file__).resolve().parents[3]
_RE_WORKER = ROOT / "src" / "video_super_resolution" / "local" / "realesrgan_worker.py"
_SV_WORKER = ROOT / "src" / "video_super_resolution" / "local" / "seedvr2_worker.py"
_SERVE = ROOT / "scripts" / "serve_batch.py"

ProgressCb = Callable[[str, float | None, str], None]


def _onprem_candidates() -> list[Path]:
    env = os.environ.get("ONPREM_PYTHON")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(ROOT / ".venv-onprem" / "bin" / "python")
    candidates.append(ROOT.parent / "qlan-vsr" / ".venv-onprem" / "bin" / "python")
    seen: set[Path] = set()
    uniq: list[Path] = []
    for path in candidates:
        if path not in seen:
            uniq.append(path)
            seen.add(path)
    return uniq


def onprem_python() -> Path:
    for path in _onprem_candidates():
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in _onprem_candidates())
    raise RuntimeError(
        "on-prem venv not found. Looked in: "
        f"{searched}. Set ONPREM_PYTHON or create .venv-onprem with torch+CUDA, basicsr and "
        "realesrgan.")


def _stream(cmd: list[str], progress: ProgressCb | None) -> dict:
    """Run a worker, forward STAGE/PROGRESS lines to `progress`, return the parsed DONE fields."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    done: dict = {}
    error: str | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            i, n = (int(x) for x in line.split()[1:3])
            if progress:
                progress("infer", i / n if n else None, f"frame {i}/{n}")
        elif line.startswith("STAGE "):
            if progress:
                progress("stage", None, line[6:])
        elif line.startswith("DONE "):
            ow, oh, n, secs = line.split()[1:5]
            done = {"out_w": int(ow), "out_h": int(oh), "frames": int(n), "seconds": float(secs)}
        elif line.startswith("ERROR "):
            error = line[6:]
    proc.wait()
    if proc.returncode != 0 or error:
        raise RuntimeError(error or f"worker exited {proc.returncode}")
    return done


class LocalRealEsrganUpscaler:
    name = "realesrgan-local"

    def __init__(self, tile: int = 256, tile_pad: int = 10, half: bool = False,
                 model: str = "RealESRGAN_x2plus", native_scale: int = 2):
        self.tile = tile
        self.tile_pad = tile_pad
        self.half = half
        self.model = model
        self.native_scale = native_scale

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float:
        return 0.0  # self-hosted: GPU time only, no per-call charge

    def weights_cached(self) -> bool:
        return realesrgan_is_cached(self.model)

    def run(self, input_path: Path, out_path: Path, out_h: int,
            progress: ProgressCb | None = None) -> UpscaleResult:
        cmd = [str(onprem_python()), str(_RE_WORKER), "--input", str(input_path),
               "--output", str(out_path), "--out-h", str(out_h), "--model", self.model,
               "--native-scale", str(self.native_scale), "--tile", str(self.tile),
               "--tile-pad", str(self.tile_pad)]
        if self.half:
            cmd.append("--half")
        _stream(cmd, progress)
        return UpscaleResult(self.name, str(out_path), 0.0)


class LocalSeedVR2Upscaler:
    name = "seedvr2-local"

    def __init__(self, variant: str = "3B", batch: int = 5, temporal_overlap: int = 4,
                 tile: int = 0, min_vram_gb: float = 0.0, download: bool = False):
        self.variant = variant
        self.batch = batch
        self.temporal_overlap = temporal_overlap
        self.tile = tile
        self.min_vram_gb = min_vram_gb
        self.download = download

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float:
        return 0.0

    def run(self, input_path: Path, out_path: Path, out_h: int,
            progress: ProgressCb | None = None) -> UpscaleResult:
        cmd = [str(onprem_python()), str(_SV_WORKER), "--input", str(input_path),
               "--output", str(out_path), "--out-h", str(out_h), "--variant", self.variant,
               "--batch", str(self.batch), "--temporal-overlap", str(self.temporal_overlap),
               "--tile", str(self.tile), "--min-vram-gb", str(self.min_vram_gb)]
        if self.download:
            cmd.append("--download")
        _stream(cmd, progress)
        return UpscaleResult(self.name, str(out_path), 0.0)


LOCAL_PROVIDERS: dict[str, type] = {
    "realesrgan-local": LocalRealEsrganUpscaler,
    "seedvr2-local": LocalSeedVR2Upscaler,
}


def run_batch_queue(jobs: list[dict], out_dir: Path, progress: ProgressCb | None = None,
                    model: str = "realesrgan", color: str = "none", color_strength: float = 1.0,
                    wavelet_levels: int = 5, headroom: float = 0.10, max_batch: int = 0,
                    fp32: bool = False, vram_gb: float = 0.0, tile: int = 0,
                    tile_pad: int = 10, variant: str = "3B",
                    temporal_batch: int = 5, temporal_overlap: int = 2,
                    download: bool = False) -> list[dict]:
    """Drain a QUEUE of videos through one GPU-saturating batched run (cross-video batching, batch
    size calibrated to ~90% VRAM). `jobs` = [{input, out_h, name}]. Fully configurable: model, color
    post-processing, batch cap, fp16, SeedVR2 windowing. Returns finished videos as
    [{video_id, frames, out_path}]. The heavy run happens in .venv-onprem via serve_batch.py.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(jobs))
    cmd = [str(onprem_python()), str(_SERVE), "--manifest", str(manifest), "--out-dir", str(out_dir),
           "--model", model, "--headroom", str(headroom), "--max-batch", str(max_batch),
           "--vram-gb", str(vram_gb), "--color", color, "--color-strength", str(color_strength),
           "--wavelet-levels", str(wavelet_levels), "--tile", str(tile),
           "--tile-pad", str(tile_pad), "--variant", variant,
           "--temporal-batch", str(temporal_batch), "--temporal-overlap", str(temporal_overlap)]
    if fp32:
        cmd.append("--fp32")
    if download:
        cmd.append("--download")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    videos: list[dict] = []
    error: str | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            i, n = (int(x) for x in line.split()[1:3])
            if progress:
                progress("infer", i / n if n else None, f"{i}/{n} frames")
        elif line.startswith("STAGE "):
            if progress:
                progress("stage", None, line[6:])
        elif line.startswith("VIDEO "):
            vid, frames, path = line.split(maxsplit=3)[1:]
            videos.append({"video_id": vid, "frames": int(frames), "out_path": path})
            if progress:
                progress("video", None, f"{vid} done ({frames} frames)")
        elif line.startswith("ERROR "):
            error = line[6:]
    proc.wait()
    if proc.returncode != 0 or error:
        raise RuntimeError(error or f"serve_batch exited {proc.returncode}")
    return videos

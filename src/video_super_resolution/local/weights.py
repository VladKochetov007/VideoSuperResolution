"""Weight resolution for self-hosted models: use the local cache if present, else download with a
progress callback. No torch here, so this imports cheaply from either venv.

`progress(stage, frac, msg)` — stage in {"cached","download","done"}, frac in [0,1] (or None when
unknown). The webui forwards it to a Streamlit progress bar; CLI workers print it.
"""

import urllib.request
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "weights"
LEGACY_CACHE = ROOT.parent / "qlan-vsr" / "weights"
_MIN_BYTES = 1_000_000

ProgressCb = Callable[[str, float | None, str], None]

# Official Real-ESRGAN release assets. One x2 model serves any outscale (RealESRGANer resamples).
REALESRGAN_URLS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}
# SeedVR2 open weights on the Hub. Configurable; defaults to the 3B variant.
SEEDVR2_REPOS = {"3B": "ByteDance-Seed/SeedVR2-3B", "7B": "ByteDance-Seed/SeedVR2-7B"}
# Rough VRAM floor (fp16 + activations) per variant; the user can override.
SEEDVR2_MIN_VRAM_GB = {"3B": 18.0, "7B": 40.0}


def _download(url: str, dst: Path, progress: ProgressCb | None, label: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https release URLs
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                read += len(chunk)
                if progress:
                    frac = read / total if total else None
                    progress("download", frac, f"{label}: {read >> 20}/{total >> 20 or '?'} MB")
    tmp.rename(dst)
    if progress:
        progress("done", 1.0, f"{label} ready")


def ensure_realesrgan_weights(name: str = "RealESRGAN_x2plus", cache_dir: Path | None = None,
                              progress: ProgressCb | None = None) -> Path:
    if name not in REALESRGAN_URLS:
        raise ValueError(f"unknown Real-ESRGAN weight {name!r}; have {sorted(REALESRGAN_URLS)}")
    caches = [Path(cache_dir)] if cache_dir is not None else [DEFAULT_CACHE, LEGACY_CACHE]
    for cache in caches:
        dst = cache / f"{name}.pth"
        if dst.exists() and dst.stat().st_size > _MIN_BYTES:
            if progress:
                progress("cached", 1.0, f"{name} cached")
            return dst
    cache = Path(cache_dir or DEFAULT_CACHE)
    dst = cache / f"{name}.pth"
    _download(REALESRGAN_URLS[name], dst, progress, name)
    return dst


def realesrgan_is_cached(name: str = "RealESRGAN_x2plus", cache_dir: Path | None = None) -> bool:
    caches = [Path(cache_dir)] if cache_dir is not None else [DEFAULT_CACHE, LEGACY_CACHE]
    for cache in caches:
        dst = cache / f"{name}.pth"
        if dst.exists() and dst.stat().st_size > _MIN_BYTES:
            return True
    return False


def ensure_seedvr2_weights(variant: str = "3B", cache_dir: Path | None = None,
                           progress: ProgressCb | None = None) -> Path:
    """Snapshot the SeedVR2 weights from the Hub into the local cache. Multi-GB — only call when the
    user/host opts in. Raises a clear message if huggingface_hub is unavailable."""
    if variant not in SEEDVR2_REPOS:
        raise ValueError(f"unknown SeedVR2 variant {variant!r}; have {sorted(SEEDVR2_REPOS)}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("huggingface_hub not installed in this environment") from exc
    cache = Path(cache_dir or DEFAULT_CACHE) / f"SeedVR2-{variant}"
    if progress:
        progress("download", None, f"SeedVR2-{variant} from {SEEDVR2_REPOS[variant]} (multi-GB)")
    path = snapshot_download(SEEDVR2_REPOS[variant], local_dir=str(cache))
    if progress:
        progress("done", 1.0, f"SeedVR2-{variant} ready")
    return Path(path)


def seedvr2_is_cached(variant: str = "3B", cache_dir: Path | None = None) -> bool:
    cache = Path(cache_dir or DEFAULT_CACHE) / f"SeedVR2-{variant}"
    return cache.is_dir() and any(cache.iterdir())

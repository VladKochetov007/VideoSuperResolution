"""Self-hosted model support.

`weights` is import-light (no torch, no fal) so the on-prem inference workers can import it. The
`provider` bridge pulls in the fal-coupled providers package, so it is loaded lazily — only the
Streamlit/eval venv (which has those deps) ever touches it, never the on-prem worker.
"""

from .weights import (
    ensure_realesrgan_weights,
    ensure_seedvr2_weights,
    realesrgan_is_cached,
    seedvr2_is_cached,
)

_LAZY = {"LOCAL_PROVIDERS", "LocalRealEsrganUpscaler", "LocalSeedVR2Upscaler", "onprem_python",
         "run_batch_queue"}

__all__ = ["LOCAL_PROVIDERS", "LocalRealEsrganUpscaler", "LocalSeedVR2Upscaler", "onprem_python",
           "run_batch_queue", "ensure_realesrgan_weights", "ensure_seedvr2_weights",
           "realesrgan_is_cached", "seedvr2_is_cached"]


def __getattr__(name: str):
    if name in _LAZY:
        from . import provider

        return getattr(provider, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

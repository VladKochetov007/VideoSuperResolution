"""Streamlit-free core for the webui: pick a model, upscale, optionally color-correct, measure.

All UI-independent so it is testable and reusable. The Streamlit script is a thin adapter over
`run_upscale`. Real fal models spend credits and are gated behind `allow_spend`; the default `mock`
model is offline and free, so the UI demo wastes no compute.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import msgspec
import numpy as np

from ..local import LOCAL_PROVIDERS
from ..media import has_audio, mux_audio
from ..postprocess import ColorConfig, color_drift, correct_file, make_corrector
from ..providers import PROVIDERS
from ..providers.base import probe
from ..providers.mock import MockUpscaler

ProgressCb = Callable[[str, float | None, str], None]

# mock (offline) + fal (billable) + self-hosted (GPU, free per-call).
WEBUI_MODELS: dict[str, type] = {"mock": MockUpscaler, **PROVIDERS, **LOCAL_PROVIDERS}


class ProcessResult(msgspec.Struct, frozen=True):
    model: str
    out_path: str
    corrected_path: str | None
    display_path: str  # H.264 + source audio, web-playable; what the UI shows and downloads
    cost_usd: float
    in_dims: tuple[int, int]
    out_dims: tuple[int, int]
    frames: int
    color_method: str
    delta_e_raw: float | None
    delta_e_corrected: float | None


def _sample_drift(out_path: Path, source_path: Path, n: int) -> float:
    out, src = cv2.VideoCapture(str(out_path)), cv2.VideoCapture(str(source_path))
    vals: list[float] = []
    try:
        while len(vals) < n:
            ok_o, fo = out.read()
            ok_s, fs = src.read()
            if not (ok_o and ok_s):
                break
            vals.append(color_drift(fo, fs))
    finally:
        out.release()
        src.release()
    return float(np.mean(vals)) if vals else float("nan")


def estimate_cost(model: str, input_path: Path, out_h: int) -> float:
    in_w, in_h, fps, frames = probe(input_path)
    return WEBUI_MODELS[model]().estimate_cost(in_w, in_h, frames, fps, out_h)


def run_upscale(input_path: Path, model: str, out_h: int, color: ColorConfig, out_dir: Path,
                allow_spend: bool = False, drift_frames: int = 8,
                provider_kwargs: dict | None = None,
                progress: ProgressCb | None = None) -> ProcessResult:
    """Upscale one clip, optionally color-correct, and measure ΔE2000 drift before/after.

    `provider_kwargs` configures the chosen provider (e.g. tile/batch for self-hosted models).
    `progress(stage, frac, msg)` is forwarded to self-hosted workers. Raises if a billable model
    (cost > 0, i.e. fal) is requested without `allow_spend`; mock and self-hosted models are free.
    """
    if model not in WEBUI_MODELS:
        raise ValueError(f"unknown model {model!r}; have {sorted(WEBUI_MODELS)}")
    prov = WEBUI_MODELS[model](**(provider_kwargs or {}))
    in_w, in_h, fps, frames_in = probe(input_path)
    cost = prov.estimate_cost(in_w, in_h, frames_in, fps, out_h)
    if cost > 0 and not allow_spend:
        raise RuntimeError(f"model {model!r} spends ~${cost:.3f} of fal credits; pass allow_spend=True")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_{model}_{out_h}p.mp4"
    res = (prov.run(input_path, out_path, out_h, progress=progress)
           if model in LOCAL_PROVIDERS else prov.run(input_path, out_path, out_h))

    iw, ih, _fps, _n = probe(input_path)
    ow, oh, _f2, frames = probe(out_path)

    corrected_path: str | None = None
    de_raw = de_fixed = None
    corrector = make_corrector(color)
    if corrector is not None:
        de_raw = _sample_drift(out_path, input_path, drift_frames)
        cp = out_dir / f"{input_path.stem}_{model}_{out_h}p_{color.method}.mp4"
        correct_file(out_path, input_path, cp, corrector)
        corrected_path = str(cp)
        de_fixed = _sample_drift(cp, input_path, drift_frames)

    # The corrector and MockUpscaler already emit H.264 + source audio. A raw fal result is H.264
    # but may have dropped the audio, so restore it for the path the UI shows/downloads.
    display = corrected_path or str(out_path)
    if corrected_path is None and not has_audio(out_path):
        webp = out_dir / f"{input_path.stem}_{model}_{out_h}p_audio.mp4"
        mux_audio(out_path, input_path, webp)
        display = str(webp)

    return ProcessResult(
        model=model, out_path=str(out_path), corrected_path=corrected_path, display_path=display,
        cost_usd=res.cost_usd, in_dims=(iw, ih), out_dims=(ow, oh), frames=frames,
        color_method=color.method, delta_e_raw=de_raw, delta_e_corrected=de_fixed,
    )

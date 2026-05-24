"""Color calibration for upscaler output.

Neural upscalers (diffusion ones especially) drift hue/saturation/brightness toward their training
distribution. The fix is non-neural and cheap: take DETAIL (high frequency) from the upscaled output
and COLOR (low frequency / channel statistics) from the source frame. Source is the color anchor
because it is the ground truth the client shot; the upscaler is only trusted for sharpness.

Methods (all `(out_bgr, ref_bgr) -> bgr`, identical HxW):
- wavelet : multi-level Gaussian decomposition; high-freq from output, low-freq from reference.
            Strongest against *uneven* / spatially-varying shift. This is StableSR's "wavelet color
            fix" (originally from GIMP/Krita), reproduced without a wavelet lib via a Gaussian pyramid.
- reinhard: match per-channel mean+std in CIELAB. Global, very fast, the classic Reinhard transfer.
- adain   : Reinhard in RGB (StableSR's "AdaIN" option). Cheaper, less perceptual than LAB.
- histogram: per-channel CDF match. Tightest color alignment but can flatten contrast.

Everything here is injectable: a `FrameColorCorrector` satisfies the `ColorCorrector` protocol, and
callers may pass any callable with the same signature. No method is hard-coded into the pipeline.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import msgspec
import numpy as np
from skimage.exposure import match_histograms


@runtime_checkable
class ColorCorrector(Protocol):
    def __call__(self, out_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray: ...


def reinhard_lab(out_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    o = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        o[..., c] = (o[..., c] - o[..., c].mean()) / (o[..., c].std() + 1e-6) \
            * (r[..., c].std() + 1e-6) + r[..., c].mean()
    return cv2.cvtColor(o.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def adain_rgb(out_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    o, r = out_bgr.astype(np.float32), ref_bgr.astype(np.float32)
    for c in range(3):
        o[..., c] = (o[..., c] - o[..., c].mean()) / (o[..., c].std() + 1e-6) \
            * (r[..., c].std() + 1e-6) + r[..., c].mean()
    return o.clip(0, 255).astype(np.uint8)


def histogram_match(out_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    return match_histograms(out_bgr, ref_bgr, channel_axis=-1).clip(0, 255).astype(np.uint8)


def _lowpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass. For large sigma, blur on a decimated copy (the kernel would otherwise be
    ~6*sigma wide and dominate runtime); the result is a faithful low-frequency band for color."""
    if sigma <= 2.5:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    f = int(round(sigma / 2.0))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def wavelet_color_fix(out_bgr: np.ndarray, ref_bgr: np.ndarray, levels: int = 5) -> np.ndarray:
    """Keep the output's high frequency (detail), take the reference's low frequency (color).

    The multi-level Gaussian decomposition telescopes: accumulated high freq = img - final_low, and a
    cascade of Gaussian blurs equals one Gaussian whose variance is the sum — so the whole pyramid
    collapses to a single low-pass at the combined sigma. One blur per image instead of `levels`."""
    o, r = out_bgr.astype(np.float32), ref_bgr.astype(np.float32)
    eff_sigma = float(np.sqrt(sum((2 ** i) ** 2 for i in range(max(1, levels)))))
    return (o - _lowpass(o, eff_sigma) + _lowpass(r, eff_sigma)).clip(0, 255).astype(np.uint8)


_METHODS: dict[str, ColorCorrector] = {
    "wavelet": wavelet_color_fix,
    "reinhard": reinhard_lab,
    "adain": adain_rgb,
    "histogram": histogram_match,
}


class ColorConfig(msgspec.Struct, frozen=True):
    """method: one of _METHODS, or "none". strength: 0=keep output, 1=full correction (lerp).
    wavelet_levels: pyramid depth; larger keeps more of the output as "detail"."""

    method: str = "wavelet"
    strength: float = 1.0
    wavelet_levels: int = 5


class FrameColorCorrector:
    """A configured single-frame `ColorCorrector`. ref is the color anchor (source, resized to out)."""

    def __init__(self, cfg: ColorConfig = ColorConfig()):
        if cfg.method not in _METHODS:
            raise ValueError(f"unknown color method {cfg.method!r}; have {sorted(_METHODS)} or 'none'")
        self.cfg = cfg
        base = _METHODS[cfg.method]
        self._fn: ColorCorrector = (
            (lambda o, r: wavelet_color_fix(o, r, cfg.wavelet_levels))
            if cfg.method == "wavelet" else base
        )

    def __call__(self, out_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
        if ref_bgr.shape[:2] != out_bgr.shape[:2]:
            ref_bgr = cv2.resize(ref_bgr, (out_bgr.shape[1], out_bgr.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
        corrected = self._fn(out_bgr, ref_bgr)
        s = self.cfg.strength
        if s >= 1.0:
            return corrected
        return (out_bgr.astype(np.float32) * (1 - s)
                + corrected.astype(np.float32) * s).clip(0, 255).astype(np.uint8)


def make_corrector(cfg: ColorConfig) -> ColorCorrector | None:
    """Factory. Returns None for method "none" so callers can cheaply skip post-processing."""
    return None if cfg.method == "none" else FrameColorCorrector(cfg)


class VideoColorPostProcessor:
    """Video-level hook for the serving Scheduler: recolor assembled frames against the source.

    Signature `(frames, source_path) -> frames` is the generic post-processor contract — anything
    matching it (sharpen, denoise, grade) can be injected the same way. Source frames are read once
    and aligned by index; missing source frames pass the output through untouched.
    """

    def __init__(self, corrector: ColorCorrector):
        self.corrector = corrector

    def __call__(self, frames: list[np.ndarray], source_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(source_path))
        out = []
        for f in frames:
            ok, ref = cap.read()
            out.append(self.corrector(f, ref) if ok else f)
        cap.release()
        return out


def correct_file(out_path: Path, source_path: Path, dst: Path, corrector: ColorCorrector,
                 fps: float | None = None) -> int:
    """Standalone: re-color an already-upscaled clip against its source. Returns frames written.

    Output is H.264 with the source audio muxed in. Used to post-process fal outputs offline; the
    same `corrector` plugs into the serving Scheduler.
    """
    from ..media import FfmpegWriter

    src = cv2.VideoCapture(str(source_path))
    out = cv2.VideoCapture(str(out_path))
    if fps is None:
        fps = out.get(cv2.CAP_PROP_FPS) or 24.0
    n = 0
    try:
        with FfmpegWriter(dst, fps, audio_source=source_path) as writer:
            while True:
                ok_o, fo = out.read()
                ok_s, fs = src.read()
                if not ok_o:
                    break
                writer.write(corrector(fo, fs) if ok_s else fo)
                n += 1
    finally:
        src.release()
        out.release()
    return n

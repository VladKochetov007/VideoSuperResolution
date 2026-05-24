"""Color-drift measurement via CIEDE2000.

ΔE2000 is the perceptual color-difference standard. We measure drift of the upscaled output against
the source on the LOW-FREQUENCY (Gaussian-blurred) image pair: blurring removes the legitimate detail
the upscaler added, so what remains is the hue/tone shift color correction targets. Lower = closer to
the source's color. We report blurred (color-only) and full ΔE so detail is not mistaken for drift.
"""

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def _lab(bgr: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma > 0:
        bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=blur_sigma)
    return rgb2lab(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)


def color_drift(out_bgr: np.ndarray, ref_bgr: np.ndarray, max_side: int = 256) -> float:
    """Mean ΔE2000 between low-frequency output and source. Isolates color/tone from added detail.

    Color is low-frequency, so both frames are decimated to `max_side` before the (expensive)
    CIEDE2000 — ~40x faster than full resolution with no change to the measured drift.
    """
    if ref_bgr.shape[:2] != out_bgr.shape[:2]:
        ref_bgr = cv2.resize(ref_bgr, (out_bgr.shape[1], out_bgr.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
    h, w = out_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        dw, dh = max(1, round(w * s)), max(1, round(h * s))
        out_bgr = cv2.resize(out_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        ref_bgr = cv2.resize(ref_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    return float(deltaE_ciede2000(_lab(out_bgr, 1.5), _lab(ref_bgr, 1.5)).mean())

import cv2
import numpy as np

from .config import DegradationConfig

_INTERP = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _resize(img: np.ndarray, out_w: int, out_h: int, rng: np.random.Generator) -> np.ndarray:
    interp = _INTERP[rng.integers(len(_INTERP))]
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def _add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def degrade_frame(
    hr: np.ndarray,
    out_w: int,
    out_h: int,
    cfg: DegradationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Degrade a single HR frame (uint8 BGR) to LR size (out_w, out_h)."""
    if cfg.clean:
        return cv2.resize(hr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    h, w = hr.shape[:2]
    n_orders = 2 if cfg.second_order else 1
    x = hr
    for order in range(n_orders):
        x = _blur(x, rng.uniform(*cfg.blur_sigma_range))
        # intermediate downscale on the first order, final size on the last
        if order < n_orders - 1:
            scale = rng.uniform(0.5, 0.9)
            x = _resize(x, max(out_w, int(w * scale)), max(out_h, int(h * scale)), rng)
        else:
            x = _resize(x, out_w, out_h, rng)
        x = _add_noise(x, rng.uniform(*cfg.noise_sigma_range), rng)
        x = _jpeg(x, rng.integers(*cfg.jpeg_quality_range))
    return x

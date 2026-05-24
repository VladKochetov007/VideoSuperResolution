import cv2
import numpy as np

from ..config import TemporalConfig


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def optical_flow(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Dense flow mapping pixels of `src` to `dst` (Farneback). Shape (H, W, 2): (dx, dy)."""
    return cv2.calcOpticalFlowFarneback(
        _gray(src), _gray(dst), None, 0.5, 3, 15, 3, 5, 1.2, 0
    )


def backward_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample `img` at (x+flow_x, y+flow_y) for every target pixel (cv2.remap)."""
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, thresh: float) -> np.ndarray:
    """Valid (non-occluded) mask via forward-backward consistency.

    A pixel is valid where following the forward flow then the backward flow returns near the
    origin: |flow_fwd(p) + flow_bwd(p + flow_fwd(p))| < thresh.
    """
    bwd_at_fwd = backward_warp(flow_bwd, flow_fwd)
    residual = flow_fwd + bwd_at_fwd
    dist = np.sqrt((residual**2).sum(axis=-1))
    return (dist < thresh).astype(np.float32)


def _warp_error_pair(a: np.ndarray, b: np.ndarray, cfg: TemporalConfig) -> float:
    """Mean masked L1 between b and (a warped toward b). Inputs uint8 BGR."""
    flow_ab = optical_flow(b, a)  # for each pixel of b, where it came from in a
    flow_ba = optical_flow(a, b)
    mask = occlusion_mask(flow_ab, flow_ba, cfg.fb_consistency_thresh)[..., None]
    warped_a = backward_warp(a, flow_ab).astype(np.float32)
    diff = np.abs(b.astype(np.float32) - warped_a) / 255.0
    denom = mask.sum() * b.shape[-1] + 1e-8
    return float((diff * mask).sum() / denom)


def warping_error(frames: list[np.ndarray], cfg: TemporalConfig | None = None) -> dict[str, float]:
    """Short-term (consecutive) and optional long-term (vs first frame) warping error.

    Returns mean E_warp over the sequence. Lower = less flicker. Needs >= ~30 frames to be stable.
    """
    cfg = cfg or TemporalConfig()
    if len(frames) < 2:
        raise ValueError("warping_error needs at least 2 frames")

    short = [_warp_error_pair(frames[t], frames[t + 1], cfg) for t in range(len(frames) - 1)]
    out = {"E_warp_short": float(np.mean(short))}
    if cfg.long_term:
        long = [_warp_error_pair(frames[0], frames[t], cfg) for t in range(1, len(frames))]
        out["E_warp_long"] = float(np.mean(long))
    return out

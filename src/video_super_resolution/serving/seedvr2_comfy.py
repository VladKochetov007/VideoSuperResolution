"""Self-hosted SeedVR2 backend that drives the ComfyUI-SeedVR2 inference core WITHOUT ComfyUI.

SeedVR2-3B is a diffusion-transformer VSR model; running it on a small GPU needs GGUF quantization
plus block-swap (transformer blocks streamed CPU<->GPU per step) and CPU offload of the DiT/VAE. The
upstream `numz/ComfyUI-SeedVR2_VideoUpscaler` node implements exactly that and exposes a runtime-free
core, `_process_frames_core`, which this wrapper calls directly. flash-attn is optional: the node
falls back to torch SDPA, and picks up flash-attn automatically when it is installed on the host.

The node owns its own temporal batching (causal VAE wants 4n+1 windows) and overlap blending, so this
backend feeds it WHOLE clips rather than the cross-video windows the Scheduler builds for Real-ESRGAN.
The heavy model is loaded once and reused across clips via the node's runner cache.

Layout knobs (all overridable, nothing hard-coded):
- repo_dir: the cloned node checkout. Resolution order: SEEDVR2_COMFY_REPO env, repo-local
  third_party/ComfyUI-SeedVR2, sibling ../qlan-vsr/third_party/ComfyUI-SeedVR2.
- model_dir: where GGUF weights are cached/downloaded (SEEDVR2_MODEL_DIR env or a default).
"""

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_REPO_CANDIDATES = [
    _ROOT / "third_party" / "ComfyUI-SeedVR2",
    _ROOT.parent / "qlan-vsr" / "third_party" / "ComfyUI-SeedVR2",
]
_MODEL_DIR_CANDIDATES = [
    _ROOT / "weights" / "seedvr2-comfy",
    _ROOT.parent / "qlan-vsr" / "weights" / "seedvr2-comfy",
]


def resolve_model_dir(model_dir: Path | None = None) -> Path:
    """Where GGUF/VAE weights live. Prefer an explicit/env dir, then any candidate that already
    holds weights (reuse a sibling cache), else the repo-local dir as the download target."""
    if model_dir:
        return Path(model_dir)
    if os.environ.get("SEEDVR2_MODEL_DIR"):
        return Path(os.environ["SEEDVR2_MODEL_DIR"])
    for cand in _MODEL_DIR_CANDIDATES:
        if cand.is_dir() and any(cand.glob("*.safetensors")):
            return cand
    return _MODEL_DIR_CANDIDATES[0]


def resolve_repo_dir(repo_dir: Path | None = None) -> Path:
    candidates = [repo_dir] if repo_dir else []
    if os.environ.get("SEEDVR2_COMFY_REPO"):
        candidates.append(Path(os.environ["SEEDVR2_COMFY_REPO"]))
    candidates += _REPO_CANDIDATES
    for cand in candidates:
        if cand and (cand / "inference_cli.py").is_file():
            return cand
    searched = "\n  ".join(str(c) for c in candidates if c)
    raise RuntimeError(
        "ComfyUI-SeedVR2 checkout not found (inference_cli.py). Clone "
        "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler and point SEEDVR2_COMFY_REPO at it.\n"
        f"  searched:\n  {searched}")


class SeedVR2ComfyRunner:
    """Loads the SeedVR2 model once and upscales whole clips. Tuned for low-VRAM block-swap by
    default; raise batch_size / drop blocks_to_swap on a large GPU for speed and temporal quality."""

    def __init__(self, repo_dir: Path | None = None, model_dir: Path | None = None,
                 dit_model: str = "seedvr2_ema_3b-Q4_K_M.gguf", batch_size: int = 1,
                 blocks_to_swap: int = 32, swap_io_components: bool = True, offload: str = "cpu",
                 attention_mode: str = "sdpa", vae_tile: int = 256, vae_tile_overlap: int = 64,
                 color_correction: str = "none", seed: int = 42):
        self.repo_dir = resolve_repo_dir(repo_dir)
        self.model_dir = resolve_model_dir(model_dir)
        self.dit_model = dit_model
        self.batch_size = batch_size
        self.blocks_to_swap = blocks_to_swap
        self.swap_io_components = swap_io_components
        self.offload = offload
        self.attention_mode = attention_mode
        self.vae_tile = vae_tile
        self.vae_tile_overlap = vae_tile_overlap
        self.color_correction = color_correction
        self.seed = seed
        self._core = None
        self._debug = None
        self._args = None
        self._runner_cache: dict = {}

    def _lazy(self):
        if self._core is not None:
            return
        sys.path.insert(0, str(self.repo_dir))
        from inference_cli import Debug, _process_frames_core, parse_arguments  # noqa: E402

        self._core = _process_frames_core
        self._debug = Debug(enabled=False)
        self._args = self._build_args(parse_arguments)

    def _effective_attention(self) -> str:
        """Fall back to SDPA when the requested kernel is not installed on this host (e.g. flash-attn
        cannot be built on a CUDA-13 box without nvcc). Node modes: sdpa, flash_attn_2/3, sageattn_2/3.
        SDPA is always available."""
        import importlib.util

        m = self.attention_mode
        pkg = ("flash_attn" if m.startswith("flash")
               else "sageattention" if m.startswith("sage") else None)
        if pkg and importlib.util.find_spec(pkg) is None:
            print(f"STAGE attention {m!r} unavailable ({pkg} not installed), falling back to sdpa",
                  flush=True)
            return "sdpa"
        return m

    def _build_args(self, parse_arguments):
        argv = [
            "PLACEHOLDER.mp4",
            "--dit_model", self.dit_model,
            "--model_dir", str(self.model_dir),
            "--resolution", "1080",
            "--batch_size", str(self.batch_size),
            "--dit_offload_device", self.offload,
            "--vae_offload_device", self.offload,
            "--tensor_offload_device", self.offload,
            "--blocks_to_swap", str(self.blocks_to_swap),
            "--vae_encode_tiled", "--vae_encode_tile_size", str(self.vae_tile),
            "--vae_encode_tile_overlap", str(self.vae_tile_overlap),
            "--vae_decode_tiled", "--vae_decode_tile_size", str(self.vae_tile),
            "--vae_decode_tile_overlap", str(self.vae_tile_overlap),
            "--attention_mode", self._effective_attention(),
            "--color_correction", self.color_correction,
            "--seed", str(self.seed),
            "--cache_dit", "--cache_vae",
        ]
        if self.swap_io_components:
            argv.append("--swap_io_components")
        saved = sys.argv
        try:
            sys.argv = ["inference_cli.py", *argv]
            return parse_arguments()
        finally:
            sys.argv = saved

    def upscale(self, frames_bgr: np.ndarray, out_h: int) -> np.ndarray:
        """frames_bgr: (T,H,W,3) uint8 BGR (OpenCV order). Returns (T,H',W',3) uint8 BGR."""
        import torch

        self._lazy()
        self._args.resolution = out_h
        rgb = np.ascontiguousarray(frames_bgr[..., ::-1])
        tensor = torch.from_numpy(rgb).to(torch.float16).div_(255.0)
        result = self._core(frames_tensor=tensor, args=self._args, device_id="0",
                            debug=self._debug, runner_cache=self._runner_cache)
        out = result.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(out[..., ::-1])

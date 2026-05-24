from pathlib import Path

from ..config import DEFAULT_PRICING, FalPricing
from ..cost import seedvr2_cost_usd
from . import _fal
from .base import UpscaleResult, probe

_TARGET = {720: "720p", 1080: "1080p", 1440: "1440p", 2160: "2160p"}


class SeedVR2Upscaler:
    name = "seedvr2"
    endpoint = "fal-ai/seedvr/upscale/video"

    def __init__(self, seed: int | None = None, pricing: FalPricing = DEFAULT_PRICING):
        self.seed = seed
        self.pricing = pricing

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float:
        out_w = round(in_w * out_h / in_h)
        return seedvr2_cost_usd(out_w, out_h, frames, self.pricing)

    def run(self, input_path: Path, out_path: Path, out_h: int) -> UpscaleResult:
        in_w, in_h, fps, frames = probe(input_path)
        _fal.ensure_fal_key()
        url = _fal.upload(input_path)
        args: dict = {"video_url": url}
        if out_h in _TARGET:
            args |= {"upscale_mode": "target", "target_resolution": _TARGET[out_h]}
        else:
            args |= {"upscale_mode": "factor", "upscale_factor": round(out_h / in_h, 4)}
        if self.seed is not None:
            args["seed"] = self.seed
        res = _fal.subscribe(self.endpoint, args)
        out_url = res["video"]["url"]
        _fal.download(out_url, out_path)
        return UpscaleResult(self.name, str(out_path),
                             self.estimate_cost(in_w, in_h, frames, fps, out_h), url, out_url)

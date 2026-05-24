from pathlib import Path

from ..config import DEFAULT_PRICING, FalPricing
from ..cost import realesrgan_cost_usd
from . import _fal
from .base import UpscaleResult, probe


class RealEsrganUpscaler:
    name = "realesrgan"
    endpoint = "fal-ai/video-upscaler"

    def __init__(self, pricing: FalPricing = DEFAULT_PRICING):
        self.pricing = pricing

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float:
        out_w = round(in_w * out_h / in_h)
        return realesrgan_cost_usd(out_w, out_h, frames, self.pricing)

    def run(self, input_path: Path, out_path: Path, out_h: int) -> UpscaleResult:
        in_w, in_h, fps, frames = probe(input_path)
        scale = round(out_h / in_h, 4)
        _fal.ensure_fal_key()
        url = _fal.upload(input_path)
        res = _fal.subscribe(self.endpoint, {"video_url": url, "scale": scale})
        out_url = res["video"]["url"]
        _fal.download(out_url, out_path)
        return UpscaleResult(self.name, str(out_path),
                             self.estimate_cost(in_w, in_h, frames, fps, out_h), url, out_url)

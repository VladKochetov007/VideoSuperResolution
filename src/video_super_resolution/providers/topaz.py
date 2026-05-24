from pathlib import Path

from ..config import DEFAULT_PRICING, FalPricing
from ..cost import topaz_cost_usd
from . import _fal
from .base import UpscaleResult, probe


class TopazUpscaler:
    name = "topaz"
    endpoint = "fal-ai/topaz/upscale/video"

    def __init__(self, model: str = "Proteus", h264: bool = True,
                 pricing: FalPricing = DEFAULT_PRICING):
        self.model = model
        self.h264 = h264
        self.pricing = pricing

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float:
        return topaz_cost_usd(frames / fps, out_h, fps, self.pricing)

    def run(self, input_path: Path, out_path: Path, out_h: int) -> UpscaleResult:
        in_w, in_h, fps, frames = probe(input_path)
        scale = round(out_h / in_h, 4)
        _fal.ensure_fal_key()
        url = _fal.upload(input_path)
        res = _fal.subscribe(self.endpoint, {
            "video_url": url, "upscale_factor": scale, "model": self.model,
            "H264_output": self.h264,
        })
        out_url = res["video"]["url"]
        _fal.download(out_url, out_path)
        return UpscaleResult(self.name, str(out_path),
                             self.estimate_cost(in_w, in_h, frames, fps, out_h), url, out_url)

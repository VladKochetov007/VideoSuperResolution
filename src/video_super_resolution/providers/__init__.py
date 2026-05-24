from .base import Upscaler, UpscaleResult, probe
from .realesrgan import RealEsrganUpscaler
from .seedvr2 import SeedVR2Upscaler
from .topaz import TopazUpscaler

PROVIDERS: dict[str, type] = {
    "topaz": TopazUpscaler,
    "seedvr2": SeedVR2Upscaler,
    "realesrgan": RealEsrganUpscaler,
}

__all__ = ["Upscaler", "UpscaleResult", "probe", "PROVIDERS",
           "TopazUpscaler", "SeedVR2Upscaler", "RealEsrganUpscaler"]

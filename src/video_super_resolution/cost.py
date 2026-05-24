import msgspec

from .config import DEFAULT_PRICING, ClipSpec, FalPricing


def topaz_cost_usd(
    seconds: float,
    out_height: int,
    out_fps: float,
    pricing: FalPricing = DEFAULT_PRICING,
    gaia2: bool = False,
) -> float:
    """Topaz video upscale cost: per-second, tiered by output height."""
    if out_height <= 720:
        rate = pricing.topaz_usd_per_s_le720
    elif out_height <= 1080:
        rate = pricing.topaz_usd_per_s_to1080
    else:
        rate = pricing.topaz_usd_per_s_above1080
    cost = rate * seconds
    if out_fps >= 60:
        cost *= pricing.topaz_60fps_multiplier
    if gaia2:
        cost *= pricing.topaz_gaia2_multiplier
    return cost


def seedvr2_cost_usd(
    out_width: int,
    out_height: int,
    num_frames: int,
    pricing: FalPricing = DEFAULT_PRICING,
) -> float:
    """SeedVR2 video upscale cost: per megapixel of output (W * H * frames)."""
    megapixels = out_width * out_height * num_frames / 1e6
    return megapixels * pricing.seedvr2_usd_per_megapixel


def realesrgan_cost_usd(
    out_width: int,
    out_height: int,
    num_frames: int,
    pricing: FalPricing = DEFAULT_PRICING,
) -> float:
    """Real-ESRGAN (fal-ai/video-upscaler) cost: per megapixel of output (W * H * frames)."""
    megapixels = out_width * out_height * num_frames / 1e6
    return megapixels * pricing.realesrgan_usd_per_megapixel


class ClipModelCost(msgspec.Struct, frozen=True):
    clip: str
    model: str
    usd: float


def estimate_matrix(
    clips: list[ClipSpec],
    out_width: int,
    out_height: int,
    models: tuple[str, ...] = ("topaz", "seedvr2"),
    pricing: FalPricing = DEFAULT_PRICING,
) -> tuple[list[ClipModelCost], float]:
    """Cost of upscaling every clip to (out_width, out_height) with every model.

    Returns the per (clip, model) breakdown and the grand total.
    """
    rows: list[ClipModelCost] = []
    for clip in clips:
        seconds = clip.num_frames / clip.fps
        for model in models:
            if model == "topaz":
                usd = topaz_cost_usd(seconds, out_height, clip.fps, pricing)
            elif model == "seedvr2":
                usd = seedvr2_cost_usd(out_width, out_height, clip.num_frames, pricing)
            elif model == "realesrgan":
                usd = realesrgan_cost_usd(out_width, out_height, clip.num_frames, pricing)
            else:
                raise ValueError(f"unknown model {model!r}; pricing not configured")
            rows.append(ClipModelCost(clip=clip.name, model=model, usd=usd))
    total = sum(r.usd for r in rows)
    return rows, total

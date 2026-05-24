import msgspec


class FalPricing(msgspec.Struct, frozen=True):
    """fal.ai upscaler pricing. Source: fal model pages (verified 2026-05).

    Topaz is priced per second of OUTPUT video, tiered by output height.
    SeedVR2 video is priced per megapixel of output (width * height * frames).
    """

    topaz_usd_per_s_le720: float = 0.01
    topaz_usd_per_s_to1080: float = 0.02
    topaz_usd_per_s_above1080: float = 0.08
    topaz_60fps_multiplier: float = 2.0
    topaz_gaia2_multiplier: float = 0.5
    seedvr2_usd_per_megapixel: float = 0.001
    realesrgan_usd_per_megapixel: float = 0.0008  # fal-ai/video-upscaler (RealESRGAN per-frame)


class DegradationConfig(msgspec.Struct, frozen=True):
    """Realistic HR->LR degradation for the full-reference track.

    Spatial ops are applied per frame here; temporal video-codec compression is applied at the
    clip level via ffmpeg (codec artifacts are inherently temporal). `clean=True` reduces this
    to plain bicubic downscale for the optimistic/academic comparison point.
    """

    blur_sigma_range: tuple[float, float] = (0.2, 2.0)
    noise_sigma_range: tuple[float, float] = (1.0, 12.0)  # on 0-255 scale
    jpeg_quality_range: tuple[int, int] = (40, 95)
    second_order: bool = True
    clean: bool = False
    # clip-level codec pass (applied by the prep script, not per frame)
    codec: str = "libx264"
    codec_crf_range: tuple[int, int] = (20, 32)


class ClipSpec(msgspec.Struct, frozen=True):
    """One test clip and failure axis it is meant to stress."""

    name: str
    content_axis: str
    width: int
    height: int
    fps: float
    num_frames: int
    track: str = "A"  # "A" = full-reference (HR master kept), "B" = no-reference (real degraded)
    source_url: str | None = None


class TemporalConfig(msgspec.Struct, frozen=True):
    """Flow-based temporal-metric settings."""

    flow: str = "farneback"  # production: "raft"; farneback needs no GPU/extra weights
    fb_consistency_thresh: float = 1.5  # px; forward-backward disagreement => occluded
    long_term: bool = True  # also measure each frame vs the first frame


class EvalConfig(msgspec.Struct, frozen=True):
    reference_metrics: tuple[str, ...] = ("psnr", "ssim", "lpips", "dists")
    noreference_metrics: tuple[str, ...] = ("musiq", "clipiqa", "niqe")  # + DOVER (separate model)
    temporal: TemporalConfig = msgspec.field(default_factory=TemporalConfig)


DEFAULT_PRICING = FalPricing()

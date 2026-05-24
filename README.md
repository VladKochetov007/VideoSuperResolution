# Video Super Resolution

Neural video upscaling for `720p -> 1080p / 2K`.

Main idea:

- videos go into one queue
- frames or temporal windows mix into shared GPU batches
- GPU stays busy instead of waiting on one video
- color post-processing can run after upscale

Project has two main self-host paths:

- `SeedVR2`: better quality on faces, hair, grass, motion
- `Real-ESRGAN`: faster and easier to scale

`Topaz` was tested because it is strong on fal, but it is closed source. Not main focus here.

Most interesting part is self-host scaling:

- batch queue for many videos
- VRAM-based batch calibration
- spatial tiling for small GPUs
- temporal windowing for diffusion models
- optional color correction: `wavelet`, `histogram`, `reinhard`, `adain`

Models were chosen from fal comparison:

https://blog.fal.ai/comparing-the-best-ai-upscalers-for-video-and-images/

## Quick start

```bash
uv venv --python 3.12
uv pip install -e '.[metrics,fal,dev]'
cp .env.example .env
```

Run UI:

```bash
streamlit run scripts/webui.py
```

Run self-host batch queue:

```bash
python scripts/serve_batch.py --help
```

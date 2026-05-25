"""Streamlit front-end for the VSR pipeline.

    .venv/bin/streamlit run scripts/webui.py

Two modes:
- Single video: one clip through a fal-ai model + color post-processing. Self-hosted models are
  not offered here: a single clip cannot be batched, so the GPU sits idle between forwards. Run
  self-hosted models through the batch queue instead, where frames are packed across videos.
- Batch queue: many clips drained through ONE GPU-saturating self-hosted run. Batch size is
  calibrated to ~90% of VRAM, so frames from different videos share each forward (cross-video
  batching). Uploading more clips just lengthens the queue.
"""

import tempfile
from collections import defaultdict
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import streamlit as st

from video_super_resolution.local import run_batch_queue
from video_super_resolution.postprocess import ColorConfig
from video_super_resolution.providers import PROVIDERS
from video_super_resolution.webui import estimate_cost, run_upscale

COLOR_METHODS = ["none", "wavelet", "reinhard", "adain", "histogram"]
SCALES = {"1.5x": 1.5, "2x": 2.0}

st.set_page_config(page_title="QLAN VSR", layout="wide")
st.title("720p -> 1080p/2K neural upscaling")


def _progress_handlers():
    bar = st.progress(0.0)
    status = st.empty()

    def cb(stage: str, frac: float | None, msg: str) -> None:
        if frac is not None:
            bar.progress(min(1.0, max(0.0, frac)))
        status.text(f"{stage}: {msg}")

    return bar, status, cb


def _build_batch_jobs(uploads, work: Path, out_h: int) -> list[dict]:
    name_counts: dict[str, int] = defaultdict(int)
    stem_counts: dict[str, int] = defaultdict(int)
    jobs = []
    for u in uploads:
        stem = Path(u.name).stem
        suffix = Path(u.name).suffix
        name_counts[u.name] += 1
        stem_counts[stem] += 1

        input_name = u.name if name_counts[u.name] == 1 else f"{stem}_{name_counts[u.name]}{suffix}"
        job_name = stem if stem_counts[stem] == 1 else f"{stem}_{stem_counts[stem]}"

        p = work / input_name
        p.write_bytes(u.getbuffer())
        jobs.append({"input": str(p), "out_h": out_h, "name": job_name})
    return jobs


def single_video_ui() -> None:
    with st.sidebar:
        model = st.selectbox("Model", list(PROVIDERS), index=0)
        scale_label = st.selectbox("Scale", list(SCALES), index=0)
        color_method = st.selectbox("Color correction", COLOR_METHODS, index=1)
        strength = st.slider("Correction strength", 0.0, 1.0, 1.0, 0.05,
                             disabled=color_method == "none")
        wavelet_levels = st.slider("Wavelet levels", 1, 7, 5, disabled=color_method != "wavelet")
        allow_spend = st.checkbox(f"Authorize fal spend for `{model}`", value=False)
        st.warning("Real fal model: this spends credits.")

    uploaded = st.file_uploader("Upload a 720p clip", type=["mp4", "mov", "mkv", "webm"])
    if uploaded is None:
        return
    work = Path(tempfile.mkdtemp(prefix="video_sr_webui_"))
    src = work / uploaded.name
    src.write_bytes(uploaded.getbuffer())
    out_h = int(720 * SCALES[scale_label])
    out_h += out_h % 2
    try:
        est = estimate_cost(model, src, out_h)
        st.info(f"Target height {out_h}px · estimated cost ${est:.3f} (fal)")
    except Exception as exc:  # noqa: BLE001 - bad upload, surface it
        st.error(f"Could not read the clip: {exc}")
        return

    if not st.button("Run", type="primary"):
        return
    color = ColorConfig(method=color_method, strength=strength, wavelet_levels=wavelet_levels)
    bar, status, cb = _progress_handlers()
    try:
        res = run_upscale(src, model, out_h, color, work, allow_spend=allow_spend, progress=cb)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    bar.progress(1.0)
    status.empty()

    c1, c2, c3 = st.columns(3)
    c1.metric("Output", f"{res.out_dims[0]}x{res.out_dims[1]}")
    c2.metric("Frames", res.frames)
    c3.metric("Cost", f"${res.cost_usd:.3f}")
    if res.delta_e_raw is not None:
        d1, d2 = st.columns(2)
        d1.metric("Color drift ΔE (raw)", f"{res.delta_e_raw:.2f}")
        d2.metric(f"ΔE after {res.color_method}", f"{res.delta_e_corrected:.2f}",
                  delta=f"{res.delta_e_corrected - res.delta_e_raw:+.2f}", delta_color="inverse")
    v1, v2 = st.columns(2)
    v1.subheader("Input")
    v1.video(str(src))
    v2.subheader("Upscaled" + (f" + {res.color_method}" if res.corrected_path else ""))
    v2.video(res.display_path)
    st.download_button("Download result (H.264 + audio)", Path(res.display_path).read_bytes(),
                       file_name=Path(res.display_path).name, mime="video/mp4")


def batch_queue_ui() -> None:
    st.caption("Real-ESRGAN: clips drain through one run, batch calibrated to ~90% VRAM (frames mixed "
               "across videos). SeedVR2: GGUF + block-swap fit the diffusion model on a small GPU "
               "(low VRAM, swap-bound). Config below.")
    with st.sidebar:
        model = st.selectbox("Model", ["realesrgan", "seedvr2"], index=0)
        scale_label = st.selectbox("Scale", list(SCALES), index=0)
        st.divider()
        st.subheader("Color post-processing")
        color = st.selectbox("Method", COLOR_METHODS, index=0)
        color_strength = st.slider("Strength", 0.0, 1.0, 1.0, 0.05, disabled=color == "none")
        wavelet_levels = st.slider("Wavelet levels", 1, 7, 5, disabled=color != "wavelet")
        st.divider()
        st.subheader("GPU / batching")
        headroom = st.slider("VRAM headroom (kept free)", 0.05, 0.50, 0.10, 0.05)
        max_batch = st.number_input("Max batch (0 = VRAM-bound)", 0, 4096, 0)
        fp32 = st.checkbox("fp32 (disable fp16)", value=False)
        sv = {}
        if model == "realesrgan":
            st.caption("Use tile=256 or 512 on small GPUs; use tile=0 on the 140 GB GPU for max throughput.")
            sv = {
                "tile": st.slider("Tile size (px, 0 = whole frame)", 0, 1024, 0, 32),
                "tile_pad": st.slider("Tile padding (px)", 0, 64, 10),
            }
        else:
            st.caption("GGUF + block-swap fit the 3B model on a small GPU. Fewer swapped blocks and a "
                       "larger VAE tile use more VRAM and run faster (less CPU<->GPU copying); on the "
                       "140 GB GPU set blocks_to_swap=0 and vae_tile=0.")
            sv = {
                "variant": st.selectbox("Variant", ["3B", "7B"], index=0),
                "temporal_batch": st.slider("Temporal window (4n+1)", 1, 25, 5,
                                            help="Larger = better temporal coherence, more VRAM"),
                "blocks_to_swap": st.slider("Blocks to swap to CPU (0 = keep all on GPU)", 0, 36, 32,
                                            help="Lower = more VRAM used, faster. 6 GB needs ~32"),
                "vae_tile": st.slider("VAE tile (px, 0 = whole frame)", 0, 1024, 256, 64,
                                      help="Larger = more VRAM, fewer tiles, faster"),
                "attention_mode": st.selectbox(
                    "Attention", ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                    index=0, help="flash/sage need the kernel installed on the host; else falls back to sdpa"),
            }
    out_h = int(720 * SCALES[scale_label])
    out_h += out_h % 2

    uploads = st.file_uploader("Upload clips (add more to grow the queue)",
                               type=["mp4", "mov", "mkv", "webm"], accept_multiple_files=True)
    if not uploads:
        return
    st.write(f"Queue: {len(uploads)} video(s) -> {model}, target {out_h}px"
             + (f", color={color}" if color != "none" else ""))
    for u in uploads:
        st.write(f"- {u.name} ({u.size / 1e6:.1f} MB)")

    if not st.button(f"Process queue ({len(uploads)})", type="primary"):
        return
    work = Path(tempfile.mkdtemp(prefix="video_sr_queue_"))
    jobs = _build_batch_jobs(uploads, work, out_h)

    bar, status, cb = _progress_handlers()
    try:
        finished = run_batch_queue(jobs, work / "out", progress=cb, model=model, color=color,
                                   color_strength=color_strength, wavelet_levels=wavelet_levels,
                                   headroom=headroom, max_batch=int(max_batch), fp32=fp32, **sv)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    bar.progress(1.0)
    status.empty()
    st.success(f"Done: {len(finished)} video(s)")
    for v in finished:
        st.subheader(f"{v['video_id']} ({v['frames']} frames)")
        st.video(v["out_path"])
        st.download_button(f"Download {v['video_id']}", Path(v["out_path"]).read_bytes(),
                           file_name=Path(v["out_path"]).name, mime="video/mp4", key=v["video_id"])


mode = st.sidebar.radio("Mode", ["Single video", "Batch queue (self-hosted)"])
st.sidebar.divider()
if mode == "Single video":
    single_video_ui()
else:
    batch_queue_ui()

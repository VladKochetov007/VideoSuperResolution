"""Queue / batching / capacity tests. CPU-only via the bicubic reference model; the real GPU path
is smoke-tested through the bridge when CUDA + the on-prem venv are present."""

import subprocess
from pathlib import Path

import pytest

import numpy as np

from video_super_resolution.postprocess import ColorConfig, VideoColorPostProcessor, make_corrector
from video_super_resolution.providers.base import probe
from video_super_resolution.serving import BicubicBatchUpscaler, CapacityModel, Scheduler, VideoFrameSource
from video_super_resolution.serving.realesrgan_batch import RealESRGANBatch
from video_super_resolution.serving.seedvr2_batch import SeedVR2Batch
from video_super_resolution.serving.unit import WorkUnit


def _clip(path: Path, n: int, size: str = "32x24") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={size}:rate=8:duration=1", "-frames:v", str(n), "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


# ---------------- capacity sizing ----------------
def test_capacity_scales_with_vram_and_cost():
    small = CapacityModel({"m": (1e6, 1e8)}, vram_bytes=2e9)
    big = CapacityModel({"m": (1e6, 1e8)}, vram_bytes=140e9)
    assert big.batch_size("m") > small.batch_size("m") > 1
    # heavier per-frame activations -> smaller batch
    heavy = CapacityModel({"m": (1e9, 1e8)}, vram_bytes=140e9)
    assert heavy.batch_size("m") < big.batch_size("m")


def test_capacity_headroom_lowers_batch():
    lo = CapacityModel({"m": (1e6, 0.0)}, vram_bytes=10e9, headroom=0.10)
    hi = CapacityModel({"m": (1e6, 0.0)}, vram_bytes=10e9, headroom=0.50)
    assert hi.batch_size("m") < lo.batch_size("m")


def test_capacity_never_below_one():
    assert CapacityModel({"m": (1e12, 1e12)}, vram_bytes=1e9).batch_size("m") == 1


def test_capacity_max_batch_cap():
    uncapped = CapacityModel({"m": (1e3, 0.0)}, vram_bytes=140e9)
    capped = CapacityModel({"m": (1e3, 0.0)}, vram_bytes=140e9, max_batch=8)
    assert uncapped.batch_size("m") > 8
    assert capped.batch_size("m") == 8


# ---------------- RealESRGAN spatial tiling ----------------
def test_realesrgan_tiled_reassembly_matches_reference():
    model = RealESRGANBatch.__new__(RealESRGANBatch)
    model.tile = 5
    model.tile_pad = 1

    def fake_tiles(tiles):
        return [np.repeat(np.repeat(t, 2, axis=0), 2, axis=1) for t in tiles]

    model._infer_tile_batch = fake_tiles
    frame = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    out = model._infer_tiled([frame], frame.shape[:2])[0]
    ref = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
    assert out.shape == (14, 22, 3)
    assert np.array_equal(out, ref)


# ---------------- color post-processing in the queue ----------------
def test_color_postprocess_runs_in_queue(tmp_path):
    a = _clip(tmp_path / "a.mp4", 3)
    cap = CapacityModel({"fake": (1.0, 0.0)}, vram_bytes=100.0, headroom=0.0)
    post = VideoColorPostProcessor(make_corrector(ColorConfig(method="reinhard")))
    sched = Scheduler({"fake": BicubicBatchUpscaler()}, cap, VideoFrameSource(), postprocess=post)
    sched.enqueue_video("a", a, "fake", 0, tmp_path / "a_out.mp4")
    sched.run_until_drained()
    _w, _h, _fps, n = probe(tmp_path / "a_out.mp4")
    assert (tmp_path / "a_out.mp4").exists() and n == 3


# ---------------- SeedVR2 batched (backend injection) ----------------
def test_seedvr2_batch_uses_injected_backend(monkeypatch):
    monkeypatch.setattr(SeedVR2Batch, "_preflight_vram", lambda self: None)
    monkeypatch.setattr(SeedVR2Batch, "_resolve_weights", lambda self, d, c: None)
    m = SeedVR2Batch(backend=lambda units, w: [u.payload for u in units])
    u = WorkUnit(video_id="v", model="seedvr2", index=0, payload=np.ones((2, 4, 4, 3), np.uint8))
    assert len(m.batch_infer([u])) == 1


def test_seedvr2_batch_requires_backend(monkeypatch):
    monkeypatch.setattr(SeedVR2Batch, "_preflight_vram", lambda self: None)
    monkeypatch.setattr(SeedVR2Batch, "_resolve_weights", lambda self, d, c: None)
    with pytest.raises(NotImplementedError, match="backend"):
        SeedVR2Batch().batch_infer([])


# ---------------- cross-video batching ----------------
def test_scheduler_drains_two_videos_in_one_batch(tmp_path):
    a = _clip(tmp_path / "a.mp4", 3)
    b = _clip(tmp_path / "b.mp4", 2)
    cap = CapacityModel({"fake": (1.0, 0.0)}, vram_bytes=100.0, headroom=0.0)  # B_max=100
    sched = Scheduler({"fake": BicubicBatchUpscaler()}, cap, VideoFrameSource())
    sched.enqueue_video("a", a, "fake", 0, tmp_path / "a_out.mp4")
    sched.enqueue_video("b", b, "fake", 0, tmp_path / "b_out.mp4")

    finished = sched.run_step()  # one forward should consume all 5 units, both videos
    assert {r.video_id for r in finished} == {"a", "b"}
    assert sched.queue.is_empty()
    w, h, _fps, n = probe(tmp_path / "a_out.mp4")
    assert (w, h) == (64, 48) and n == 3  # bicubic x2 of 32x24


def test_scheduler_respects_batch_limit(tmp_path):
    a = _clip(tmp_path / "a.mp4", 4)
    cap = CapacityModel({"fake": (1.0, 0.0)}, vram_bytes=2.0, headroom=0.0)  # B_max=2
    sched = Scheduler({"fake": BicubicBatchUpscaler()}, cap, VideoFrameSource())
    sched.enqueue_video("a", a, "fake", 0, tmp_path / "a_out.mp4")
    assert cap.batch_size("fake") == 2
    sched.run_step()
    assert sched.queue.pending("fake") == 2  # 4 frames, 2 per batch -> 2 left after one step


# ---------------- real GPU path (skipped without hardware) ----------------
def _cuda_onprem() -> bool:
    onprem = Path(__file__).resolve().parents[1] / ".venv-onprem" / "bin" / "python"
    if not onprem.exists():
        return False
    r = subprocess.run([str(onprem), "-c", "import torch;print(torch.cuda.is_available())"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "True"


def test_batch_queue_gpu_smoke(tmp_path):
    if not _cuda_onprem():
        pytest.skip("no CUDA on-prem venv")
    from video_super_resolution.local import run_batch_queue

    a = _clip(tmp_path / "a.mp4", 4, size="96x64")
    b = _clip(tmp_path / "b.mp4", 4, size="96x64")
    jobs = [{"input": str(a), "out_h": 128, "name": "a"}, {"input": str(b), "out_h": 128, "name": "b"}]
    batches = []
    finished = run_batch_queue(jobs, tmp_path / "out",
                               progress=lambda s, f, m: batches.append(m) if s == "stage" else None)
    assert {v["video_id"] for v in finished} == {"a", "b"}
    assert any("batch size=" in m and "size=1 " not in m for m in batches)  # batched >1 at this size

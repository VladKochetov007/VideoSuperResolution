"""Webui core tests. Offline only (mock model), no fal spend, no GPU."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import webui
from video_super_resolution.media import has_audio
from video_super_resolution.postprocess import ColorConfig
from video_super_resolution.providers.base import probe
from video_super_resolution.webui import WEBUI_MODELS, estimate_cost, run_upscale


def _vcodec(path):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def clip_720ish(tmp_path_factory):
    """Small clip WITH an audio track, to verify audio is carried through to the output."""
    p = tmp_path_factory.mktemp("vid") / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=240x160:rate=4:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-frames:v", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)],
        check=True,
    )
    return p


def test_registry_has_mock_and_real():
    assert "mock" in WEBUI_MODELS
    assert {"topaz", "seedvr2", "realesrgan"} <= set(WEBUI_MODELS)


def test_mock_is_free():
    assert WEBUI_MODELS["mock"]().estimate_cost(1280, 720, 100, 24.0, 1080) == 0.0


def test_mock_upscales_to_target_height(clip_720ish, tmp_path):
    res = run_upscale(clip_720ish, "mock", 320, ColorConfig(method="none"), tmp_path)
    assert res.cost_usd == 0.0
    assert res.out_dims[1] == 320
    assert res.in_dims == (240, 160)
    assert Path(res.out_path).exists()
    _w, h, _fps, n = probe(Path(res.out_path))
    assert h == 320 and n >= 2
    assert res.corrected_path is None and res.delta_e_raw is None


def test_output_is_h264_with_audio(clip_720ish, tmp_path):
    res = run_upscale(clip_720ish, "mock", 320, ColorConfig(method="none"), tmp_path)
    disp = Path(res.display_path)
    assert disp.exists()
    assert _vcodec(disp) == "h264"  # web-playable, not OpenCV's mp4v
    assert has_audio(disp)          # source audio carried through


def test_corrected_output_keeps_audio(clip_720ish, tmp_path):
    res = run_upscale(clip_720ish, "mock", 320, ColorConfig(method="wavelet"), tmp_path)
    assert Path(res.display_path) == Path(res.corrected_path)
    assert _vcodec(Path(res.display_path)) == "h264"
    assert has_audio(Path(res.display_path))


@pytest.mark.parametrize("method", ["wavelet", "reinhard", "adain", "histogram"])
def test_color_correction_measures_and_writes(clip_720ish, tmp_path, method):
    res = run_upscale(clip_720ish, "mock", 320, ColorConfig(method=method), tmp_path)
    assert res.corrected_path is not None and Path(res.corrected_path).exists()
    assert res.delta_e_raw is not None and res.delta_e_corrected is not None
    assert res.color_method == method


def test_real_model_refuses_without_spend_authorization(clip_720ish, tmp_path):
    with pytest.raises(RuntimeError, match="allow_spend"):
        run_upscale(clip_720ish, "topaz", 320, ColorConfig(method="none"), tmp_path, allow_spend=False)


def test_unknown_model_raises(clip_720ish, tmp_path):
    with pytest.raises(ValueError, match="unknown model"):
        run_upscale(clip_720ish, "nope", 320, ColorConfig(method="none"), tmp_path)


def test_estimate_cost_real_model_positive(clip_720ish):
    assert estimate_cost("realesrgan", clip_720ish, 320) > 0.0


def test_build_batch_jobs_dedupes_duplicate_upload_names(tmp_path):
    class Upload:
        def __init__(self, name: str, payload: bytes):
            self.name = name
            self._payload = payload

        def getbuffer(self):
            return self._payload

    uploads = [Upload("same.mp4", b"one"), Upload("same.mp4", b"two")]
    jobs = webui._build_batch_jobs(uploads, tmp_path, 1080)
    assert [j["name"] for j in jobs] == ["same", "same_2"]
    assert Path(jobs[0]["input"]).name == "same.mp4"
    assert Path(jobs[1]["input"]).name == "same_2.mp4"
    assert Path(jobs[0]["input"]).read_bytes() == b"one"
    assert Path(jobs[1]["input"]).read_bytes() == b"two"

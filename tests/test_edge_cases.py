"""Edge-condition and resilience tests. No fal spend, no GPU required.

Covers the real failure surfaces: bad/missing auth, corrupt or degenerate input, fal outage
mid-batch, truncated downloads, retry exhaustion, cost-guard refusal, output verification.
"""

import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from video_super_resolution.cost import realesrgan_cost_usd, seedvr2_cost_usd, topaz_cost_usd
from video_super_resolution.metrics.temporal import warping_error
from video_super_resolution.postprocess import ColorConfig, FrameColorCorrector, color_drift, make_corrector
from video_super_resolution.providers import _fal
from video_super_resolution.providers.base import UpscaleResult, probe


@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=96x64:rate=2:duration=1", "-frames:v", "2", "-pix_fmt", "yuv420p", str(p)],
        check=True,
    )
    return p


# ---------------- cost model ----------------
def test_topaz_cost_tiers_and_60fps():
    assert topaz_cost_usd(10, 720, 30) < topaz_cost_usd(10, 1080, 30) < topaz_cost_usd(10, 1440, 30)
    assert topaz_cost_usd(10, 1080, 60) == pytest.approx(2 * topaz_cost_usd(10, 1080, 30))


def test_megapixel_costs_monotonic():
    assert seedvr2_cost_usd(1920, 1080, 10) < seedvr2_cost_usd(1920, 1080, 20)
    assert realesrgan_cost_usd(1280, 720, 10) < realesrgan_cost_usd(2560, 1440, 10)


# ---------------- auth ----------------
def test_ensure_fal_key_raises(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FALAI_TOKEN", raising=False)
    monkeypatch.setattr(_fal, "find_dotenv", lambda *a, **k: "")
    monkeypatch.setattr(_fal, "load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="FAL_KEY"):
        _fal.ensure_fal_key()


# ---------------- corrupt / degenerate input ----------------
def test_probe_corrupt_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(Exception):
        probe(bad)


def test_probe_ok(tiny_mp4):
    w, h, _fps, n = probe(tiny_mp4)
    assert (w, h) == (96, 64)
    assert n >= 2


def test_warping_error_needs_two_frames():
    with pytest.raises(ValueError):
        warping_error([np.zeros((16, 16, 3), np.uint8)])


# ---------------- retry / download integrity ----------------
def test_retry_succeeds_second_attempt(monkeypatch):
    monkeypatch.setattr(_fal.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise OSError("transient")
        return "ok"

    assert _fal._retry(flaky, "x") == "ok"
    assert calls["n"] == 2


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(_fal.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="after"):
        _fal._retry(lambda: (_ for _ in ()).throw(OSError("nope")), "x")


def test_download_rejects_truncated(monkeypatch, tmp_path):
    monkeypatch.setattr(_fal.time, "sleep", lambda *_: None)

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(_fal.urllib.request, "urlopen", lambda *a, **k: FakeResp(b"tiny"))
    with pytest.raises(RuntimeError):
        _fal.download("http://x/y.mp4", tmp_path / "o.mp4")


# ---------------- batch resilience (execute_job) ----------------
def _run_eval():
    import run_eval

    return run_eval


class _RaisingProv:
    name = "boom"

    def run(self, *a, **k):
        raise RuntimeError("fal 500")


class _MissingOutProv:
    name = "ghost"

    def run(self, inp, out, gt_h):
        return UpscaleResult("ghost", str(out), 0.0)  # returns but never writes the file


class _GoodProv:
    name = "good"

    def __init__(self, src):
        self.src = src

    def run(self, inp, out, gt_h):
        shutil.copy(self.src, out)
        return UpscaleResult("good", str(out), 0.01)


def test_execute_job_propagates_provider_failure(tmp_path):
    with pytest.raises(Exception):
        _run_eval().execute_job(_RaisingProv(), tmp_path / "in.mp4", tmp_path / "out.mp4", 64)


def test_execute_job_detects_missing_output(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        _run_eval().execute_job(_MissingOutProv(), tmp_path / "in.mp4", tmp_path / "out.mp4", 64)


def test_execute_job_accepts_good_output(tiny_mp4, tmp_path):
    res = _run_eval().execute_job(_GoodProv(tiny_mp4), tiny_mp4, tmp_path / "out.mp4", 64)
    assert res.cost_usd == 0.01
    assert (tmp_path / "out.mp4").exists()


def test_execute_job_rejects_wrong_height(tiny_mp4, tmp_path):
    with pytest.raises(RuntimeError, match="height"):
        _run_eval().execute_job(_GoodProv(tiny_mp4), tiny_mp4, tmp_path / "out.mp4", 1080)


# ---------------- color correction ----------------
@pytest.mark.parametrize("method", ["wavelet", "reinhard", "adain", "histogram"])
def test_color_correction_reduces_drift(method):
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    shifted = np.clip(ref.astype(int) + 35, 0, 255).astype(np.uint8)  # warm hue/brightness shift
    corr = FrameColorCorrector(ColorConfig(method=method))
    fixed = corr(shifted, ref)
    assert fixed.shape == shifted.shape and fixed.dtype == np.uint8
    assert color_drift(fixed, ref) < color_drift(shifted, ref)


def test_color_corrector_resizes_ref_to_output():
    rng = np.random.default_rng(1)
    out = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    ref = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)  # half-res source, as in real upscaling
    fixed = FrameColorCorrector(ColorConfig(method="wavelet"))(out, ref)
    assert fixed.shape == out.shape


def test_color_strength_zero_is_passthrough():
    rng = np.random.default_rng(2)
    out = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    ref = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    fixed = FrameColorCorrector(ColorConfig(method="reinhard", strength=0.0))(out, ref)
    assert np.array_equal(fixed, out)


def test_color_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown color method"):
        FrameColorCorrector(ColorConfig(method="bogus"))


def test_make_corrector_none_is_none():
    assert make_corrector(ColorConfig(method="none")) is None
    assert make_corrector(ColorConfig(method="wavelet")) is not None


# ---------------- budget guard ----------------
def test_budget_guard_refuses():
    if not (ROOT / "data" / "eval" / "jobs.json").exists():
        pytest.skip("no jobs.json")
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_eval.py"),
         "--dry-run", "--budget", "0.0001"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "REFUSING" in (r.stdout + r.stderr)

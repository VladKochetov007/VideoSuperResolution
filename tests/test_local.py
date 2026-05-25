"""Self-hosted provider tests. Mostly offline; GPU tests skip when CUDA / the on-prem venv is absent."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import serve_batch
from video_super_resolution.local import (
    LOCAL_PROVIDERS,
    LocalSeedVR2Upscaler,
    ensure_realesrgan_weights,
    realesrgan_is_cached,
    seedvr2_is_cached,
)
from video_super_resolution.local import provider as local_provider
from video_super_resolution.local.provider import _stream
from video_super_resolution.webui import WEBUI_MODELS


def _cuda_onprem() -> bool:
    try:
        onprem_py = local_provider.onprem_python()
    except RuntimeError:
        return False
    r = subprocess.run([str(onprem_py), "-c", "import torch;print(torch.cuda.is_available())"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "True"


# ---------------- registry / cost ----------------
def test_local_providers_registered():
    assert set(LOCAL_PROVIDERS) == {"realesrgan-local", "seedvr2-local"}
    assert {"realesrgan-local", "seedvr2-local"} <= set(WEBUI_MODELS)


def test_local_models_are_free():
    for cls in LOCAL_PROVIDERS.values():
        assert cls().estimate_cost(1280, 720, 100, 24.0, 1080) == 0.0


# ---------------- weight cache ----------------
def test_realesrgan_weights_cached_returns_path():
    if not realesrgan_is_cached("RealESRGAN_x2plus"):
        pytest.skip("x2plus weights not cached")
    p = ensure_realesrgan_weights("RealESRGAN_x2plus")  # must not download when present
    assert p.exists() and p.stat().st_size > 1_000_000


def test_unknown_weight_name_raises():
    with pytest.raises(ValueError, match="unknown Real-ESRGAN weight"):
        ensure_realesrgan_weights("NotAModel")


def test_seedvr2_not_cached_by_default():
    assert seedvr2_is_cached("3B") is False


def test_realesrgan_cache_falls_back_to_legacy_dir(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    weight = legacy / "RealESRGAN_x2plus.pth"
    weight.write_bytes(b"x" * 1_000_001)
    monkeypatch.setattr(local_provider, "ROOT", tmp_path / "repo")
    from video_super_resolution.local import weights as local_weights
    monkeypatch.setattr(local_weights, "DEFAULT_CACHE", tmp_path / "repo" / "weights")
    monkeypatch.setattr(local_weights, "LEGACY_CACHE", legacy)
    assert local_weights.realesrgan_is_cached("RealESRGAN_x2plus")
    assert local_weights.ensure_realesrgan_weights("RealESRGAN_x2plus") == weight


# ---------------- stdout progress protocol ----------------
def test_stream_parses_progress_and_done():
    seen = []
    prog = [sys.executable, "-c",
            "print('STAGE loading');print('PROGRESS 1 2');print('PROGRESS 2 2');"
            "print('DONE 1920 1080 2 3.5')"]
    done = _stream(prog, lambda s, f, m: seen.append((s, f, m)))
    assert done == {"out_w": 1920, "out_h": 1080, "frames": 2, "seconds": 3.5}
    assert ("stage", None, "loading") in seen
    assert ("infer", 1.0, "frame 2/2") in seen


def test_stream_raises_on_worker_error():
    bad = [sys.executable, "-c", "import sys;print('ERROR boom');sys.exit(2)"]
    with pytest.raises(RuntimeError, match="boom"):
        _stream(bad, None)


def test_batch_queue_passes_realesrgan_tile_args(tmp_path, monkeypatch):
    seen = {}

    class FakeProc:
        returncode = 0
        stdout = iter(["VIDEO a 1 /tmp/a.mp4\n", "DONE\n"])

        def wait(self):
            return None

    def fake_popen(cmd, **_kwargs):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(local_provider, "_onprem_candidates", lambda: [Path(sys.executable)])
    monkeypatch.setattr(local_provider.subprocess, "Popen", fake_popen)
    local_provider.run_batch_queue(
        [{"input": "in.mp4", "out_h": 1080, "name": "a"}],
        tmp_path / "out",
        tile=256,
        tile_pad=12,
    )
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tile") + 1] == "256"
    assert cmd[cmd.index("--tile-pad") + 1] == "12"


def test_onprem_python_uses_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setenv("ONPREM_PYTHON", str(fake))
    assert local_provider.onprem_python() == fake


def test_unique_jobs_dedupes_duplicate_names():
    jobs = [{"name": "clip", "input": "a.mp4"}, {"name": "clip", "input": "b.mp4"}]
    uniq = serve_batch._unique_jobs(jobs)
    assert [j["name"] for j in uniq] == ["clip", "clip_2"]


def test_seedvr2_variant_maps_to_gguf():
    assert local_provider._SEEDVR2_GGUF["3B"].endswith("3b-Q4_K_M.gguf")
    assert local_provider._SEEDVR2_GGUF["7B"].endswith("7b-Q4_K_M.gguf")


def test_resolve_model_dir_prefers_populated(tmp_path, monkeypatch):
    from video_super_resolution.serving import seedvr2_comfy as sv

    monkeypatch.delenv("SEEDVR2_MODEL_DIR", raising=False)
    empty, populated = tmp_path / "empty", tmp_path / "have"
    empty.mkdir()
    populated.mkdir()
    (populated / "ema_vae_fp16.safetensors").write_bytes(b"x")
    monkeypatch.setattr(sv, "_MODEL_DIR_CANDIDATES", [empty, populated])
    assert sv.resolve_model_dir() == populated
    assert sv.resolve_model_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_resolve_repo_dir_errors_when_missing(tmp_path, monkeypatch):
    from video_super_resolution.serving import seedvr2_comfy as sv

    monkeypatch.delenv("SEEDVR2_COMFY_REPO", raising=False)
    monkeypatch.setattr(sv, "_REPO_CANDIDATES", [tmp_path / "nope"])
    with pytest.raises(RuntimeError, match="ComfyUI-SeedVR2"):
        sv.resolve_repo_dir()


# ---------------- GPU / on-prem (skipped without hardware) ----------------
def test_seedvr2_vram_guard_fails_fast():
    if not _cuda_onprem():
        pytest.skip("no CUDA on-prem venv")
    with pytest.raises(RuntimeError, match="VRAM"):
        LocalSeedVR2Upscaler(variant="3B", download=False).run(
            ROOT / "nonexistent.mp4", ROOT / "out.mp4", 1080)


def test_realesrgan_local_gpu_smoke(tmp_path):
    if not _cuda_onprem():
        pytest.skip("no CUDA on-prem venv")
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=4:duration=0.5", "-frames:v", "2", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    from video_super_resolution.local import LocalRealEsrganUpscaler

    res = LocalRealEsrganUpscaler(tile=128, half=True).run(src, tmp_path / "out.mp4", 480)
    assert res.cost_usd == 0.0 and Path(res.out_path).exists()

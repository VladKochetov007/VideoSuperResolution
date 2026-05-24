import json
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

import msgspec


class UpscaleResult(msgspec.Struct, frozen=True):
    model: str
    out_path: str
    cost_usd: float
    source_url: str = ""
    out_url: str = ""


@runtime_checkable
class Upscaler(Protocol):
    name: str

    def estimate_cost(self, in_w: int, in_h: int, frames: int, fps: float, out_h: int) -> float: ...

    def run(self, input_path: Path, out_path: Path, out_h: int) -> UpscaleResult: ...


def probe(path: Path) -> tuple[int, int, float, int]:
    """(width, height, fps, frame_count) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    frames = int(s.get("nb_read_frames") or 0)
    return int(s["width"]), int(s["height"]), float(num) / float(den), frames

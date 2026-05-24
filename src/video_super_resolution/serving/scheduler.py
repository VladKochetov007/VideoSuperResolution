from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import msgspec
import numpy as np

from .assembler import Assembler
from .capacity import CapacityModel
from .queue import WorkQueue
from .source import FrameSource


@runtime_checkable
class FramePostProcessor(Protocol):
    # Transforms assembled output frames before encode, given the source clip for reference
    # (color anchor, etc.). Identity if None. Injected, not hard-coded — see postprocess module.
    def __call__(self, frames: list[np.ndarray], source_path: Path) -> list[np.ndarray]: ...


class JobResult(msgspec.Struct):
    video_id: str
    out_path: str
    frames: int


class Scheduler:
    # Orchestration only (no upscaling, no VRAM math). enqueue_video loads units into the shared
    # queue and registers a per-video Assembler (no GPU). run_step runs ONE batched forward for one
    # model, routes outputs to assemblers, and muxes any finished video. Backfill is emergent: a
    # single drain pulls the next B_max units across all videos in the model's FIFO.
    def __init__(self, models: dict[str, object], capacity: CapacityModel, source: FrameSource,
                 verbose: bool = False, postprocess: "FramePostProcessor | None" = None):
        self.models = models
        self.capacity = capacity
        self.source = source
        self.verbose = verbose
        self.postprocess = postprocess
        self.queue = WorkQueue()
        self.assemblers: dict[str, Assembler] = {}
        self.meta: dict[str, dict] = {}

    def enqueue_video(self, video_id: str, path, model: str, out_h: int, out_path) -> None:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        n, overlap = 0, 0
        for u in self.source.units(video_id, Path(path), model, out_h):
            self.queue.enqueue([u])
            n += 1
            overlap = u.overlap
        self.assemblers[video_id] = Assembler(video_id, n, overlap, windowed=model == "seedvr2")
        self.meta[video_id] = {"fps": fps, "out_path": Path(out_path), "model": model,
                               "source": Path(path), "target_h": out_h}

    def run_step(self) -> list[JobResult]:
        models = self.queue.models_with_work()
        if not models:
            return []
        model = models[0]
        units = self.queue.drain(model, self.capacity.batch_size(model))
        outs = self.models[model].batch_infer(units)
        if self.verbose:
            vids = sorted({u.video_id for u in units})
            print(f"  batch model={model} size={len(units)} videos={vids}")
        touched = set()
        for u, out in zip(units, outs):
            self.assemblers[u.video_id].put(u.index, out)
            touched.add(u.video_id)
        finished = []
        for vid in touched:
            if self.assemblers[vid].done():
                finished.append(self._mux(vid))
        return finished

    def run_until_drained(self) -> list[JobResult]:
        results = []
        while not self.queue.is_empty():
            results.extend(self.run_step())
        return results

    def _mux(self, video_id: str) -> JobResult:
        from ..media import FfmpegWriter

        frames = self.assemblers.pop(video_id).emit()
        meta = self.meta.pop(video_id)
        target_h = meta.get("target_h") or 0
        if target_h and frames and frames[0].shape[0] != target_h:
            tw = round(frames[0].shape[1] * target_h / frames[0].shape[0])
            tw += tw % 2
            frames = [cv2.resize(f, (tw, target_h), interpolation=cv2.INTER_CUBIC) for f in frames]
        if self.postprocess is not None:
            frames = self.postprocess(frames, meta["source"])
        with FfmpegWriter(meta["out_path"], meta["fps"], audio_source=meta["source"]) as writer:
            for f in frames:
                writer.write(f)
        return JobResult(video_id=video_id, out_path=str(meta["out_path"]), frames=len(frames))

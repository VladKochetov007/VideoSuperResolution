from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .unit import WorkUnit


class RealESRGANBatch:
    # Self-hosted Real-ESRGAN x2 as a true batched forward (RRDBNet on a (B,3,H,W) tensor), unlike
    # RealESRGANer which is one image at a time. Frames of differing resolution (mixed videos in one
    # batch) are grouped by shape into sub-forwards. CUDA OOM is caught and the batch is halved.
    name = "realesrgan"

    def __init__(self, weights: Path, device: str | None = None, half: bool = True,
                 tile: int = 0, tile_pad: int = 10):
        from basicsr.archs.rrdbnet_arch import RRDBNet

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device == "cuda"
        self.tile = max(0, int(tile))
        self.tile_pad = max(0, int(tile_pad))
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = state.get("params_ema", state.get("params", state))
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        if self.half:
            model.half()
        self.model = model

    @torch.no_grad()
    def batch_infer(self, units: list[WorkUnit]) -> list[np.ndarray]:
        try:
            return self._infer(units)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(units) == 1:
                raise
            mid = len(units) // 2
            return self.batch_infer(units[:mid]) + self.batch_infer(units[mid:])

    def _infer(self, units: list[WorkUnit]) -> list[np.ndarray]:
        results: list[np.ndarray | None] = [None] * len(units)
        by_shape: dict[tuple, list[int]] = defaultdict(list)
        for pos, u in enumerate(units):
            by_shape[u.payload.shape[:2]].append(pos)
        for shape, positions in by_shape.items():
            frames = [units[p].payload for p in positions]
            outs = self._infer_tiled(frames, shape) if self.tile else self._infer_whole(frames, shape)
            for p, frame in zip(positions, outs):
                results[p] = frame
        return results

    def _infer_whole(self, frames: list[np.ndarray], shape: tuple) -> list[np.ndarray]:
        batch = self._to_tensor(frames)
        with torch.no_grad():
            out = self.model(batch)
        return self._from_tensor(out, shape)

    def _infer_tiled(self, frames: list[np.ndarray], shape: tuple) -> list[np.ndarray]:
        h, w = shape
        if self.tile <= 0 or (h <= self.tile and w <= self.tile):
            return self._infer_whole(frames, shape)

        outputs = [np.zeros((h * 2, w * 2, 3), dtype=np.uint8) for _ in frames]
        tiles_x = (w + self.tile - 1) // self.tile
        tiles_y = (h + self.tile - 1) // self.tile
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x0 = tx * self.tile
                y0 = ty * self.tile
                x1 = min(x0 + self.tile, w)
                y1 = min(y0 + self.tile, h)
                px0 = max(x0 - self.tile_pad, 0)
                py0 = max(y0 - self.tile_pad, 0)
                px1 = min(x1 + self.tile_pad, w)
                py1 = min(y1 + self.tile_pad, h)

                tiles = [f[py0:py1, px0:px1] for f in frames]
                out_tiles = self._infer_tile_batch(tiles)

                ox0, ox1 = x0 * 2, x1 * 2
                oy0, oy1 = y0 * 2, y1 * 2
                crop_x0 = (x0 - px0) * 2
                crop_y0 = (y0 - py0) * 2
                crop_x1 = crop_x0 + (x1 - x0) * 2
                crop_y1 = crop_y0 + (y1 - y0) * 2
                for dst, tile in zip(outputs, out_tiles):
                    dst[oy0:oy1, ox0:ox1] = tile[crop_y0:crop_y1, crop_x0:crop_x1]
        return outputs

    def _infer_tile_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        try:
            batch = self._to_tensor(tiles)
            with torch.no_grad():
                out = self.model(batch)
            return self._from_tensor(out, tiles[0].shape[:2])
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if getattr(self, "_calibrating", False):
                raise
            if len(tiles) == 1:
                raise RuntimeError(
                    f"tile={self.tile} still OOMs; reduce tile size or disable other GPU work") from exc
            mid = len(tiles) // 2
            return self._infer_tile_batch(tiles[:mid]) + self._infer_tile_batch(tiles[mid:])

    def _to_tensor(self, frames: list[np.ndarray]) -> torch.Tensor:
        arr = np.stack([f[:, :, ::-1] for f in frames]).astype(np.float32) / 255.0  # BGR -> RGB
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).to(self.device)
        if self.half:
            t = t.half()
        h, w = t.shape[2], t.shape[3]
        if h % 2 or w % 2:  # pixel_unshuffle (scale 2) needs even dims
            t = F.pad(t, (0, w % 2, 0, h % 2), mode="reflect")
        return t

    def _from_tensor(self, out: torch.Tensor, shape: tuple) -> list[np.ndarray]:
        h, w = shape
        arr = out.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        frames = []
        for o in arr:
            rgb = o[: h * 2, : w * 2]  # crop the reflect padding (x2)
            frames.append(np.ascontiguousarray((rgb[:, :, ::-1] * 255.0).round().astype(np.uint8)))
        return frames

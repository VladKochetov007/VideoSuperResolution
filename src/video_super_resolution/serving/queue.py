from collections import defaultdict, deque
from typing import Iterable

from .unit import WorkUnit


class WorkQueue:
    # One FIFO per model: per-frame and windowed units have different tensor rank and cannot share
    # a forward. FIFO order within a model gives starvation-free service across videos; cross-video
    # backfill is emergent (drain pops the next n regardless of which video they belong to).
    def __init__(self):
        self._q: dict[str, deque] = defaultdict(deque)

    def enqueue(self, units: Iterable[WorkUnit]) -> None:
        for u in units:
            self._q[u.model].append(u)

    def drain(self, model: str, n: int) -> list[WorkUnit]:
        q = self._q[model]
        out = []
        while q and len(out) < n:
            out.append(q.popleft())
        return out

    def pending(self, model: str) -> int:
        return len(self._q[model])

    def models_with_work(self) -> list[str]:
        return [m for m, q in self._q.items() if q]

    def is_empty(self) -> bool:
        return all(not q for q in self._q.values())

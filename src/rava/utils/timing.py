from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimingResult:
    name: str
    started_at: float
    ended_at: float

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000.0


@contextmanager
def timed(name: str) -> Iterator[TimingResult]:
    start = time.time()
    result = TimingResult(name=name, started_at=start, ended_at=start)
    try:
        yield result
    finally:
        result.ended_at = time.time()

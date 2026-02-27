from __future__ import annotations


class PhiLeakTracker:
    def __init__(self) -> None:
        self.leak_events = 0

    def record(self, leaked: bool) -> None:
        if leaked:
            self.leak_events += 1

    def snapshot(self) -> dict[str, int]:
        return {"phi_leak_events": self.leak_events}

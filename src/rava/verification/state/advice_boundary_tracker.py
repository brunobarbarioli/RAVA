from __future__ import annotations


class AdviceBoundaryTracker:
    def __init__(self) -> None:
        self.violations = 0

    def record(self, violated: bool) -> None:
        if violated:
            self.violations += 1

    def snapshot(self) -> dict[str, int]:
        return {"advice_boundary_violations": self.violations}

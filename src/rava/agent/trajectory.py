from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrajectoryStep:
    step_id: int
    phase: str
    action: str
    observation: str
    started_at: float
    ended_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ms"] = self.duration_ms
        return payload


@dataclass
class AgentTrajectory:
    run_id: str
    example_id: str
    domain: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def to_rows(self) -> list[dict[str, Any]]:
        base = {
            "run_id": self.run_id,
            "example_id": self.example_id,
            "domain": self.domain,
        }
        return [{**base, **step.to_dict()} for step in self.steps]

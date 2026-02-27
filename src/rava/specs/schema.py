from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConstraintType(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"
    STATISTICAL = "STATISTICAL"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class VerificationAction(str, Enum):
    APPROVE = "APPROVE"
    BLOCK = "BLOCK"
    MODIFY = "MODIFY"
    FLAG = "FLAG"
    HALT = "HALT"


class Trigger(BaseModel):
    events: list[str] = Field(default_factory=list)
    condition: dict[str, Any] | None = None


class Constraint(BaseModel):
    id: str
    type: ConstraintType
    description: str
    trigger: Trigger = Field(default_factory=Trigger)
    predicate: str
    threshold: float | None = None
    metric: str | None = None
    grouping_fields: list[str] = Field(default_factory=list)


class Specification(BaseModel):
    domain: str
    composition: dict[str, Any] | None = None
    constraints: list[Constraint]


class ConstraintVerdict(BaseModel):
    constraint_id: str
    constraint_type: ConstraintType
    verdict: Verdict
    details: str = ""

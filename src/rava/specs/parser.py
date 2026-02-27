from __future__ import annotations

from pathlib import Path

from rava.specs.schema import Specification
from rava.utils.serialization import read_yaml


def load_spec(path: str | Path) -> Specification:
    raw = read_yaml(path)
    return Specification.model_validate(raw)

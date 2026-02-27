from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    observation: str
    metadata: dict[str, Any]


class Tool(ABC):
    name: str

    @abstractmethod
    def run(self, query: str, context: dict[str, Any] | None = None) -> ToolResult:
        raise NotImplementedError

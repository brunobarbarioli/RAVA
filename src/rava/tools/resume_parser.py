from __future__ import annotations

import re
from typing import Any

from rava.tools.base import Tool, ToolResult


class ResumeParserTool(Tool):
    name = "resume_parser"

    def run(self, query: str, context: dict[str, Any] | None = None) -> ToolResult:
        years = re.findall(r"(\d+)\s+years", query.lower())
        skills = re.findall(r"skills?:\s*([a-z0-9,\s\-]+)", query.lower())
        degree = re.search(r"(bachelor|master|phd)", query.lower())
        payload = {
            "years_experience": int(years[0]) if years else None,
            "skills": [s.strip() for s in skills[0].split(",")] if skills else [],
            "degree": degree.group(1) if degree else None,
        }
        return ToolResult(ok=True, observation=str(payload), metadata={"tool": self.name, **payload})

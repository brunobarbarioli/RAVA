from __future__ import annotations

import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from rava.utils.serialization import append_jsonl, write_json

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.path,
            {
                "ts": time.time(),
                "event": event,
                "payload": payload,
            },
        )


def capture_environment(path: str | Path) -> None:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    write_json(path, env)


def maybe_init_wandb(enable: bool = False) -> Any | None:
    if not enable:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        logging.getLogger(__name__).warning("wandb requested but not installed.")
        return None
    return wandb

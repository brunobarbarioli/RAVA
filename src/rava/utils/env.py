from __future__ import annotations


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)

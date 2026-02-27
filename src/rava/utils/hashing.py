from __future__ import annotations

import hashlib


def stable_hash(text: str, n_hex: int = 12) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:n_hex]

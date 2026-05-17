from __future__ import annotations

import os


def is_done(path: str) -> bool:
    return os.path.isfile(path)


def mark_done(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ok\n")

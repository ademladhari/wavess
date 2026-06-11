#!/usr/bin/env python3
"""Backward-compatible alias — use mainbenchmark instead."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _mainbenchmark():
    path = Path(__file__).resolve().parent / "mainbenchmark"
    spec = importlib.util.spec_from_file_location("dct_mainbenchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    return int(_mainbenchmark().main())


if __name__ == "__main__":
    raise SystemExit(main())

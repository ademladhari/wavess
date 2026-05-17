from __future__ import annotations

import os

from wmbench.watermarks.base import WatermarkAdapter
from wmbench.watermarks.dct import DCTAdapter
from wmbench.watermarks.dwt import DWTAdapter

_ADAPTERS: dict[str, type[WatermarkAdapter]] = {
    "dct": DCTAdapter,
    "dwt": DWTAdapter,
}


def get_adapter(method_id: str, **kwargs) -> WatermarkAdapter:
    method_id = method_id.strip().lower()
    if method_id not in _ADAPTERS:
        raise ValueError(f"Unknown method {method_id!r}; known: {sorted(_ADAPTERS)}")
    return _ADAPTERS[method_id](**kwargs)


def register_adapter(method_id: str, cls: type[WatermarkAdapter]) -> None:
    _ADAPTERS[method_id.strip().lower()] = cls


__all__ = ["DCTAdapter", "DWTAdapter", "WatermarkAdapter", "get_adapter", "register_adapter"]

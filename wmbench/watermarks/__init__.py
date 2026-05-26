from __future__ import annotations

import os

from wmbench.watermarks.base import WatermarkAdapter
from wmbench.watermarks.dct import DCTAdapter
from wmbench.watermarks.dct_dwt import DCTDWTAdapter
from wmbench.watermarks.dwt_dct_svd import DWTDCTSVDAdapter
from wmbench.watermarks.dwt import DWTAdapter
from wmbench.watermarks.flexible import FlexibleAdapter
from wmbench.watermarks.robin import ROBINAdapter
from wmbench.watermarks.tree_ring import TreeRingAdapter
from wmbench.watermarks.ssl import SSLAdapter
from wmbench.watermarks.svd import SVDAdapter

_ADAPTERS: dict[str, type[WatermarkAdapter]] = {
    "dct": DCTAdapter,
    "dct-dwt": DCTDWTAdapter,
    "dct_dwt": DCTDWTAdapter,
    "dctdwt": DCTDWTAdapter,
    "dwt": DWTAdapter,
    "dwt-dct-svd": DWTDCTSVDAdapter,
    "dwt_dct_svd": DWTDCTSVDAdapter,
    "dwtdctsvd": DWTDCTSVDAdapter,
    "flexible": FlexibleAdapter,
    "flex": FlexibleAdapter,
    "robin": ROBINAdapter,
    "tree-ring": TreeRingAdapter,
    "tree_ring": TreeRingAdapter,
    "treering": TreeRingAdapter,
    "ssl": SSLAdapter,
    "ssl-wm": SSLAdapter,
    "ssl_wm": SSLAdapter,
    "svd": SVDAdapter,
}


def get_adapter(method_id: str, **kwargs) -> WatermarkAdapter:
    method_id = method_id.strip().lower()
    if method_id not in _ADAPTERS:
        raise ValueError(f"Unknown method {method_id!r}; known: {sorted(_ADAPTERS)}")
    return _ADAPTERS[method_id](**kwargs)


def register_adapter(method_id: str, cls: type[WatermarkAdapter]) -> None:
    _ADAPTERS[method_id.strip().lower()] = cls


__all__ = [
    "DCTAdapter",
    "DCTDWTAdapter",
    "DWTDCTSVDAdapter",
    "DWTAdapter",
    "FlexibleAdapter",
    "ROBINAdapter",
    "TreeRingAdapter",
    "SSLAdapter",
    "SVDAdapter",
    "WatermarkAdapter",
    "get_adapter",
    "register_adapter",
]

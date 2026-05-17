from __future__ import annotations

import base64
import gzip
import json
import os
import stat
import warnings
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def chmod_group_write(path: str) -> None:
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")
    # Windows has no os.getuid(); skip unix group chmod logic there.
    if not hasattr(os, "getuid"):
        return
    if os.stat(path).st_uid == os.getuid():
        current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
        os.chmod(path, current_permissions | stat.S_IWGRP)


def _load_json_bytes(data: bytes) -> Any:
    # WAVES sometimes writes JSON via orjson; stdlib json can read it fine.
    return json.loads(data.decode("utf-8"))


def load_json(filepath: str) -> Any:
    try:
        with open(filepath, "rb") as json_file:
            return _load_json_bytes(json_file.read())
    except json.JSONDecodeError:
        warnings.warn(f"Found invalid JSON file {filepath}, deleting")
        os.remove(filepath)
        return None


def _compare_dicts(dict1: Any, dict2: Any) -> bool:
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(_compare_dicts(dict1[k], dict2[k]) for k in dict1)
    return dict1 == dict2


def save_json(data: Any, filepath: str) -> None:
    if os.path.exists(filepath) and (existing_data := load_json(filepath)) is not None:
        if _compare_dicts(data, existing_data):
            return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as json_file:
        json_file.write(json.dumps(data, separators=(",", ":")).encode("utf-8"))
    chmod_group_write(filepath)


def encode_array_to_string(array: np.ndarray) -> str:
    meta = json.dumps({"shape": array.shape, "dtype": str(array.dtype)}, separators=(",", ":")).encode(
        "utf-8"
    )
    combined = meta + b"\x00" + array.tobytes()
    compressed = gzip.compress(combined)
    return base64.b64encode(compressed).decode("utf-8")


def decode_array_from_string(encoded_string: str) -> np.ndarray:
    decoded_bytes = base64.b64decode(encoded_string)
    decompressed = gzip.decompress(decoded_bytes)
    meta_encoded, array_bytes = decompressed.split(b"\x00", 1)
    meta = json.loads(meta_encoded.decode("utf-8"))
    shape, dtype = meta["shape"], meta["dtype"]
    return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)


def encode_image_to_string(image: Image.Image, quality: int = 90) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(gzip.compress(buffered.getvalue())).decode("utf-8")


def decode_image_from_string(encoded_string: str) -> Image.Image:
    img_data = gzip.decompress(base64.b64decode(encoded_string))
    return Image.open(BytesIO(img_data))

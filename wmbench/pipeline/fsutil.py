"""Filesystem helpers for pipeline stages."""

from __future__ import annotations

import os

from PIL import Image


def atomic_image_save(image: Image.Image, dest: str) -> None:
    """Write image to ``dest`` via a temp file + ``os.replace`` to avoid truncated files on interrupt."""
    dest = os.path.abspath(dest)
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    base_name = os.path.basename(dest)
    stem, ext = os.path.splitext(base_name)
    tmp = os.path.join(parent, f".{stem}.wmbench_partial{ext}")
    try:
        image.save(tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            if os.path.isfile(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise

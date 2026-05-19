from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class WatermarkAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed(self, image: Image.Image) -> Image.Image:
        """Return watermarked image (RGB)."""

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        """Detection score in [0, 1]; higher = more likely watermarked.

        Non-blind (``blind=False``): uses ``original`` host image (oracle reference).
        Blind (``blind=True``): uses only ``meta`` from embed sidecar (no original pixels).
        """

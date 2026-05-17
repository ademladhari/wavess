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
    def detect(self, image: Image.Image, original: Image.Image, *, meta: dict | None = None) -> float:
        """Non-blind score in [0, 1]; higher = more likely watermarked.

        Optional `meta` carries method-specific state (e.g. DWT payload from embed).
        """

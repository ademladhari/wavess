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

    def embed_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """Embed a minibatch. Override for GPU-batched or parallel embed paths."""
        return [self.embed(im) for im in images]

    @property
    def embed_meta_shared(self) -> bool:
        """When True, one sidecar payload applies to every image in an embed batch."""
        return False

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

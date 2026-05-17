from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from PIL import Image


class Attack(ABC):
    """Single attack operator (one named benchmark attack, possibly combo under the hood)."""

    name: str
    strengths: Sequence[float | int]

    @abstractmethod
    def apply(self, image: Image.Image, strength: float | int) -> Image.Image:
        raise NotImplementedError

    def apply_batch(self, images: list[Image.Image], strength: float | int) -> list[Image.Image]:
        """Optional batched fast path.

        Default keeps legacy semantics by applying single-image operator repeatedly.
        """
        return [self.apply(im, strength) for im in images]

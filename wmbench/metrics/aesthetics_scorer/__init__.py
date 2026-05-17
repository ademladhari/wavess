"""Aesthetics scorer utilities.

Based on code vendored in the original WAVES repository. We avoid importing `waves.*`
and instead load local weights from disk.
"""

from .model import load_model, preprocess

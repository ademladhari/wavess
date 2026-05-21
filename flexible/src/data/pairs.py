"""Dataset wrapping the pre-generated (w, zT, Iw) pool (see generate_pairs.py)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PairsDataset(Dataset):
    """Iterate over .pt pair files produced by generate_pairs.py."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob("pair_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No pair_*.pt files under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        blob = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        image = blob["image_uint8"].float() / 255.0  # (3, H, W) in [0, 1]
        return {
            "prompt": blob["prompt"],
            "w": blob["w"].float(),
            "zT": blob["zT"].float(),
            "image": image,
            "n_bits": int(blob["n_bits"]),
        }


def collate_pairs(batch: list[dict]) -> dict:
    out = {
        "prompt": [b["prompt"] for b in batch],
        "w": torch.stack([b["w"] for b in batch], dim=0),
        "zT": torch.stack([b["zT"] for b in batch], dim=0),
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "n_bits": batch[0]["n_bits"],
    }
    return out

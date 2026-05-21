# pip install git+https://github.com/openai/CLIP.git
import hashlib
import os
import urllib.request
import warnings
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import clip
from .fid import compute_fid
from tqdm import tqdm


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_streaming(url: str, root: str):
    """
    Drop-in replacement for clip._download that avoids loading whole model files
    into RAM for SHA256 checks (prevents MemoryError on Windows).
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if _sha256_file(download_target) == expected_sha256:
            return download_target
        warnings.warn(f"{download_target} exists, but checksum does not match; re-downloading")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total = source.info().get("Content-Length")
        total_i = int(total) if total is not None else None
        with tqdm(total=total_i, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if _sha256_file(download_target) != expected_sha256:
        raise RuntimeError("Model downloaded but SHA256 does not match")

    return download_target


# Patch CLIP downloader globally for this process.
clip._download = _download_streaming


def img_preprocess_clip(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    T = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    )
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)


class CLIP_fx:
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.eval()
        self.name = "clip_" + name.lower().replace("-", "_").replace("/", "_")

    def __call__(self, img_t):
        img_x = img_t / 255.0
        T_norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape) == 3:
            img_x = img_x.unsqueeze(0)
        B, C, H, W = img_x.shape
        with torch.no_grad():
            z = self.model.encode_image(img_x)
        return z

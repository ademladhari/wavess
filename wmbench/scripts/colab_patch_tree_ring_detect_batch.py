"""
In-place Colab patch: tree-ring batched detect + fp32 FFT + 512 resize.
Run from repo root:
  python wmbench/scripts/colab_patch_tree_ring_detect_batch.py
Or in Colab:
  %run /content/wavess/wmbench/scripts/colab_patch_tree_ring_detect_batch.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _waves_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent.parent.parent, Path("/content/wavess"), Path.cwd()]:
        if (p / "wmbench" / "run_benchmark.py").is_file():
            return p
    raise SystemExit("Could not find waves root (wmbench/run_benchmark.py). Set cwd or fix path.")


def _patch_detect_py(path: Path) -> None:
    """Incremental detect.py patches (old wmbench without batched tree-ring)."""
    _patch(
        path,
        '''def _tree_ring_detect_worker(
    items: list[tuple[int, str]],
    *,
    blind_detect: bool,
    originals_dir: str,
    visible_gpu: str,
) -> list[tuple[int, float]]:
    global _TREE_RING_WORKER_ADAPTER, _TREE_RING_WORKER_DEVICE

    # Hard-pin this worker to a concrete CUDA device id.
    device = f"cuda:{visible_gpu}"
    os.environ["WMBENCH_TREE_RING_DEVICE"] = device
    os.environ["WMBENCH_TREE_RING_INVERT_DEVICE"] = device

    if _TREE_RING_WORKER_ADAPTER is None or _TREE_RING_WORKER_DEVICE != device:
        from wmbench.watermarks import get_adapter

        _TREE_RING_WORKER_ADAPTER = get_adapter("tree-ring")
        _TREE_RING_WORKER_DEVICE = device

    adapter = _TREE_RING_WORKER_ADAPTER
    scores: list[tuple[int, float]] = []
    for idx, p in items:
        with Image.open(p) as im:
            img = im.convert("RGB")
        if blind_detect:
            sc = float(adapter.detect(img, None, meta=None, blind=True))
        else:
            orig_path = os.path.join(originals_dir, os.path.basename(p))
            with Image.open(orig_path) as oi:
                oimg = oi.convert("RGB")
            sc = float(adapter.detect(img, oimg, meta=None, blind=False))
        scores.append((idx, sc))
    return scores


def _tree_ring_score_paths_multi_gpu(
    paths: list[str],
    *,
    blind_detect: bool,
    originals_dir: str,
    gpu_ids: list[str],
    desc: str,
) -> list[float]:''',
        '''def _tree_ring_detect_worker(
    items: list[tuple[int, str]],
    *,
    blind_detect: bool,
    originals_dir: str,
    visible_gpu: str,
    detect_batch_size: int,
) -> list[tuple[int, float]]:
    global _TREE_RING_WORKER_ADAPTER, _TREE_RING_WORKER_DEVICE

    # Hard-pin this worker to a concrete CUDA device id.
    device = f"cuda:{visible_gpu}"
    os.environ["WMBENCH_TREE_RING_DEVICE"] = device
    os.environ["WMBENCH_TREE_RING_INVERT_DEVICE"] = device

    if _TREE_RING_WORKER_ADAPTER is None or _TREE_RING_WORKER_DEVICE != device:
        from wmbench.watermarks import get_adapter

        _TREE_RING_WORKER_ADAPTER = get_adapter("tree-ring")
        _TREE_RING_WORKER_DEVICE = device

    adapter = _TREE_RING_WORKER_ADAPTER
    batch_size = max(1, int(detect_batch_size))
    scores: list[tuple[int, float]] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        images: list[Image.Image] = []
        for _, p in chunk:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        if blind_detect:
            batch_scores = adapter.detect_batch(images, blind=True)
        else:
            originals: list[Image.Image | None] = []
            for _, p in chunk:
                orig_path = os.path.join(originals_dir, os.path.basename(p))
                with Image.open(orig_path) as oi:
                    originals.append(oi.convert("RGB"))
            batch_scores = adapter.detect_batch(images, originals, blind=False)
        for (idx, _), sc in zip(chunk, batch_scores):
            scores.append((idx, float(sc)))
    return scores


def _tree_ring_score_paths_batched(
    adapter: WatermarkAdapter,
    paths: list[str],
    *,
    blind_detect: bool,
    originals_dir: str,
    detect_batch_size: int,
    desc: str,
    embed_meta_by_base: dict[str, dict | None] | None = None,
    svd_neg_payload_bank: list[dict] | None = None,
    dwt_dct_svd_neg_payload_bank: list[dict] | None = None,
) -> list[float]:
    """Batched tree-ring detect (DDIM invert minibatches on one GPU)."""
    batch_size = max(1, int(detect_batch_size))
    scores: list[float] = []
    n = len(paths)
    with tqdm(total=n, desc=desc, unit="img") as pbar:
        for start in range(0, n, batch_size):
            batch_paths = paths[start : start + batch_size]
            images: list[Image.Image] = []
            originals: list[Image.Image | None] = []
            metas: list[dict | None] = []
            for i, p in enumerate(batch_paths):
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
                if blind_detect:
                    if adapter.name == "svd" and svd_neg_payload_bank:
                        metas.append(svd_neg_payload_bank[(start + i) % len(svd_neg_payload_bank)])
                    elif embed_meta_by_base is not None:
                        metas.append(embed_meta_by_base.get(os.path.basename(p)))
                    else:
                        metas.append(None)
                    originals.append(None)
                else:
                    orig_path = os.path.join(originals_dir, os.path.basename(p))
                    with Image.open(orig_path) as oi:
                        originals.append(oi.convert("RGB"))
                    if adapter.name == "dwt-dct-svd" and dwt_dct_svd_neg_payload_bank:
                        metas.append(
                            dwt_dct_svd_neg_payload_bank[(start + i) % len(dwt_dct_svd_neg_payload_bank)]
                        )
                    else:
                        metas.append(None)
            if adapter.name == "tree-ring":
                if blind_detect:
                    batch_scores = adapter.detect_batch(images, blind=True)
                else:
                    batch_scores = adapter.detect_batch(images, originals, blind=False)
            else:
                batch_scores = adapter.detect_batch(images, originals, metas=metas, blind=blind_detect)
            scores.extend(float(s) for s in batch_scores)
            pbar.update(len(batch_paths))
    return scores


def _tree_ring_score_paths_multi_gpu(
    paths: list[str],
    *,
    blind_detect: bool,
    originals_dir: str,
    gpu_ids: list[str],
    desc: str,
    detect_batch_size: int,
) -> list[float]:''',
        name="detect.py worker+batched helper",
    )
    _patch(
        path,
        '''                visible_gpu=gid,
            )
            for gid, chunk in chunks''',
        '''                visible_gpu=gid,
                detect_batch_size=detect_batch_size,
            )
            for gid, chunk in chunks''',
        name="detect.py mgpu worker arg",
    )
    _patch(
        path,
        '''    resume: bool = False,
    blind_detect: bool = False,
) -> None:
    watermarked_dir = os.path.join(work_dir, "watermarked")''',
        '''    resume: bool = False,
    blind_detect: bool = False,
    detect_batch_size: int = 1,
) -> None:
    watermarked_dir = os.path.join(work_dir, "watermarked")''',
        name="detect.py run_detect_stage sig",
    )
    _patch(
        path,
        '''    tree_ring_gpus = _tree_ring_detect_gpu_ids() if adapter.name == "tree-ring" else []
    if os.path.isfile(cache_path):''',
        '''    tree_ring_gpus = _tree_ring_detect_gpu_ids() if adapter.name == "tree-ring" else []
    if adapter.name == "tree-ring":
        detect_batch_size = max(1, int(detect_batch_size))
        env_batch = os.environ.get("WMBENCH_TREE_RING_DETECT_BATCH", "").strip()
        if env_batch and detect_batch_size <= 1:
            detect_batch_size = max(1, int(env_batch))
        if detect_batch_size > 1:
            print(f"tree-ring detect batch size: {detect_batch_size}", flush=True)
    if os.path.isfile(cache_path):''',
        name="detect.py batch size setup",
    )
    _patch(
        path,
        '''                    desc=f"neg/{adapter.name}",
                )
                used_tree_ring_mgpu = True''',
        '''                    desc=f"neg/{adapter.name}",
                    detect_batch_size=detect_batch_size,
                )
                used_tree_ring_mgpu = True''',
        name="detect.py neg mgpu batch arg",
    )
    _patch(
        path,
        '''        if not used_tree_ring_mgpu:
            for i, p in enumerate(tqdm(neg_paths, desc=f"neg/{adapter.name}")):
                with Image.open(p) as im:
                    neg = im.convert("RGB")
                if blind_detect:
                    if adapter.name == "svd":
                        key_meta = svd_neg_payload_bank[i % len(svd_neg_payload_bank)]
                        neg_scores.append(float(adapter.detect(neg, None, meta=key_meta, blind=True)))
                    else:
                        neg_scores.append(float(adapter.detect(neg, None, meta=None, blind=True)))
                else:
                    orig_path = os.path.join(originals_dir, os.path.basename(p))
                    if not os.path.isfile(orig_path):
                        raise FileNotFoundError(
                            "Negative image does not have a matching original by basename for calibration: "
                            f"{os.path.basename(p)!r} (expected original at {orig_path})"
                        )
                    with Image.open(orig_path) as oi:
                        orig = oi.convert("RGB")
                    neg_meta = None
                    if adapter.name == "dwt-dct-svd":
                        neg_meta = dwt_dct_svd_neg_payload_bank[i % len(dwt_dct_svd_neg_payload_bank)]
                    neg_scores.append(float(adapter.detect(neg, orig, meta=neg_meta, blind=False)))
        try:''',
        '''        if not used_tree_ring_mgpu:
            if adapter.name == "tree-ring" and detect_batch_size > 1:
                neg_scores = _tree_ring_score_paths_batched(
                    adapter,
                    neg_paths,
                    blind_detect=blind_detect,
                    originals_dir=originals_dir,
                    detect_batch_size=detect_batch_size,
                    desc=f"neg/{adapter.name}",
                )
            else:
                for i, p in enumerate(tqdm(neg_paths, desc=f"neg/{adapter.name}")):
                    with Image.open(p) as im:
                        neg = im.convert("RGB")
                    if blind_detect:
                        if adapter.name == "svd":
                            key_meta = svd_neg_payload_bank[i % len(svd_neg_payload_bank)]
                            neg_scores.append(float(adapter.detect(neg, None, meta=key_meta, blind=True)))
                        else:
                            neg_scores.append(float(adapter.detect(neg, None, meta=None, blind=True)))
                    else:
                        orig_path = os.path.join(originals_dir, os.path.basename(p))
                        if not os.path.isfile(orig_path):
                            raise FileNotFoundError(
                                "Negative image does not have a matching original by basename for calibration: "
                                f"{os.path.basename(p)!r} (expected original at {orig_path})"
                            )
                        with Image.open(orig_path) as oi:
                            orig = oi.convert("RGB")
                        neg_meta = None
                        if adapter.name == "dwt-dct-svd":
                            neg_meta = dwt_dct_svd_neg_payload_bank[i % len(dwt_dct_svd_neg_payload_bank)]
                        neg_scores.append(float(adapter.detect(neg, orig, meta=neg_meta, blind=False)))
        try:''',
        name="detect.py neg batched path",
    )
    _patch(
        path,
        '''                        desc=f"scores/{attack_name}/{stren_tag}",
                    )
                except Exception as e:
                    print(
                        f"tree-ring multi-GPU scores failed ({e!r}); falling back to single-process detect.",
                        flush=True,
                    )
                    for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                        pos_scores.append(_score_path(ap))
            else:
                for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                    pos_scores.append(_score_path(ap))''',
        '''                        desc=f"scores/{attack_name}/{stren_tag}",
                        detect_batch_size=detect_batch_size,
                    )
                except Exception as e:
                    print(
                        f"tree-ring multi-GPU scores failed ({e!r}); falling back to single-process detect.",
                        flush=True,
                    )
                    if detect_batch_size > 1:
                        pos_scores = _tree_ring_score_paths_batched(
                            adapter,
                            atk_paths,
                            blind_detect=blind_detect,
                            originals_dir=originals_dir,
                            detect_batch_size=detect_batch_size,
                            desc=f"scores/{attack_name}/{stren_tag}",
                            embed_meta_by_base=embed_meta_by_base,
                        )
                    else:
                        for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                            pos_scores.append(_score_path(ap))
            elif adapter.name == "tree-ring" and detect_batch_size > 1:
                pos_scores = _tree_ring_score_paths_batched(
                    adapter,
                    atk_paths,
                    blind_detect=blind_detect,
                    originals_dir=originals_dir,
                    detect_batch_size=detect_batch_size,
                    desc=f"scores/{attack_name}/{stren_tag}",
                    embed_meta_by_base=embed_meta_by_base,
                )
            else:
                for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                    pos_scores.append(_score_path(ap))''',
        name="detect.py scores batched path",
    )
    print("  ok   detect.py (incremental patches)")


def _patch(path: Path, old: str, new: str, *, name: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  skip {name} (already patched)")
        return
    if old not in text:
        raise SystemExit(f"  FAIL {name}: expected block not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  ok   {name}")


def main() -> None:
    root = _waves_root()
    print(f"Patching under {root}")

    fourier = root / "tree-ring" / "src" / "treering" / "fourier.py"
    _patch(
        fourier,
        '''def fft2_shifted(latents: torch.Tensor) -> torch.Tensor:
    """Return 2D FFT with centered frequencies."""
    return torch.fft.fftshift(torch.fft.fft2(latents, dim=(-2, -1)), dim=(-2, -1))''',
        '''def fft2_shifted(latents: torch.Tensor) -> torch.Tensor:
    """Return 2D FFT with centered frequencies."""
    # cuFFT half/complex-half requires power-of-2 sizes; fp32 FFT works for any latent shape.
    if latents.is_cuda and latents.dtype in (torch.float16, torch.bfloat16):
        x = latents.float()
        return torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fft2(latents, dim=(-2, -1)), dim=(-2, -1))''',
        name="fourier.py fft fp32",
    )

    invert = root / "tree-ring" / "src" / "treering" / "invert.py"
    _patch(
        invert,
        '''    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        tensor = tensor * 2.0 - 1.0
        with torch.no_grad():
            latent_dist = self.pipeline.vae.encode(tensor).latent_dist
            # Deterministic encoding is crucial for stable inversion-based detection.
            latents = latent_dist.mode() * self.pipeline.vae.config.scaling_factor
        return latents''',
        '''    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("images must be non-empty")
        tensors = [TF.to_tensor(image.convert("RGB")).unsqueeze(0) for image in images]
        batch = torch.cat(tensors, dim=0).to(device=self.device, dtype=self.dtype)
        batch = batch * 2.0 - 1.0
        with torch.no_grad():
            latent_dist = self.pipeline.vae.encode(batch).latent_dist
            # Deterministic encoding is crucial for stable inversion-based detection.
            latents = latent_dist.mode() * self.pipeline.vae.config.scaling_factor
        return latents''',
        name="invert.py batched VAE encode",
    )
    _patch(
        invert,
        "        latents = torch.cat([self._encode_image(image) for image in images], dim=0)",
        "        latents = self._encode_images(images)",
        name="invert.py use _encode_images",
    )

    base = root / "wmbench" / "watermarks" / "base.py"
    if "def detect_batch(" not in base.read_text(encoding="utf-8"):
        _patch(
            base,
            '''        Blind (``blind=True``): uses only ``meta`` from embed sidecar (no original pixels).
        """
''',
            '''        Blind (``blind=True``): uses only ``meta`` from embed sidecar (no original pixels).
        """

    def detect_batch(
        self,
        images: list[Image.Image],
        originals: list[Image.Image | None] | None = None,
        *,
        metas: list[dict | None] | None = None,
        blind: bool = False,
    ) -> list[float]:
        """Score a minibatch. Override when the backend supports batched inference."""
        if originals is None:
            originals = [None] * len(images)
        if metas is None:
            metas = [None] * len(images)
        return [
            self.detect(im, orig, meta=meta, blind=blind)
            for im, orig, meta in zip(images, originals, metas)
        ]

''',
            name="base.py detect_batch",
        )
    else:
        print("  skip base.py detect_batch (already patched)")

    tree_ring = root / "wmbench" / "watermarks" / "tree_ring.py"
    if "def detect_batch(" not in tree_ring.read_text(encoding="utf-8"):
        _patch(
            tree_ring,
            '''def _to_score_from_pvalue(p_value: float) -> float:
    # Tree-Ring uses smaller p-value => stronger watermark evidence.
    return float(np.clip(1.0 - float(p_value), 0.0, 1.0))


class TreeRingAdapter''',
            '''def _to_score_from_pvalue(p_value: float) -> float:
    # Tree-Ring uses smaller p-value => stronger watermark evidence.
    return float(np.clip(1.0 - float(p_value), 0.0, 1.0))


def _tree_ring_detect_image_size() -> int:
    return max(8, int(os.environ.get("WMBENCH_TREE_RING_IMAGE_SIZE", "512") or "512"))


def _prepare_tree_ring_image(image: Image.Image) -> Image.Image:
    size = _tree_ring_detect_image_size()
    im = image.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return im


class TreeRingAdapter''',
            name="tree_ring.py resize helpers",
        )
        _patch(
            tree_ring,
            '''    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, meta, blind
        inverter = self._ensure_inverter()
        inverted = inverter.invert(
            images=[image.convert("RGB")],
            invert_prompt=self._cfg.detection.invert_prompt,
            num_inversion_steps=self._cfg.detection.num_inversion_steps,
        )
        detection = self._detect_watermark(
            inverted_latents=inverted,
            key=self._key,
            threshold=self._cfg.detection.threshold,
            alpha=self._cfg.detection.alpha,
            variance_estimation=self._cfg.detection.variance_estimation,
            pvalue_tail=self._cfg.detection.pvalue_tail,
        )[0]
        return _to_score_from_pvalue(detection.p_value)
''',
            '''    def _invert_and_score(self, images: list[Image.Image]) -> list[float]:
        if not images:
            return []
        prepared = [_prepare_tree_ring_image(im) for im in images]
        inverter = self._ensure_inverter()
        inverted = inverter.invert(
            images=prepared,
            invert_prompt=self._cfg.detection.invert_prompt,
            num_inversion_steps=self._cfg.detection.num_inversion_steps,
        )
        detections = self._detect_watermark(
            inverted_latents=inverted,
            key=self._key,
            threshold=self._cfg.detection.threshold,
            alpha=self._cfg.detection.alpha,
            variance_estimation=self._cfg.detection.variance_estimation,
            pvalue_tail=self._cfg.detection.pvalue_tail,
        )
        return [_to_score_from_pvalue(d.p_value) for d in detections]

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        del original, meta, blind
        return self._invert_and_score([image])[0]

    def detect_batch(
        self,
        images: list[Image.Image],
        originals: list[Image.Image | None] | None = None,
        *,
        metas: list[dict | None] | None = None,
        blind: bool = False,
    ) -> list[float]:
        del originals, metas, blind
        return self._invert_and_score(images)
''',
            name="tree_ring.py detect_batch",
        )
    else:
        print("  skip tree_ring.py (already patched)")

    detect_py = root / "wmbench" / "pipeline" / "detect.py"
    if "_tree_ring_score_paths_batched" not in detect_py.read_text(encoding="utf-8"):
        bundled = root / "wmbench" / "scripts" / "_detect_py_tree_ring_batch.patched"
        if bundled.is_file():
            detect_py.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
            print("  ok   detect.py (bundled full file)")
        else:
            _patch_detect_py(detect_py)
    else:
        print("  skip detect.py (already patched)")

    run_bm = root / "wmbench" / "run_benchmark.py"
    if "--detect-batch-size" not in run_bm.read_text(encoding="utf-8"):
        _patch(
            run_bm,
            '''    p.add_argument(
        "--lpips-batch-size",
        type=int,
        default=1,
        help="LPIPS pair minibatch size during evaluate phase.",
    )
    p.add_argument(
        "--profile-stages",''',
            '''    p.add_argument(
        "--lpips-batch-size",
        type=int,
        default=1,
        help="LPIPS pair minibatch size during evaluate phase.",
    )
    p.add_argument(
        "--detect-batch-size",
        type=int,
        default=1,
        help=(
            "Tree-Ring detect minibatch (batched DDIM inversion per chunk). "
            "Try 8–16 on A100/L4; lower on 8GB GPUs."
        ),
    )
    p.add_argument(
        "--profile-stages",''',
            name="run_benchmark.py CLI flag",
        )
        _patch(
            run_bm,
            '''            blind_detect=args.blind_detect,
        )
        if args.profile_stages:
            _print_stage_profile(f"{m}/detect", time.perf_counter() - t0, device)''',
            '''            blind_detect=args.blind_detect,
            detect_batch_size=args.detect_batch_size,
        )
        if args.profile_stages:
            _print_stage_profile(f"{m}/detect", time.perf_counter() - t0, device)''',
            name="run_benchmark.py pass detect_batch_size",
        )
    else:
        print("  skip run_benchmark.py (already patched)")

    print("Done. Use --detect-batch-size 16 on tree-ring runs.")


if __name__ == "__main__":
    main()

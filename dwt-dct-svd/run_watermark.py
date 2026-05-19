"""
Run the paper's §3.2–§3.3 pipeline on local images.

- original.png → RGB cover (host).
- fingerprint.* or picture.* → biometric (grayscale, resized), §3.2 part 1 (fingerprint wins if both exist).
- logo.png     → signature (grayscale, resized smaller than biometric), §3.2 part 1.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from skimage import color, io, util
from skimage.transform import resize

from dwt_dct_svd import (
    embed_signature_in_biometric,
    embed_tw_in_cover,
    extract_signature_from_tw,
    extract_tw_from_watermarked,
    key_generation,
    ncc,
    psnr_channelwise,
    to_float01,
)


def _rgba_to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[-1] != 4:
        return arr
    rgb = arr[..., :3].astype(np.float64) / 255.0
    a = arr[..., 3:4].astype(np.float64) / 255.0
    comp = rgb * a + (1.0 - a)
    return np.clip(np.round(comp * 255.0), 0, 255).astype(np.uint8)


def _find_image(directory: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No image named {stem}.* in {directory} (tried .png, .jpg, .jpeg, .webp, .bmp)"
    )


def _find_biometric_image(directory: Path) -> Path:
    """Prefer fingerprint.* (e.g. fingerprint.png), else picture.*."""
    for stem in ("fingerprint", "picture"):
        try:
            return _find_image(directory, stem)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"No biometric image in {directory}: add fingerprint.png or picture.png "
        "(or .jpg, .jpeg, .webp, .bmp)"
    )


def _save_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = util.img_as_ubyte(np.clip(to_float01(img), 0.0, 1.0))
    io.imsave(path, u8)


def _save_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.clip(to_float01(img), 0.0, 1.0)
    if g.ndim == 3:
        g = color.rgb2gray(g)
    io.imsave(path, util.img_as_ubyte(g))


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Paper §3.2–3.3: biometric+signature TW, then embed TW in cover"
    )
    p.add_argument("--original", type=Path, default=None, help="Cover RGB (default: original.png)")
    p.add_argument(
        "--picture",
        type=Path,
        default=None,
        help="Biometric image (default: fingerprint.* if present, else picture.*)",
    )
    p.add_argument("--logo", type=Path, default=None, help="Signature (default: logo.png)")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.04,
        help="Strength α (§3.2 part 1 / §3.3 part 2); tune with SV energy matching on",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.002,
        help="Strength β (§3.2 part 2 / §3.3 part 1); smaller → higher PSNR",
    )
    p.add_argument("--bio-size", type=int, default=256, help="Biometric side length (Sec. 4: 256)")
    p.add_argument("--sig-size", type=int, default=128, help="Signature max side (smaller than biometric)")
    p.add_argument(
        "--cover-size",
        type=int,
        default=None,
        help="If set (e.g. 512), resize RGB cover to S×S before embedding (Sec. 4 host size).",
    )
    p.add_argument("--out-dir", type=Path, default=root / "output")
    p.add_argument(
        "--key-hex",
        type=str,
        default=None,
        help="Optional 32 hex chars = 128-bit key for §3.1; default = SHA-256(cover file)[:16]",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Try many α, β values (see sweep_metrics.py), print best PSNR / NCC rows, then exit.",
    )
    p.add_argument(
        "--auto-search",
        action="store_true",
        help="Up to 100 (α,β) trials on a built-in grid: maximize NCC_TW with PSNR≥50 (see auto_search_ab.py).",
    )
    args = p.parse_args()

    if args.sweep:
        import subprocess
        import sys

        cmd = [sys.executable, str(root / "sweep_metrics.py")]
        if args.cover_size is not None:
            cmd.extend(["--cover-size", str(args.cover_size)])
        raise SystemExit(subprocess.call(cmd))

    if args.auto_search:
        import subprocess
        import sys

        cmd = [sys.executable, str(root / "auto_search_ab.py")]
        if args.cover_size is not None:
            cmd.extend(["--cover-size", str(args.cover_size)])
        raise SystemExit(subprocess.call(cmd))

    orig_path = Path(args.original) if args.original else _find_image(root, "original")
    pic_path = Path(args.picture) if args.picture else _find_biometric_image(root)
    logo_path = Path(args.logo) if args.logo else _find_image(root, "logo")

    cover = _rgba_to_rgb_u8(io.imread(orig_path))
    if args.cover_size is not None:
        s = int(args.cover_size)
        cover = resize(to_float01(cover), (s, s), anti_aliasing=True)
    picture_u8 = _rgba_to_rgb_u8(io.imread(pic_path))
    logo_u8 = _rgba_to_rgb_u8(io.imread(logo_path))

    # Sec. 3: biometric larger than signature; Sec. 4: ~256 biometric, smaller signature.
    bio_gray = color.rgb2gray(to_float01(picture_u8))
    biometric = resize(bio_gray, (args.bio_size, args.bio_size), anti_aliasing=True)
    sig_gray = color.rgb2gray(to_float01(logo_u8))
    signature = resize(sig_gray, (args.sig_size, args.sig_size), anti_aliasing=True)

    # §3.1: 128-bit key (first 16 bytes of SHA-256 of cover file bytes — deterministic host binding).
    H, W = cover.shape[0], cover.shape[1]
    if args.key_hex:
        key128 = bytes.fromhex(args.key_hex.replace(" ", ""))
        if len(key128) != 16:
            raise SystemExit("--key-hex must be exactly 32 hexadecimal characters (128 bits).")
    else:
        key128 = hashlib.sha256(Path(orig_path).read_bytes()).digest()[:16]
    A, _meta = key_generation(key128, H, W)

    # §3.2 — non-blind: state carries Us/Vs/Ub/Vb/Sb/details so §3.3 can rebuild images.
    TW, state = embed_signature_in_biometric(biometric, signature, alpha=args.alpha)
    watermarked, state = embed_tw_in_cover(cover, TW, beta=args.beta, state=state)

    # Realistic uint8 PNG round-trip: the "watermarked image" as delivered (quantized to 8 bits).
    wm_uint8 = util.img_as_ubyte(np.clip(watermarked, 0.0, 1.0))
    wm_png = wm_uint8.astype(np.float64) / 255.0

    # §3.3 (non-blind: cover + biometric + saved state matrices), extracted from the saved PNG.
    TW_ext = extract_tw_from_watermarked(wm_png, cover, beta=args.beta, state=state)
    sig_ext = extract_signature_from_tw(TW_ext, biometric, alpha=args.alpha, state=state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(out_dir / "watermarked.png", wm_uint8)
    _save_gray(out_dir / "transformed_watermark_TW.png", TW)
    _save_gray(out_dir / "TW_extracted.png", TW_ext)
    _save_gray(out_dir / "signature_extracted.png", sig_ext)
    _save_gray(out_dir / "biometric_used.png", biometric)
    np.save(out_dir / "key_matrix_A.npy", A)

    psnr_mean, psnr_ch = psnr_channelwise(cover, wm_png)
    sig_ncc = ncc(signature, sig_ext)
    tw_ncc = ncc(TW, TW_ext)

    print(f"Cover:       {orig_path}  shape={cover.shape}")
    print(f"Biometric:   {pic_path}  (resized to {biometric.shape})")
    print(f"Signature:   {logo_path}  (resized to {signature.shape})")
    print(
        f"§3.1 key A:  shape={A.shape}, saved {out_dir / 'key_matrix_A.npy'} "
        "(non-blind state matrices are additional keys; see saved_state.npz)"
    )
    print(f"α={args.alpha}, β={args.beta}")
    pch = [round(float(x), 2) for x in psnr_ch]
    print(f"PSNR (mean of R,G,B vs paper Sec. 4.1): {psnr_mean:.2f} dB  per-channel: {pch}")
    print(f"NCC (TW vs TW_extracted) Eq. (7):        {tw_ncc:.4f}")
    print(f"NCC (signature vs signature_extracted):  {sig_ncc:.4f}")
    print(f"Outputs in: {out_dir}")

    np.savez(
        out_dir / "state.npz",
        Us=state.Us,
        Vts=state.Vts,
        Ub=state.Ub,
        Vtb=state.Vtb,
        sb=state.sb,
        alpha=np.array([args.alpha]),
        beta=np.array([args.beta]),
    )


if __name__ == "__main__":
    main()

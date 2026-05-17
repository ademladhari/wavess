"""
SCRATCHPAD (required mapping before writing any `wmbench/` code)

Constraint recap
- `wmbench/` must be self-contained.
- At runtime, it must NOT import from `waves/`, `dct/`, or `dwt/`.
- Logic must be ported by rewrite, not via re-export/import.
- Unknown / missing pieces must be called out explicitly; do NOT invent values.

What exists in this workspace (source of truth)

A) DCT baseline (Cox Eq. (2) style)
- Main reference implementation (paper reproduction): `dct/reproduce_dct_paper.py`
  - `embed_watermark(image, n, alpha, rng, watermark=None) -> WatermarkedImage(image, watermark, rows, cols)`
  - `extract_watermark(original_image, candidate_image, rows, cols, alpha, eps=1e-10) -> np.ndarray`
  - `similarity(original_mark, extracted_mark) -> float` (dot / ||extracted||)
  - Uses 2D DCT (`dct2`/`idct2`) + top-magnitude coefficient indices (excluding DC).

- WAVES-aligned bit-sign variant used in WAVES scripts:
  - `waves/utils/dct_utils.py`
    - `generate_ground_truth_bits(length, seed) -> bool[length]`
    - `embed_bits(image, bits, alpha) -> image` with sign-only payload (+1/-1)
    - `extract_bits(original_image, candidate_image, length, alpha) -> bool[length]`
  - WAVES DCT parameters: `waves/dev/constants.py` (`DCT_WATERMARK_LEN=1000`, `DCT_WATERMARK_SEED=42`, `DCT_WATERMARK_ALPHA=0.1`).
  - WAVES DCT decode semantics (non-blind): `waves/scripts/decode.py::process_dct`
    - Loads original grayscale from real dir (derived from `DATA_DIR`).
    - Resizes attacked candidate to original size if needed (bicubic).
    - Calls `extract_bits(...)`.

B) DWT baseline (Xia 1998-ish; notebook-only)
- Implementation lives only in `dwt/dwt_watermark_xia1998.ipynb`.
  - `embed_watermark_dwt(image, alpha=0.04, levels=2, wavelet='haar', seed=1234, largest_fraction=0.10)`
    - Performs multilevel 2D DWT, adds watermark only to “large” detail coeffs (outside LL), with:
      y~ = y + alpha * y^2 * N   (mask selects top `largest_fraction` by magnitude per subband)
    - Reconstructs, crops to original shape, clips to original min/max.
    - Returns `(watermarked_image, payload)` where payload contains `watermark_signal` per (level, band).
  - `detect_watermark_hierarchical(original_image, received_image, payload, ratio_threshold=1.05)`
    - For each level, tests band schedule HH -> HH+LH -> HH+LH+HL.
    - Computes normalized 2D xcorr of (received - original) against watermark_signal.
    - Computes peak ratio (largest / second-largest) and averages across selected bands.
    - Detects when mean_peak_ratio >= threshold.

C) WAVES attacks
- Distortion attacks are implemented in `waves/distortions/distortions.py` and invoked by
  `waves/scripts/apply_distortion_attacks_flat.py`.
- Attacks include singles and combos:
  - singles: rotation, resizedcrop, erasing, brightness, contrast, blurring, noise, jpeg
  - combos (ordered):
    - geometric: rotation -> resizedcrop
    - photometric: brightness -> contrast
    - degradation: blurring -> compression
    - all: rotation -> brightness -> noise -> blurring -> compression
- Strength handling:
  - WAVES code largely uses a single `--relative-strength` (default 0.5) which is converted
    to absolute params via `relative_strength_to_absolute(...)`.
  - No fixed per-attack “strength grid” list was found in code; leaderboard tooling suggests
    strengths are discovered from result directories (data-driven), not hard-coded.

D) WAVES regeneration attacks
- Implemented: `waves/regeneration/regen.py` includes `regen_diff`, `regen_vae`, `rinse_2xDiff`, `rinse_4xDiff`.
- Referenced in `waves/dev/constants.py` but missing in code: `regen_diffusion_prompt`, `kl_vae`.
  - These must be treated as unsupported/missing unless the exact implementation is provided.

E) WAVES quality metrics (8+)
- Implementations:
  - PSNR/SSIM/NMI: `waves/metrics/image.py`
  - LPIPS: `waves/metrics/perceptual.py`
  - FID + CLIP-FID: `waves/metrics/distributional.py` (CLIP-FID uses mode='clip')
  - Aesthetics + Artifacts models: `waves/metrics/aesthetics.py` (also provides CLIP score)
- Metric runner + aggregation behavior:
  - Per-image metrics JSON: `waves/scripts/metric.py`
  - Aggregation to mean/std tables: `waves/dev/aggregate.py`
  - For aesthetics/artifacts/clip_score, WAVES records deltas as clean - attacked.

F) “Q normalization” / anchors
- Searched in `waves/` for p10/p90/anchors/normalize logic; none found in this repo version.
- Conclusion (based on repo contents): WAVES computes raw metrics and aggregates mean/std;
  any “Q normalization” described in a paper is not implemented here.
- Action for `wmbench/`: do NOT invent normalization. If needed, require an explicit formula/spec.

Open questions / blockers (must not guess)
1) Attack strength grids: should `wmbench/` (a) accept a user-provided grid config, or (b) discover strengths
   from existing result dirs (WAVES-style)? No hard-coded lists exist in repo.
2) Q normalization: if required, please provide the exact paper excerpt/formula and inversion rules.

End scratchpad.
"""

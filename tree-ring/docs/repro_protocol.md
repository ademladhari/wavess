# Reproduction Protocol

This protocol follows the Tree-Ring paper workflow at a practical level.

## Baseline configuration

- Model: Stable Diffusion 2.1 base (`DDIM` scheduler).
- Generation steps: 50.
- Inversion steps: 50.
- Guidance scale: 7.5.
- Watermark radius: 10 (ablate up to 16).
- Key type: `rand` or `rings` for p-value testing.

## Dataset generation

1. Prepare a prompt file with one prompt per line.
2. Generate watermarked images.
3. Generate an equal-sized clean set using the same prompts.

## Detection

1. Optionally apply one attack per run:
   - `rotation` (e.g., `angle=75`)
   - `jpeg` (e.g., `quality=25`)
   - `crop_rescale` (e.g., `crop_ratio=0.75`)
   - `gaussian_blur` (e.g., `radius=2`)
   - `gaussian_noise` (e.g., `sigma=0.1`)
   - `color_jitter` (e.g., `brightness=6`)
2. Invert each image back to estimated initial latent with DDIM inversion.
3. Compute both detection scores:
   - distance score (`d_detection`),
   - p-value score.

## Metrics

- ROC curve and AUC for distance score.
- ROC curve and AUC for p-value score.
- TPR@1%FPR for both scores.

## Suggested ablations

- Radius sweep: `r in {6, 8, 10, 12, 16}`.
- Key variant sweep: `zeros`, `rand`, `rings`.
- Generation/inversion step mismatch.
- Guidance scale sweep.

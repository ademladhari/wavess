# Tree-Ring Diffusers Reproduction

This project implements a Tree-Ring watermarking workflow for latent diffusion models using
PyTorch and Hugging Face Diffusers.

## What is included

- Fourier-domain watermark embedding into initial noise (`x_T`).
- Three key variants: `zeros`, `rand`, and `rings`.
- DDIM generation wrapper with custom latent injection.
- DDIM inversion-based detection with:
  - masked L1 distance score (paper Eq. 3),
  - non-central chi-square p-value score (paper Eq. 5/6).
- Attack harness (rotation, JPEG, crop+rescale, blur, noise, color jitter).
- Evaluation script for ROC AUC and TPR@target FPR.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Run pipeline

Generate watermarked images:

```bash
python scripts/run_generate.py --config configs/base.yaml --watermarked --num-samples 16 --output-dir outputs/wm
```

Generate clean images:

```bash
python scripts/run_generate.py --config configs/base.yaml --num-samples 16 --output-dir outputs/clean
```

Detect watermarks:

```bash
python scripts/run_detect.py --config configs/base.yaml --manifest outputs/wm/manifest.json --output outputs/wm_detect.json
python scripts/run_detect.py --config configs/base.yaml --manifest outputs/clean/manifest.json --output outputs/clean_detect.json
```

Evaluate:

```bash
python scripts/run_eval.py --config configs/base.yaml --watermarked-results outputs/wm_detect.json --clean-results outputs/clean_detect.json --output-dir outputs/eval
```

## Notes

- For Stable Diffusion 2.1 base, latent shape defaults to `4 x 64 x 64`.
- Keep generation and inversion schedulers aligned for best reconstruction.
- `key_variant: rand` or `rings` is recommended for the p-value detector.

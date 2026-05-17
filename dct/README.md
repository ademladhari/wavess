# Faithful Reproduction: Cox et al. (1997) DCT Watermarking

This repository contains an executable reproduction of the core method and experiments from:

- **I. J. Cox, J. Kilian, F. T. Leighton, T. Shamoon**  
  *Secure Spread Spectrum Watermarking for Multimedia* (1997)

## What is reproduced

- Frequency-domain watermarking using **global 2D DCT**
- Watermark vector sampled from **iid Gaussian** noise
- Embedding into the **top-magnitude DCT coefficients** (excluding DC)
- Embedding rule matching paper Eq. (2):  
  `x_i* = x_i * (1 + alpha * w_i)`
- Non-blind extraction using the original image and similarity detector:
  `sim(w, w_hat) = (w · w_hat) / ||w_hat||`
- Attack/evaluation suite aligned to the paper:
  - uniqueness vs random watermarks
  - scale down/up
  - JPEG quality 10 and 5
  - dithering
  - crop + restore with original image
  - repeated watermarking
  - collusion by averaging independently watermarked copies

## Setup

Uses Python 3.10+ and these libraries:

- `numpy`
- `scipy`
- `pillow`
- `matplotlib`

If needed:

```powershell
py -m pip install numpy scipy pillow matplotlib
```

## Run

Place a grayscale or RGB image in `dataset/` (or pass `--image`).

```powershell
py reproduce_dct_paper.py --dataset-dir dataset --out-dir outputs --watermark-length 1000 --alpha 0.1 --seed 42
```

Optional explicit image:

```powershell
py reproduce_dct_paper.py --image dataset\your_image.png
```

## Output

Created under `outputs/`:

- `original.png`
- `watermarked.png`
- experiment figures (`exp1_*.png`, ..., `exp8_*.png`)
- `report.json` with detector scores and run parameters

## Notes on fidelity

- The implementation follows the paper's algorithmic choices (`n=1000`, `alpha=0.1` by default).
- Experiment 6 in the paper uses a physical print/xerox/scan chain. This code includes a **digital proxy** for that step unless you provide a true scanned image.
- Exact numeric results depend on host image and attack toolchain; trend-level behavior should match the paper.

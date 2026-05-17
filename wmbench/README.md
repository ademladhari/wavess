# wmbench — Clean Watermark Benchmarking Framework

A self-contained benchmarking framework for image watermarking detection and evaluation, with **zero runtime imports** from `waves/`, `dct/`, or `dwt/`.

## End-to-end benchmark (`run_benchmark.py`)

```bash
py -3 wmbench/run_benchmark.py --methods dct dwt --images ./data/test_images/ --negatives ./data/clean_images/ --output ./results/
```

Writes `results/work/{method}/` (watermarked, attacked, scores, metrics), `results/results_raw.csv`, `results/results_leaderboard.csv`, `results/normalization_anchors.json`, and `results/plots/`. Optional: `--strength-config`, `--attacks`, `--resume`, `--diffusion-model`, `--vae-model`, `--diffusion-attack-batch-size`, `--lpips-batch-size`, `--profile-stages`. Missing upstream attacks (`Regen-DiffP`, `Regen-KLVAE`) are logged under `results/missing_components.txt`. Provenance notes: docstring at the top of `wmbench/run_benchmark.py`.

### 22GB VRAM tuning profile (same workload)

Use larger attack/metric batches without changing attack grids:

```bash
py -3 wmbench/run_benchmark.py \
  --methods dct \
  --images D:/images \
  --negatives D:/negatives \
  --output D:/results \
  --device cuda \
  --diffusion-attack-batch-size 4 \
  --lpips-batch-size 16 \
  --profile-stages \
  --resume
```

Increase `--diffusion-attack-batch-size` until VRAM is near full but stable; if you hit OOM, lower it by 1 step.

## Architecture

```
wmbench/
├── __init__.py              # Package entry point
├── app.py                   # (placeholder for potential web/UI)
├── cli.py                   # Main CLI entry point
├── dev/                     # Development utilities
│   ├── constants.py         # ATTACK_NAMES, GROUND_TRUTH_MESSAGES, etc.
│   ├── eval.py              # ROC-based detection metrics
│   ├── find.py              # Directory/dataset discovery
│   ├── io.py                # JSON + array encoding/decoding
│   └── aggregate.py         # (future) result aggregation
├── utils/                   # Core utilities
│   ├── dct_utils.py         # DCT embedding/extraction
│   ├── image_utils.py       # Image I/O + formatting
│   └── (other utils)
├── distortions/             # Attack operators
│   └── distortions.py       # Strength mapping + single/combo distortions
├── methods/                 # Watermarking methods
│   ├── dwt_xia1998.py       # DWT Xia1998-style watermark
│   └── (other methods)
├── metrics/                 # Quality metrics (SELF-CONTAINED COPIES)
│   ├── image.py             # PSNR, SSIM, NMI, MSE (scikit-image)
│   ├── aesthetics.py        # Aesthetics + artifacts (CLIP-based)
│   ├── clip.py              # CLIP score (OpenCLIP)
│   ├── perceptual.py        # LPIPS + Watson wrappers
│   ├── distributional.py    # FID (via clean_fid)
│   ├── aesthetics_scorer/   # Vendored: weight loader + model
│   ├── lpips/               # Vendored: full LPIPS module
│   ├── watson/              # Vendored: full Watson module
│   ├── clean_fid/           # Vendored: full FID module
│   └── __init__.py          # Metric exports
├── scripts/                 # Executable runners
│   ├── decode.py            # WAVES-style decode (ONNX stega stamp / stable sig)
│   ├── metric.py            # Compute image quality metrics
│   ├── embed_dct.py         # Embed DCT watermarks
│   ├── apply_distortions.py # Apply distortion attacks
│   └── (future: CLI wrappers)
├── regeneration/            # (future) diffusion-based regen
├── scripts/                 # Additional runners
├── decoders/                # Decoder models (ONNX)
│   ├── stable_signature.onnx
│   └── stega_stamp.onnx
└── _scratchpad.py           # (MAPPING NOTES ONLY — non-executable)
```

## Key Features

### 1. **Zero Upstream Imports**
- All code is rewritten or vendored; no `import waves`, `import dct`, `import dwt` at runtime
- Weights and model dependencies (LPIPS, Watson, FID, aesthetics_scorer) are copied locally
- Fallback search for weights respects `MODEL_DIR` and `WMBENCH_AESTHETICS_WEIGHTS_DIR` env vars

### 2. **Faithfully Ported Functionality**
- **DCT Embedding**: Bit-sign embedding in top K DCT magnitudes (`wmbench/utils/dct_utils.py`)
- **DWT Watermarking**: Hierarchical watermark embed/extract (Xia1998-ish, `wmbench/methods/dwt_xia1998.py`)
- **Distortion Attacks**: Geometry + photometry + degradation + combinations (via `wmbench/distortions/distortions.py`)
- **Decode**: ONNX-based StegaStamp + Stable Signature detection (CUDA + CPU fallback)
- **Metrics**: PSNR/SSIM/LPIPS/Watson/FID/CLIP-score/aesthetics/artifacts

### 3. **Environment Variables** (Optional)
```bash
DATA_DIR=/path/to/images           # Input image directory
RESULT_DIR=/path/to/results        # Output results directory
MODEL_DIR=/path/to/models          # Model weights directory (fallback for aesthetics, etc.)
WMBENCH_AESTHETICS_WEIGHTS_DIR=/path  # Aesthetics weights override
```

### 4. **CLI Runners**

#### Decode (ONNX-based detection)
```bash
python -m wmbench.scripts.decode psnr --data-dir /path/to/attacks
```

#### Metrics (Image quality + detection robustness)
```bash
python -m wmbench.scripts.metric psnr --result-dir /path/to/results
python -m wmbench.scripts.metric clip_fid --result-dir /path/to/results
python -m wmbench.scripts.metric clip_score --result-dir /path/to/results
```

#### Embed DCT
```bash
python -m wmbench.scripts.embed_dct --dataset diffusiondb
```

#### Apply Distortions
```bash
python -m wmbench.scripts.apply_distortions \
  --data-dir /data \
  --source-subdir dct \
  --attack-suite 17 \
  --relative-strength 0.5
```

## Implementation Notes

### Missing from Upstream Repo
Per the scratchpad (`_scratchpad.py`):
- **Strength grids**: Per-attack relative-to-absolute strength mappings not formalized in upstream; using best-guess formulas or `relative_strength_to_absolute()` defaults
- **Q-normalization**: Mentioned in papers but not implemented in upstream codebase
- **Regen methods**: `regen_diffusion_prompt`, `kl_vae` not implemented; regenerative attacks must be provided separately

### Known Issues
1. **FID on CPU**: ONNX runtime may lack GPU kernels on some systems. Fallback to CPU is implemented but slow.
2. **Watson weights**: If Watson dependencies not installed, `LossProvider` may not load. Graceful error provided.
3. **Aesthetics weights search**: Searches multiple fallback paths; ensure weights are in one of them or set `WMBENCH_AESTHETICS_WEIGHTS_DIR`.

## Supported Attacks & Metrics

### Attacks (from `wmbench/dev/constants.py`)
- `distortion_single_*`: rotation, resizedcrop, erasing, brightness, contrast, blurring, noise, jpeg
- `distortion_combo_*`: geometric, photometric, degradation, all
- `stega_stamp`, `stable_sig`: ONNX-based decoding targets

### Metrics
- **Image Distance**: psnr, ssim, nmi, mse
- **Perceptual**: lpips, watson
- **Distributional**: legacy_fid, clean_fid, clip_fid
- **Semantic**: clip_score, aesthetics, artifacts

## Configuration

### Constants (wmbench/dev/constants.py)
```python
LIMIT = 10000              # Default max images
SUBSET_LIMIT = 100         # Subset for quick testing
DCT_WATERMARK_LEN = 128    # DCT watermark bit length
DCT_WATERMARK_SEED = 42    # Deterministic seed
DCT_WATERMARK_ALPHA = 0.1  # Embedding strength
```

## Example Workflow

```bash
# 1. Set environment
export DATA_DIR=/data/images
export RESULT_DIR=/results
export MODEL_DIR=/models

# 2. Embed DCT watermarks
python -m wmbench.scripts.embed_dct --dataset diffusiondb

# 3. Apply distortion attacks
python -m wmbench.scripts.apply_distortions \
  --data-dir $DATA_DIR \
  --source-subdir dct \
  --relative-strength 0.5

# 4. Decode (detect watermark presence)
python -m wmbench.scripts.decode psnr --data-dir $DATA_DIR

# 5. Compute metrics
python -m wmbench.scripts.metric psnr --result-dir $RESULT_DIR
python -m wmbench.scripts.metric clip_score --result-dir $RESULT_DIR
python -m wmbench.scripts.metric aesthetics_and_artifacts --result-dir $RESULT_DIR
```

## Dependencies

### Core
- `numpy`, `scipy`, `scikit-image`
- `torch`, `torchvision`
- `PIL`, `click`, `tqdm`

### Optional (for specific metrics)
- `open_clip`, `transformers` (CLIP + aesthetics)
- `onnxruntime` (ONNX decode)
- `clean_fid` (Inception/CLIP FID)
- `lpips` (LPIPS metric)
- `Watson` (Watson perceptual metric) — may require extra deps

## Project Notes

### Provenance
All code is hand-rewritten or vendored without modification from `waves/`, `dct/`, `dwt/` source trees.
See `_scratchpad.py` for detailed mapping of what was ported and what remains unimplemented.

### Testing
- All Python files pass `py_compile` syntax check
- `import wmbench` succeeds without errors
- No integration tests yet; recommend manual validation on sample datasets
- Baseline-vs-optimized drift check helper: `python -m wmbench.dev.compare_results --baseline <old/results_raw.csv> --candidate <new/results_raw.csv> --tol 1e-4`

### Future Work
- [ ] Regen methods (diffusion-based attacks)
- [ ] Full integration test suite
- [ ] CLI entry points in setup.py
- [ ] Comprehensive logging + debugging
- [ ] Performance optimization (batching, GPU acceleration)

---

**Last Updated**: 2025 (Post-port verification complete)

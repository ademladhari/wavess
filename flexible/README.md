# Strict reproduction: Flexible and Lightweight Watermarking Framework for SDMs

Faithful replication of

> H. Luo, L. Li, X. Zhang, *"Flexible and Lightweight Watermarking Framework
> for Stable Diffusion Models"*, IEEE Internet of Things Journal, vol. 13,
> no. 7, Apr. 2026, pp. 13950-13963.

Scope: SDM V2.1 + `Gustavosta/Stable-Diffusion-Prompts`, all 8 traditional
image-processing attacks + 3 generative attacks, full training pipeline (E,
D, Dext, Ddec). The IoT FastAPI serving benchmark, collusion attack, and
model-editing attack are explicitly out of scope.

## Environment (Windows / PowerShell, RTX 5060 8 GB)

```powershell
# One-shot setup. Creates ./.venv, installs torch (cu124) + deps.
./setup_env.ps1

# If your driver/toolchain needs PyTorch nightly for sm_120 (Blackwell):
$env:FLEX_TORCH_NIGHTLY = "1"; ./setup_env.ps1
```

Verify CUDA:

```powershell
. .\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The first SDM run will download `stabilityai/stable-diffusion-2-1` into
`./hf_cache` (~5 GB). Accept the HuggingFace license via `huggingface-cli login`
if required.

## End-to-end reproduction

```powershell
# 1. Pretrain watermark encoder E + decoder D (Stage 1).
./scripts/01_pretrain_ed.ps1 -Bits 48,24,16

# 2. Pre-generate (w, zT, Iw) pool with frozen SDM V2.1.
./scripts/02_generate_pairs.ps1 -Bits 48,24,16

# 3. Train Dext (Transformer + MLP). Add -IncludeResNet18 for Table V.
./scripts/03_train_extractor.ps1 -Bits 48,24,16 -IncludeResNet18

# 4. Fine-tune D with Dext frozen -> Ddec.
./scripts/04_finetune_decoder.ps1 -Bits 48,24,16

# 5. Evaluate (each script writes a JSON to outputs/eval/).
./scripts/05_eval_fidelity.ps1            # Table I
./scripts/07_eval_normality.ps1           # Table II
./scripts/06_eval_attacks.ps1 -Mode fixed # Table III
./scripts/06_eval_attacks.ps1 -Mode swept # Figure 5
./scripts/06_eval_attacks.ps1 -Mode generative  # Figure 6
./scripts/08_eval_extractor_compare.ps1   # Table V subset
```

## Paper -> code map

| Paper item | Code |
|---|---|
| Watermark Encoder E (eq. 2) | [src/models/encoder.py](src/models/encoder.py) |
| Watermark Decoder D (eq. 3) | [src/models/decoder.py](src/models/decoder.py) |
| Watermark Extractor Dext (eq. 5) | [src/models/extractor.py](src/models/extractor.py) |
| Frozen SDM V2.1 + DPMSolver 25 steps | [src/models/sdm.py](src/models/sdm.py) |
| Alg. 1 step 1 (train Dext) | [src/train/train_extractor.py](src/train/train_extractor.py) |
| Alg. 1 step 2 (fine-tune D) | [src/train/finetune_decoder.py](src/train/finetune_decoder.py) |
| Stage 1 (pretrain E+D, eq. 4) | [src/train/pretrain_ed.py](src/train/pretrain_ed.py) |
| 8 image-processing attacks (Fig. 4) | [src/attacks/image.py](src/attacks/image.py) |
| Bmshj18 / Cheng20 / Zhao23 | [src/attacks/generative.py](src/attacks/generative.py) |
| BitAcc + TPR@0.01FPR | [src/eval/extraction.py](src/eval/extraction.py) |
| FID + NIQE + PIQE + CLIP | [src/eval/fidelity.py](src/eval/fidelity.py) |
| Normality tests (Table II) | [src/eval/normality.py](src/eval/normality.py) |
| ResNet18 baseline (Table V) | [src/models/extractor_resnet18.py](src/models/extractor_resnet18.py) |
| All hyperparameters | [configs/default.yaml](configs/default.yaml) |

## Hyperparameters (from the paper)

Kept identical to Sec. IV-A.4 and Alg. 1:

* Adam optimizer, `lr = 1e-5`, `batch_size = 2`
* `lambda1 = lambda2 = 1`
* DPMSolver-Multistep, 25 inference steps
* Classifier-free guidance scale 7.5
* Image size 512x512; latent 4 x 64 x 64
* Watermark capacities 16 / 24 / 48 bits; TPR threshold for 48 bits is 33

## Memory strategy on 8 GB VRAM

The paper was trained on a single RTX 3090 (24 GB). On 8 GB we keep every
optimizer setting identical and only change the data-loading pattern:

1. `pretrain_ed` trains only E and D, never loads SDM -> fits easily.
2. `generate_pairs` loads SDM 2.1 in fp16 with `enable_attention_slicing()`
   and `enable_vae_slicing()`, runs inference-only, and stores
   `(w, zT, Iw)` tuples as .pt files on disk.
3. `train_extractor` and `finetune_decoder` consume the pre-generated pool
   without ever loading SDM -> trivially fits at batch size 2.

Disk footprint: 8 000 train + 500 val pairs at fp16 / uint8 is about 6-12 GB
depending on capacity.

## Known assumptions vs. paper

Items the paper does not specify; fixed to the smallest plausible values,
documented inline in `configs/default.yaml`:

* Training epoch counts for Stage 1 / Stage 2 -- paper only reports
  optimizer + batch size. We train to a convergence threshold
  (`target_L_r = 1e-3`, `target_L_d = 1e-2`) with a hard cap of 100 k /
  80 k / 40 k steps respectively.
* Convolution channel widths and Transformer token dimension inside E, D,
  Dext -- Fig. 3 only labels block counts. We use 64 base channels, 128
  token dim, 256 MLP/FFN hidden.
* Size of the pre-generated training pool: 8000 (large enough that each
  Stage 2 step still sees fresh random watermarks).

## Out-of-scope items

Explicitly skipped per the reproduction brief:

* **Sec. IV-F** IoT FastAPI + Locust serving benchmark (Figs. 8-9).
* **Sec. IV-E** collusion attack (Fig. 7).
* **Sec. IV-D** model-editing attacks (Table IV).
* **Table VII** SDM V1.4 / V1.5 / V2.0 generalization ablation.
* **Table VIII / IX** inference-step / guidance-scale ablations.

Re-enabling any of these only requires loading additional SDM variants and
rerunning `scripts/05_eval_fidelity.ps1` with the corresponding checkpoint
IDs -- no changes to training are needed because Dext is designed to be
SDM-version-agnostic.

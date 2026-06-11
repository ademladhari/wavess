# Local CPU benchmark: classical watermark methods + distortion attacks only.
# Usage:  .\.venv\Scripts\Activate.ps1; .\run_cpu_classical.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUNBUFFERED = "1"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing venv. Run setup from repo root first (see run_flexible_gpu.ps1 / chat)."
}

& $python wmbench/run_benchmark.py `
  --methods dct dwt svd dct-dwt `
  --images "D:\dataset\images" `
  --negatives "D:\dataset\negatives" `
  --output "$PSScriptRoot\wmbench_cpu_output" `
  --strength-config "$PSScriptRoot\strengths_fast.json" `
  --device cpu `
  --embed-batch-size 16 `
  --lpips-batch-size 32 `
  --attacks Dist-Rotation Dist-RCrop Dist-Erase Dist-Bright Dist-Contrast `
            Dist-Blur Dist-Noise Dist-JPEG `
            DistCom-Geo DistCom-Photo DistCom-Deg DistCom-All `
  --blind-detect `
  --skip-aesthetics-metrics `
  --resume `
  --profile-stages

# Local flexible GPU benchmark (regen attacks only).
# Usage:  .\.venv\Scripts\Activate.ps1; .\run_flexible_gpu.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:WMBENCH_FLEXIBLE_ROOT = "$PSScriptRoot\flexible"
$env:WMBENCH_FLEXIBLE_CHECKPOINTS = "$PSScriptRoot\flexible\checkpoints"
$env:WMBENCH_FLEXIBLE_PROMPT = "a photo"
$env:PYTHONUNBUFFERED = "1"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing venv. Run: py -m venv .venv; then install deps (see README in chat)."
}

& $python wmbench/run_benchmark.py `
  --methods flexible `
  --images "$PSScriptRoot\wmbench_data\images" `
  --negatives "$PSScriptRoot\wmbench_data\negatives" `
  --output "$PSScriptRoot\wmbench_gpu_output\flexible" `
  --device cuda `
  --embed-batch-size 4 `
  --diffusion-attack-batch-size 2 `
  --lpips-batch-size 16 `
  --attacks Regen-Diff Rinse-2xDiff `
  --skip-rinse4xdiff `
  --blind-detect `
  --skip-aesthetics-metrics `
  --resume `
  --profile-stages

# Strict reproduction env setup for Windows / PowerShell.
# Creates a local venv (./.venv), installs PyTorch with CUDA 12.4 support
# (required for RTX 5060 / Blackwell sm_120), then installs the rest from
# requirements.txt.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "[setup] Creating venv at $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "[setup] Reusing existing venv at $venvPath"
}

$activate = Join-Path $venvPath "Scripts\Activate.ps1"
. $activate

Write-Host "[setup] Upgrading pip / wheel / setuptools"
python -m pip install --upgrade pip wheel setuptools

Write-Host "[setup] Installing PyTorch with CUDA 12.4 (sm_120 supported from 2.7+)"
# Stable channel first; if a wheel for sm_120 is missing on your runtime, rerun
# with: $env:FLEX_TORCH_NIGHTLY = "1"; ./setup_env.ps1
if ($env:FLEX_TORCH_NIGHTLY -eq "1") {
    Write-Host "[setup] Using PyTorch NIGHTLY (cu124)"
    pip install --pre --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
} else {
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

Write-Host "[setup] Installing remaining requirements"
pip install -r requirements.txt

Write-Host "[setup] Verifying CUDA / device capability"
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('cap:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

Write-Host ""
Write-Host "[setup] Done. Activate the env with:  . .\.venv\Scripts\Activate.ps1"

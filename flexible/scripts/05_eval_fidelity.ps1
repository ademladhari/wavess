param(
    [int[]]$Bits = @(48, 24, 16),
    [int]$NPrompts = 1000,
    [string]$Config = "configs/default.yaml"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. .\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = $repo
python -m src.eval.eval_fidelity --config $Config --bits @Bits --n-prompts $NPrompts

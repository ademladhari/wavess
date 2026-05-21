param(
    [int]$Bits = 48,
    [int]$NSamples = 5000,
    [string]$Config = "configs/default.yaml"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. .\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = $repo
python -m src.eval.eval_normality --config $Config --bits $Bits --n-samples $NSamples

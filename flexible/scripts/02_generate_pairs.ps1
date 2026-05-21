param(
    [int[]]$Bits = @(48, 24, 16),
    [string]$Config = "configs/default.yaml"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. .\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = $repo
foreach ($b in $Bits) {
    Write-Host "[scripts] generate pairs (train) for $b-bit"
    python -m src.generate_pairs --config $Config --bits $b --split train
    Write-Host "[scripts] generate pairs (val) for $b-bit"
    python -m src.generate_pairs --config $Config --bits $b --split val
}

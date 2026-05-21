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
    Write-Host "[scripts] pretrain E+D for $b-bit"
    python -m src.train.pretrain_ed --config $Config --bits $b
}

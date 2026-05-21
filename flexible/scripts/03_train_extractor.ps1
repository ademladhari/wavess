param(
    [int[]]$Bits = @(48, 24, 16),
    [string]$Config = "configs/default.yaml",
    [switch]$IncludeResNet18
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. .\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = $repo
foreach ($b in $Bits) {
    Write-Host "[scripts] train Dext (Transformer+MLP) for $b-bit"
    python -m src.train.train_extractor --config $Config --bits $b
    if ($IncludeResNet18) {
        Write-Host "[scripts] train Dext (ResNet18 baseline) for $b-bit"
        python -m src.train.train_extractor_resnet18 --config $Config --bits $b
    }
}

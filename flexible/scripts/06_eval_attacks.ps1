param(
    [int]$Bits = 48,
    [string]$Config = "configs/default.yaml",
    [ValidateSet("fixed", "swept", "generative", "all")]
    [string]$Mode = "all",
    [int]$N = 500
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. .\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = $repo

$modes = if ($Mode -eq "all") { @("fixed", "swept", "generative") } else { @($Mode) }
foreach ($m in $modes) {
    Write-Host "[scripts] eval attacks mode=$m bits=$Bits"
    python -m src.eval.eval_attacks --config $Config --bits $Bits --mode $m --n $N
}

param([string]$Project = (Get-Location).Path)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path $Project).Path

$Required = @(
    "app.py",
    "requirements.txt",
    ".streamlit\config.toml",
    "src\data_loader.py",
    "src\metrics.py",
    "src\spatial.py",
    "data\economic_returns_1981_2018.csv",
    "data\processed\cropping_systems_long.csv",
    "data\processed\cropping_systems_spatial.parquet"
)

Write-Host "===================================================================="
Write-Host "GITHUB / STREAMLIT DEPLOYMENT READINESS CHECK"
Write-Host "===================================================================="
Write-Host "Project: $Project"

$Missing = @()
foreach ($Relative in $Required) {
    $Path = Join-Path $Project $Relative
    if (-not (Test-Path $Path)) {
        $Missing += $Relative
    }
}
if ($Missing.Count -gt 0) {
    Write-Host "FAIL: Missing required files:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

$Spatial = Get-Item (Join-Path $Project "data\processed\cropping_systems_spatial.parquet")
$Long = Get-Item (Join-Path $Project "data\processed\cropping_systems_long.csv")
Write-Host "Required files: PASS" -ForegroundColor Green
Write-Host ("Spatial Parquet: {0:N2} MB" -f ($Spatial.Length / 1MB))
Write-Host ("State summary CSV: {0:N2} MB" -f ($Long.Length / 1MB))

if ($Spatial.Length -ge 100MB) {
    Write-Host "Git LFS required: YES (file is at least 100 MB)" -ForegroundColor Yellow
} elseif ($Spatial.Length -ge 50MB) {
    Write-Host "Git LFS recommended: YES (large binary file)" -ForegroundColor Yellow
} else {
    Write-Host "Git LFS recommended: optional; the kit tracks Parquet with LFS for reliability." -ForegroundColor Cyan
}

$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$CheckCode = @'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
spatial = pd.read_parquet(root / "data" / "processed" / "cropping_systems_spatial.parquet")
summary = pd.read_csv(root / "data" / "processed" / "cropping_systems_long.csv")
required_spatial = {"year","base_system","water_regime","site_id","latitude","longitude","yield_kg_ha"}
missing = sorted(required_spatial - set(spatial.columns))
if missing:
    raise SystemExit(f"Spatial file missing columns: {missing}")
print(f"Spatial rows: {len(spatial):,}")
print(f"Unique spatial sites: {spatial['site_id'].nunique():,}")
print(f"Spatial systems: {spatial['base_system'].nunique():,}")
print(f"Spatial years: {spatial['year'].min()}-{spatial['year'].max()}")
print(f"State-summary rows: {len(summary):,}")
if spatial['site_id'].nunique() != 2776:
    raise SystemExit("Expected 2,776 unique spatial sites.")
print("Data validation: PASS")
'@

& $Python -c $CheckCode $Project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Deployment readiness: PASS" -ForegroundColor Green

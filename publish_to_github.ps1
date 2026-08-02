param(
    [Parameter(Mandatory=$true)]
    [string]$RepositoryUrl,
    [string]$Branch = "main",
    [string]$CommitMessage = "Publish Kansas cropping systems dashboard"
)

$ErrorActionPreference = "Stop"
$Project = (Get-Location).Path

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or not available in PATH. Install Git for Windows or use GitHub Desktop."
}

& "$Project\check_deployment_readiness.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Spatial = Get-Item "$Project\data\processed\cropping_systems_spatial.parquet"
$UseLfs = $Spatial.Length -ge 50MB
if ($UseLfs) {
    if (-not (git lfs version 2>$null)) {
        throw "Git LFS is required/recommended for the spatial Parquet file but is not installed. Install Git LFS, then rerun."
    }
    git lfs install
    git lfs track "*.parquet"
}

if (-not (Test-Path "$Project\.git")) {
    git init
}

git branch -M $Branch
if ((git remote) -contains "origin") {
    git remote set-url origin $RepositoryUrl
} else {
    git remote add origin $RepositoryUrl
}

git add .
$Status = git status --porcelain
if ($Status) {
    git commit -m $CommitMessage
} else {
    Write-Host "No new changes to commit."
}

git push -u origin $Branch
Write-Host "GitHub push completed." -ForegroundColor Green

# Environment setup for the Diffusion Artwork project.
# Target: RTX 5090 / Blackwell sm_120 / Windows 11.
# Installs: uv -> managed Python 3.12 -> project .venv -> PyTorch (cu128) -> ML deps -> verifies GPU.
# Safe to re-run (idempotent). Full transcript saved to environment/setup.log.

$ErrorActionPreference = 'Stop'
$ProjRoot = Split-Path $PSScriptRoot -Parent
$log = Join-Path $PSScriptRoot 'setup.log'
Start-Transcript -Path $log -Append | Out-Null

function Check([string]$what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

try {
    Write-Host "=== [1/6] Install uv (via the existing Python 3.14 pip) ==="
    & py -m pip install --upgrade --quiet pip uv; Check 'pip install uv'
    $pyexe = (& py -c "import sys;print(sys.executable)").Trim()
    $uv = Join-Path (Split-Path $pyexe) 'Scripts\uv.exe'
    if (-not (Test-Path $uv)) { throw "uv.exe not found at $uv" }
    Write-Host "uv -> $uv"
    & $uv --version

    Write-Host "=== [2/6] Install managed Python 3.12 ==="
    & $uv python install 3.12; Check 'uv python install 3.12'

    Write-Host "=== [3/6] Create project venv (.venv) on Python 3.12 ==="
    $venvDir = Join-Path $ProjRoot '.venv'
    & $uv venv -p 3.12 $venvDir; Check 'uv venv'
    $venvPy = Join-Path $venvDir 'Scripts\python.exe'

    Write-Host "=== [4/6] Install PyTorch + torchvision (CUDA 12.8 / Blackwell) ==="
    & $uv pip install -p $venvPy torch torchvision --index-url https://download.pytorch.org/whl/cu128
    Check 'torch (cu128) install'

    Write-Host "=== [5/6] Install ML dependencies ==="
    & $uv pip install -p $venvPy -r (Join-Path $PSScriptRoot 'requirements.txt')
    Check 'requirements install'

    Write-Host "=== [6/6] Verify GPU ==="
    & $venvPy (Join-Path $PSScriptRoot 'verify_gpu.py'); Check 'verify_gpu'

    Write-Host "SETUP_DONE rc=0"
}
catch {
    Write-Host "SETUP_FAILED: $($_.Exception.Message)"
    throw
}
finally {
    Stop-Transcript | Out-Null
}

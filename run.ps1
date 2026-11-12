# ==============================================================================
# GridPulse AI - PowerShell Launcher
# ==============================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "    ⚡ Starting GridPulse AI Intelligent Prediction Dashboard   " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check for local virtual environment
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $ScriptDir ".venv\Scripts\streamlit.exe"

if (Test-Path $VenvStreamlit) {
    Write-Host "[INFO] Launching using project virtual environment..." -ForegroundColor Green
    & $VenvStreamlit run app.py --server.port 8501
} elseif (Get-Command streamlit -ErrorAction SilentlyContinue) {
    Write-Host "[INFO] Launching using system Streamlit..." -ForegroundColor Green
    streamlit run app.py --server.port 8501
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "[INFO] Launching using 'py -m streamlit'..." -ForegroundColor Green
    py -m streamlit run app.py --server.port 8501
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[INFO] Launching using 'python -m streamlit'..." -ForegroundColor Green
    python -m streamlit run app.py --server.port 8501
} else {
    Write-Host "[ERROR] Neither streamlit, py, nor python was found in PATH." -ForegroundColor Red
    Write-Host "Please ensure Python 3.9+ is installed and accessible." -ForegroundColor Yellow
}

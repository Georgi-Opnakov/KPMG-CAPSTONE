$ErrorActionPreference = "Stop"

$chatbotDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $chatbotDir

Write-Host "Starting Airbnb Intelligent Advisor..." -ForegroundColor Cyan
Write-Host "Folder: $chatbotDir"
Write-Host ""

$pythonCandidates = @(
    "$env:USERPROFILE\anaconda3\python.exe",
    "$env:USERPROFILE\miniconda3\python.exe",
    "python"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    try {
        $resolved = Get-Command $candidate -ErrorAction Stop
        $pythonExe = $resolved.Source
        break
    }
    catch {
        if (Test-Path $candidate) {
            $pythonExe = $candidate
            break
        }
    }
}

if (-not $pythonExe) {
    Write-Host "Could not find Python. Please open Anaconda Prompt and install/run the app manually." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Using Python: $pythonExe"
Write-Host "Opening Streamlit at http://localhost:8501"
Write-Host "Keep this window open while using the chatbot."
Write-Host ""

$port = 8501
$url = "http://localhost:$port"
$existingServer = $null

try {
    $existingServer = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop
}
catch {
    $existingServer = $null
}

if ($existingServer) {
    Write-Host "Port $port is already in use, so Lilly may already be running." -ForegroundColor Yellow
    Write-Host "Opening $url in your browser instead of starting a second server."
    Start-Process $url
    Read-Host "Press Enter to close this launcher window"
    exit 0
}

& $pythonExe -m streamlit run app.py --server.port $port

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Streamlit stopped with exit code $LASTEXITCODE." -ForegroundColor Red
    Write-Host "If a dependency is missing, run: pip install -r requirements.txt"
    Read-Host "Press Enter to close"
}

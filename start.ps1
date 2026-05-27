# VeriQuery Startup Script
# Usage: .\start.ps1
$ErrorActionPreference = "Continue"
$PROJECT_DIR = $PSScriptRoot
$VENV_PYTHON = "$PROJECT_DIR\.venv\Scripts\python.exe"
$VENV_STREAMLIT = "$PROJECT_DIR\.venv\Scripts\streamlit.exe"
$API_PORT = 8000
$UI_PORT = 8501

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  VeriQuery Auto Start" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Kill old processes on ports
Write-Host "[1/4] Cleaning ports $API_PORT, $UI_PORT ..." -ForegroundColor Yellow

foreach ($port in @($API_PORT, $UI_PORT)) {
    $connections = netstat -ano | Select-String ":$port\s.*LISTENING"
    foreach ($conn in $connections) {
        $pidNum = ($conn.ToString() -split '\s+')[-1]
        if ($pidNum -match '^\d+$') {
            Write-Host "  -> Killing PID $pidNum (port $port)" -ForegroundColor DarkGray
            taskkill /F /PID $pidNum 2>$null | Out-Null
        }
    }
}
Start-Sleep -Seconds 2
Write-Host "  [OK] Ports cleaned`n" -ForegroundColor Green

# Step 2: Start Backend API with .venv
Write-Host "[2/4] Starting Backend API (port $API_PORT) ..." -ForegroundColor Yellow
Write-Host "  Using venv: $VENV_PYTHON" -ForegroundColor DarkGray

$backendJob = Start-Job -ScriptBlock {
    param($py, $dir)
    Set-Location $dir
    & $py -m api.main 2>&1
} -ArgumentList $VENV_PYTHON, $PROJECT_DIR

Start-Sleep -Seconds 8

$apiReady = $false
for ($i = 1; $i -le 15; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$API_PORT/health" -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            $apiReady = $true
            break
        }
    } catch {}
    Start-Sleep -Milliseconds 800
}

if ($apiReady) {
    Write-Host "  [OK] Backend ready: http://localhost:$API_PORT`n" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Backend failed to start!" -ForegroundColor Red
    Receive-Job $backendJob 2>$null | Select-Object -Last 20
    exit 1
}

# Step 3: Start Frontend UI
Write-Host "[3/4] Starting Frontend UI (port $UI_PORT) ..." -ForegroundColor Yellow

$frontendJob = Start-Job -ScriptBlock {
    param($sl, $dir)
    Set-Location $dir
    & $sl run ui/app.py --server.port 8501 2>&1
} -ArgumentList $VENV_STREAMLIT, $PROJECT_DIR

Start-Sleep -Seconds 8

$uiReady = $false
for ($i = 1; $i -le 10; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$UI_PORT" -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            $uiReady = $true
            break
        }
    } catch {}
    Start-Sleep -Milliseconds 500
}

if ($uiReady) {
    Write-Host "  [OK] Frontend ready: http://localhost:$UI_PORT`n" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Frontend still starting...`n" -ForegroundColor DarkYellow
}

# Step 4: Final verification
Write-Host "[4/4] Verifying connectivity ..." -ForegroundColor Yellow

try {
    $health = Invoke-RestMethod -Uri "http://localhost:$API_PORT/health" -TimeoutSec 5
    Write-Host "  Backend: $($health.status) (v$($health.version))" -ForegroundColor Green
    
    $docs = Invoke-RestMethod -Uri "http://localhost:$API_PORT/api/v1/documents/" -TimeoutSec 5
    Write-Host "  Docs API: OK ($($docs.total) docs)" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] Verify failed: $_" -ForegroundColor DarkYellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ALL SERVICES STARTED!" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "  Frontend: http://localhost:$UI_PORT" -ForegroundColor White
Write-Host "  Backend:  http://localhost:$API_PORT" -ForegroundColor White
Write-Host "  Health:   http://localhost:$API_PORT/health" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "  Press Ctrl+C to stop all services" -ForegroundColor DarkGray
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    while ($true) {
        Receive-Job $backendJob 2>$null | Select-Object -Last 1 | ForEach-Object { 
            if ($_ -and $_.ToString().Trim()) { Write-Host "[Backend] $_" -ForegroundColor DarkGray }
        }
        Receive-Job $frontendJob 2>$null | Select-Object -Last 1 | ForEach-Object { 
            if ($_ -and $_.ToString().Trim()) { Write-Host "[Frontend] $_" -ForegroundColor DarkGray }
        }
        Start-Sleep -Seconds 3
    }
} finally {
    Write-Host "`nCleaning up..." -ForegroundColor Yellow
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -Force -ErrorAction SilentlyContinue
    Remove-Job $frontendJob -Force -ErrorAction SilentlyContinue
    Write-Host "All services stopped." -ForegroundColor Green
}

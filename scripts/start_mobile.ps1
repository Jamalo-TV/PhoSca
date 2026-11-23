param(
    [string]$DatabaseFile = "annotation.db"
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$frontend = Join-Path $root "frontend"
$storage = Join-Path $root "storage"
$models = Join-Path $root "models"

$env:PYTHONUTF8 = "1"
$env:DATABASE_URL = "sqlite+aiosqlite:///./$DatabaseFile"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:STORAGE_PATH = $storage
$env:YOLO_MODEL_PATH = Join-Path $models "yolov8-seg-album.onnx"
$env:MODEL_MANIFEST_PATH = Join-Path $models "manifest.json"
$env:SEGMENTATION_CALIBRATION_PATH = Join-Path $models "segmentation-calibration.json"
$env:REQUIRE_REGISTERED_MODELS = "true"
$env:REQUIRE_CONFIDENCE_CALIBRATION = "true"
$env:ENABLE_CLASSICAL_SEGMENTATION_FALLBACK = "true"
$env:ENABLE_PROTECTED_REGION_GUARD = "true"
$env:ENABLE_PHOTO_ORIENTATION_CORRECTION = "false"
$env:ENABLE_CONSERVATIVE_ENHANCEMENT = "true"
$env:ENABLE_DEEP_ENHANCEMENT = "false"
$env:ENABLE_PREMIUM_ENHANCEMENT = "false"
$env:AUTO_ANALYZE_UPLOADS = "true"
$env:SHADOW_MODE = "true"
$env:RELEASE_KILL_SWITCH = "true"

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

if (-not (Test-Path $env:YOLO_MODEL_PATH)) {
    Write-Warning "Segmentierungsmodell fehlt. Die Erkennung verwendet Bildkanten und ist weniger genau."
}

if (-not (Test-Port 8000)) {
    Start-Process -FilePath python -ArgumentList "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000" -WorkingDirectory $root -WindowStyle Hidden
}
if (-not (Test-Port 5173)) {
    Start-Process -FilePath npm.cmd -ArgumentList "run", "dev", "--", "--host", "0.0.0.0" -WorkingDirectory $frontend -WindowStyle Hidden
}

Start-Sleep -Seconds 2
$candidates = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object {
    $_.IPAddress -like "192.168.*" -and $_.IPAddress -notlike "169.254.*"
} | Sort-Object InterfaceMetric
$address = ($candidates | Select-Object -First 1).IPAddress
if (-not $address) { $address = "127.0.0.1" }

Write-Host "PhoSca Mobile läuft: https://$address`:5173/mobile" -ForegroundColor Green
Write-Host "PC-Ansicht:          https://$address`:5173/" -ForegroundColor Cyan
Write-Host "Kamera benötigt HTTPS und ein auf dem Smartphone akzeptiertes lokales Zertifikat."

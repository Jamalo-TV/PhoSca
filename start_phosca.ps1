param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $PSScriptRoot).Path
Set-Location $root

Write-Host "Starting PhoSca..." -ForegroundColor Cyan

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Start Docker Desktop and try again."
}

try {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not running. Start it and try again."
    }
} catch {
    throw $_
}

if (-not (Test-Path (Join-Path $root ".env"))) {
    throw "The project .env file is missing. Copy .env.example to .env and configure it first."
}

docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    throw "PhoSca could not be started. Check the Docker output above."
}

Write-Host "PhoSca is running." -ForegroundColor Green
Write-Host "Web UI: http://localhost/" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000/" -ForegroundColor Green

if (-not $NoBrowser) {
    Start-Process "http://localhost/"
}

Write-Host "This window can be closed; the Docker services will keep running." -ForegroundColor DarkGray

# Re-run the model, rebuild every chart, refresh the GitHub Pages index,
# then commit and push the HTML outputs so the live URLs update.
#
# Usage:  pwsh -File scripts/publish_charts.ps1
# Or just:  scripts/publish_charts.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

# Resolve repo root from the script location so this works from anywhere
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[1/7] Re-running model..." -ForegroundColor Cyan
python src/model.py | Select-Object -Last 12

Write-Host "[2/7] Building maps..." -ForegroundColor Cyan
python src/build_maps.py | Select-Object -Last 4

Write-Host "[3/7] Building competitive map..." -ForegroundColor Cyan
python src/build_competitive_map.py | Select-Object -Last 2

Write-Host "[4/7] Building tables..." -ForegroundColor Cyan
python src/build_table.py | Select-Object -Last 2

Write-Host "[5/7] Building control chart + index..." -ForegroundColor Cyan
python src/build_charts.py | Select-Object -Last 2
python scripts/build_index.py

Write-Host "[6/7] Building Flourish CSV..." -ForegroundColor Cyan
python scripts/build_competitive_csv.py | Select-Object -Last 2

Write-Host "[7/7] Committing + pushing outputs..." -ForegroundColor Cyan
git add output/*.html output/competitive_races.csv
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "  Nothing changed in output/ — skipping commit." -ForegroundColor Yellow
} else {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "Publish charts $stamp"
    git push
    Write-Host ""
    Write-Host "Done. Charts will be live in ~30 seconds at:" -ForegroundColor Green
    Write-Host "  https://bgriffin0312.github.io/tx-leg-model/" -ForegroundColor Green
    Write-Host "  https://bgriffin0312.github.io/tx-leg-model/competitive_races.csv  (Flourish live data URL)" -ForegroundColor Green
}

# Weekly poll check. Run by Scheduled Task 'poll-check-weekly'.
# Writes a dated checklist; extraction stays a session job.
#
# Deliberately NOT "-ErrorAction Stop" around the python call. The first version
# was, and a single failing URL probe aborted the run partway: the dated log was
# left truncated at 906 bytes and the copy to poll-check-latest.txt never
# happened, so the newest file on disk was silently stale. A checklist that
# half-writes is worse than one that reports a broken probe.
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd"
$out = Join-Path $logDir "poll-check-$stamp.txt"

try {
    $ErrorActionPreference = "Continue"
    & C:\Python314\python.exe src\weekly_poll_check.py --fetch 2>&1 |
        Out-String -Width 200 |
        Set-Content -Path $out -Encoding utf8
} catch {
    Add-Content -Path $out -Value "`nLAUNCHER ERROR: $_" -Encoding utf8
} finally {
    # Always leave a findable latest, even on a partial run.
    if (Test-Path $out) {
        Copy-Item $out (Join-Path $logDir "poll-check-latest.txt") -Force
    }
}

Get-ChildItem $logDir -Filter "poll-check-2*.txt" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-90) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

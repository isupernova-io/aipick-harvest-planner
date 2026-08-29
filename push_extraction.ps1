# Push the extraction notebook and the updated README
#
# Run from anywhere; the paths are absolute. Each step prints what it is about to do, and the
# staging check stops before the commit if anything that should stay local has crept in.

$ErrorActionPreference = 'Stop'
$GIT = 'C:\aipick\git'

Write-Host ""
Write-Host "1. copying files" -ForegroundColor Cyan

Copy-Item C:\aipick\final\00_extraction.ipynb $GIT -Force
Write-Host "   00_extraction.ipynb"

# README.md comes from the outputs folder — replace this path with wherever you saved it
$readme = Read-Host "   full path to the new README.md (Enter to skip)"
if ($readme -and (Test-Path $readme)) {
    Copy-Item $readme "$GIT\README.md" -Force
    Write-Host "   README.md"
}

Set-Location $GIT

Write-Host ""
Write-Host "2. what would be committed" -ForegroundColor Cyan
git add -A
git status --short

# Nothing under apple_crops, runs, or visual should ever reach the repository. If the ignore
# rules have slipped, catching it here is cheaper than rewriting history afterwards.
$leaked = git status --short | Select-String "apple_crops/|runs/|visual/|\.bak|__pycache__|papple_trainval"
if ($leaked) {
    Write-Host ""
    Write-Host "   STOP — these should not be committed:" -ForegroundColor Yellow
    $leaked | ForEach-Object { Write-Host "     $_" -ForegroundColor Yellow }
    Write-Host "   Check .gitignore, then run this script again." -ForegroundColor Yellow
    return
}

# The extraction notebook keeps its cell outputs on purpose — they are the evidence behind the
# filtering table in the report — but the source images were cleared, so it should be small.
$nb = Get-Item "$GIT\00_extraction.ipynb"
Write-Host ""
Write-Host ("3. 00_extraction.ipynb is {0:N1} MB" -f ($nb.Length / 1MB)) -ForegroundColor Cyan
if ($nb.Length -gt 3MB) {
    Write-Host "   Larger than expected — the FRESH sample images may still be in the outputs." -ForegroundColor Yellow
    Write-Host "   Clear the 0b cell's output and save before continuing." -ForegroundColor Yellow
    return
}

Write-Host ""
Write-Host "4. committing" -ForegroundColor Cyan
git commit -m "Add the extraction notebook and document the data lineage

00_extraction filters the FRESH labels and writes the per-fruit crops and manifest that
everything downstream is built from. The README now starts at the orchard imagery rather than
at the 8,133 detections, and records what the filtering removes — 709 of the 911 are occluded
fruit, which are disproportionately the ones growing in clusters."

Write-Host ""
Write-Host "5. pushing" -ForegroundColor Cyan
git push

Write-Host ""
Write-Host "done — check https://github.com/isupernova-io/aipick-harvest-planner" -ForegroundColor Green
Write-Host ""

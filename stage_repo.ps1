# stage_repo.ps1 — copy the working tree into a clean repository folder
#
# `final` is the working directory: it holds intermediate runs, superseded weights, scratch
# notebooks and the FRESH source, none of which belong in a public repository. This script
# copies only what a reader needs to reproduce the results, and reports anything it expected
# but could not find rather than failing silently.
#
#   powershell -ExecutionPolicy Bypass -File stage_repo.ps1
#
# Run it from anywhere; paths are absolute.

$ErrorActionPreference = 'Stop'
$SRC = 'C:\aipick\final'
$DST = 'C:\aipick\git'

Write-Host ""
Write-Host "staging  $SRC  ->  $DST" -ForegroundColor Cyan
Write-Host ""

$missing = @()
$copied  = 0

function Copy-One {
    param([string]$Rel, [string]$ToDir)
    $from = Join-Path $SRC $Rel
    if (-not (Test-Path $from)) {
        $script:missing += $Rel
        Write-Host ("  MISSING  {0}" -f $Rel) -ForegroundColor Yellow
        return
    }
    $to = Join-Path $DST $ToDir
    if (-not (Test-Path $to)) { New-Item -ItemType Directory -Force -Path $to | Out-Null }
    Copy-Item $from -Destination $to -Force
    $size = (Get-Item $from).Length / 1MB
    Write-Host ("  ok       {0,-42} {1,8:N2} MB" -f $Rel, $size)
    $script:copied++
}

function Copy-Many {
    param([string]$RelDir, [string]$Pattern, [string]$ToDir)
    $dir = Join-Path $SRC $RelDir
    if (-not (Test-Path $dir)) {
        $script:missing += "$RelDir\$Pattern"
        Write-Host ("  MISSING  {0}\{1}" -f $RelDir, $Pattern) -ForegroundColor Yellow
        return
    }
    $files = Get-ChildItem -Path $dir -Filter $Pattern -File
    if ($files.Count -eq 0) {
        $script:missing += "$RelDir\$Pattern"
        Write-Host ("  MISSING  {0}\{1}  (none matched)" -f $RelDir, $Pattern) -ForegroundColor Yellow
        return
    }
    $to = Join-Path $DST $ToDir
    if (-not (Test-Path $to)) { New-Item -ItemType Directory -Force -Path $to | Out-Null }
    foreach ($f in $files) {
        Copy-Item $f.FullName -Destination $to -Force
        Write-Host ("  ok       {0,-42} {1,8:N2} MB" -f "$RelDir\$($f.Name)", ($f.Length/1MB))
        $script:copied++
    }
}

# ── source modules ───────────────────────────────────────────────────────────
Write-Host "src/" -ForegroundColor Green
foreach ($m in 'physics.py','generate.py','environment.py','planner.py',
                'reobserve.py','replay.py') {
    Copy-One "src\$m" 'src'
}
Copy-Many 'src' '*.xml' 'src'

# ── execution notebooks ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "notebooks (repository root)" -ForegroundColor Green
foreach ($n in '01_data_preparation','02_integrated_model','03_evaluation',
                '04_fidelity','05_test','06_orchard_estimate',
                '07_station_sweep','08_dataset_figures') {
    Copy-One "$n.ipynb" '.'
}

# ── visualisation ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "visual/" -ForegroundColor Green
Copy-Many 'visual' '*.ipynb' 'visual'

# ── trained weights ──────────────────────────────────────────────────────────
# Only the four the shipping configuration loads. `final\models` also holds superseded
# checkpoints from earlier configurations; shipping those invites someone to load the wrong one.
Write-Host ""
Write-Host "models/" -ForegroundColor Green
foreach ($w in 'outcome.pkl','dynamics.joblib','station_planner_k20.pt','pick_policy.pt') {
    Copy-One "models\$w" 'models'
}

# ── derived data ─────────────────────────────────────────────────────────────
# The FRESH source is not redistributed. What goes in is what this project generated.
Write-Host ""
Write-Host "data/" -ForegroundColor Green
Copy-One 'data\picks.csv' 'data'
Copy-One 'data\trees_measured_pose.csv' 'data'
Copy-Many 'data\transitions' '*.csv' 'data\transitions'

# ── empty directories the notebooks expect ───────────────────────────────────
Write-Host ""
Write-Host "empty directories" -ForegroundColor Green
foreach ($d in 'apple_crops','runs','runs\report','runs\orchard','runs\stations',
                'runs\fidelity','visual\out') {
    $p = Join-Path $DST $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
    New-Item -ItemType File -Force -Path (Join-Path $p '.gitkeep') | Out-Null
    Write-Host ("  ok       {0}" -f $d)
}

# ── report ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("copied {0} files" -f $copied) -ForegroundColor Cyan
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host ("{0} expected files were not found:" -f $missing.Count) -ForegroundColor Yellow
    foreach ($m in $missing) { Write-Host "    $m" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "  Check the names against the working tree. A missing module or weight" -ForegroundColor Yellow
    Write-Host "  will surface as an import error the first time a notebook is run." -ForegroundColor Yellow
}

# ── what deliberately stays behind ───────────────────────────────────────────
Write-Host ""
Write-Host "not copied, by design:" -ForegroundColor DarkGray
Write-Host "    apple_crops\      FRESH source — redistribution terms unclear" -ForegroundColor DarkGray
Write-Host "    runs\             regenerated by running the notebooks" -ForegroundColor DarkGray
Write-Host "    models\ (others)  superseded checkpoints from earlier configurations" -ForegroundColor DarkGray
Write-Host "    archive\          training notebooks kept for provenance, not for reuse" -ForegroundColor DarkGray
Write-Host ""
Write-Host "next:" -ForegroundColor Cyan
Write-Host "    1. put the FRESH dataset in git\apple_crops\"
Write-Host "    2. run the notebooks 01 to 08 in order"
Write-Host "    3. if all of them finish, commit"
Write-Host ""

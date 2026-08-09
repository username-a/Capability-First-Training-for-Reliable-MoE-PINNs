param(
    [string]$SourceRoot = "Burger2D\results\true_staged_vs_coadapt_20260806",
    [string]$ResultsRoot = "Burger2D\results\capability_gate_confirmatory_20260806"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path (Get-Location) ".venv310\Scripts\python.exe"
$script = Join-Path (Get-Location) "Burger2D\scripts\run_capability_gate_experiment.py"
$candidates = @("cap", "hyb_0p1", "hyb_0p3", "hyb_1", "hyb_3", "hyb_bal_0p1", "hyb_bal_0p3", "hyb_bal_1", "hyb_bal_3")
$screenSeeds = @(42, 43, 44)
$allSeeds = @(42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
$evalSteps = @(50, 100, 200, 400, 750)

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
foreach ($candidate in $candidates) {
    foreach ($seed in $screenSeeds) {
        $source = Join-Path $SourceRoot ("seed{0}_staged" -f $seed)
        $run = Join-Path $ResultsRoot ("screen\{0}\seed{1}" -f $candidate, $seed)
        if (Test-Path -LiteralPath (Join-Path $run "val_step750.json")) {
            Write-Output "[screen skip] $candidate seed=$seed"
            continue
        }
        New-Item -ItemType Directory -Path $run -Force | Out-Null
        Write-Output "[screen] $candidate seed=$seed"
        & $pythonExe $script train --source-dir $source --output-dir $run --candidate $candidate --steps 750 --eval-steps $evalSteps
        if ($LASTEXITCODE -ne 0) { throw "Screen failed: $candidate seed=$seed" }
    }
}

& $pythonExe $script select --root $ResultsRoot --seeds $screenSeeds --eval-steps $evalSteps
if ($LASTEXITCODE -ne 0) { throw "Selection failed" }
$selection = Get-Content -Raw -LiteralPath (Join-Path $ResultsRoot "selection.json") -Encoding UTF8 | ConvertFrom-Json
$selectedCandidate = [string]$selection.selected.candidate
$selectedStep = [int]$selection.selected.step
$capStep = [int]$selection.selected_capability.step
Write-Output "[selected] candidate=$selectedCandidate step=$selectedStep; cap_step=$capStep"

foreach ($seed in $allSeeds) {
    $source = Join-Path $SourceRoot ("seed{0}_staged" -f $seed)
    foreach ($kind in @("selected", "capability")) {
        if ($kind -eq "selected") { $candidate = $selectedCandidate; $steps = $selectedStep }
        else { $candidate = "cap"; $steps = $capStep }
        $run = Join-Path $ResultsRoot ("confirm\{0}\seed{1}" -f $kind, $seed)
        if (Test-Path -LiteralPath (Join-Path $run "test_metrics.json")) {
            Write-Output "[confirm skip] $kind seed=$seed"
            continue
        }
        New-Item -ItemType Directory -Path $run -Force | Out-Null
        Write-Output "[confirm] $kind candidate=$candidate step=$steps seed=$seed"
        & $pythonExe $script train --source-dir $source --output-dir $run --candidate $candidate --steps $steps --eval-steps $steps
        if ($LASTEXITCODE -ne 0) { throw "Confirm training failed: $kind seed=$seed" }
        & $pythonExe $script evaluate-test --run-dir $run --step $steps
        if ($LASTEXITCODE -ne 0) { throw "Confirm evaluation failed: $kind seed=$seed" }
    }
}

& $pythonExe $script summarize --root $ResultsRoot --source-root $SourceRoot --seeds $allSeeds
if ($LASTEXITCODE -ne 0) { throw "Summary failed" }

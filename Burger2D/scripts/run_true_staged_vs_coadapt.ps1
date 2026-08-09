param(
    [string]$ResultsRoot = "Burger2D\results\true_staged_vs_coadapt_20260806"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path (Get-Location) ".venv310\Scripts\python.exe"
$experiment = Join-Path (Get-Location) "Burger2D\scripts\run_true_staged_vs_coadapt.py"
$summary = Join-Path (Get-Location) "Burger2D\scripts\summarize_true_staged_vs_coadapt.py"
$seeds = @(42, 43, 44, 45, 46, 47, 48, 49, 50, 51)

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
foreach ($seed in $seeds) {
    foreach ($mode in @("staged", "coadapt")) {
        $runDir = Join-Path $ResultsRoot ("seed{0}_{1}" -f $seed, $mode)
        if (Test-Path -LiteralPath (Join-Path $runDir "test_metrics.json")) {
            Write-Output "[skip] $runDir"
            continue
        }
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        Write-Output "[train] $runDir"
        & $pythonExe $experiment train --mode $mode --seed $seed --output-dir $runDir
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $runDir" }
        Write-Output "[evaluate pre/post] $runDir"
        & $pythonExe $experiment evaluate --run-dir $runDir
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $runDir" }
    }
}

& $pythonExe $summary --root $ResultsRoot --seeds $seeds
if ($LASTEXITCODE -ne 0) { throw "Summary failed" }

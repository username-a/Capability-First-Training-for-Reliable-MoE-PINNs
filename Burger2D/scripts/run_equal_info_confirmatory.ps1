param(
    [string]$ResultsRoot = "Burger2D\results\equal_info_2x2_confirmatory_20260806"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path (Get-Location) ".venv310\Scripts\python.exe"
$experimentScript = Join-Path (Get-Location) "Burger2D\scripts\run_equal_information_2x2.py"
$summaryScript = Join-Path (Get-Location) "Burger2D\scripts\summarize_equal_information_2x2.py"
$groups = @("P-B", "P-I", "R-B", "R-I")
$seeds = @(42, 43, 44, 45, 46)

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
foreach ($seed in $seeds) {
    foreach ($group in $groups) {
        $runDir = Join-Path $ResultsRoot ("seed{0}_{1}" -f $seed, $group)
        $metricsPath = Join-Path $runDir "test_metrics.json"
        if (Test-Path -LiteralPath $metricsPath) {
            Write-Output "[skip] $runDir"
            continue
        }
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        Write-Output "[train] $runDir"
        & $pythonExe $experimentScript train --group $group --seed $seed --output-dir $runDir
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $runDir" }
        Write-Output "[evaluate] $runDir"
        & $pythonExe $experimentScript evaluate --run-dir $runDir
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $runDir" }
    }
}

& $pythonExe $summaryScript --root $ResultsRoot --seeds 42 43 44 45 46
if ($LASTEXITCODE -ne 0) { throw "Summary failed" }

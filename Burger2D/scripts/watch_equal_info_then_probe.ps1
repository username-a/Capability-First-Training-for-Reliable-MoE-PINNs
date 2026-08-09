param(
    [string]$ResultsRoot = "Burger2D\results\equal_info_2x2_confirmatory_20260806",
    [string]$ProbeRoot = "Burger2D\results\probe_expert_pilot_20260806"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path (Get-Location) ".venv310\Scripts\python.exe"
$experimentScript = Join-Path (Get-Location) "Burger2D\scripts\run_equal_information_2x2.py"
$summaryScript = Join-Path (Get-Location) "Burger2D\scripts\summarize_equal_information_2x2.py"
$decisionScript = Join-Path (Get-Location) "Burger2D\scripts\decide_equal_info_2x2.py"
$probeScript = Join-Path (Get-Location) "Burger2D\scripts\run_probe_expert_experiment.py"
$initialSummary = Join-Path $ResultsRoot "confirmatory_summary.json"

while (-not (Test-Path -LiteralPath $initialSummary)) {
    Start-Sleep -Seconds 30
}

& $pythonExe $decisionScript --summary $initialSummary
$perfect = $LASTEXITCODE -eq 0
if (-not $perfect) {
    Write-Output "[decision] initial 2x2 not strong enough; extending to seeds 47-51"
    foreach ($seed in @(47, 48, 49, 50, 51)) {
        foreach ($group in @("P-B", "P-I", "R-B", "R-I")) {
            $runDir = Join-Path $ResultsRoot ("seed{0}_{1}" -f $seed, $group)
            if (Test-Path -LiteralPath (Join-Path $runDir "test_metrics.json")) { continue }
            New-Item -ItemType Directory -Path $runDir -Force | Out-Null
            Write-Output "[extended train] $runDir"
            & $pythonExe $experimentScript train --group $group --seed $seed --output-dir $runDir
            if ($LASTEXITCODE -ne 0) { throw "Extended training failed: $runDir" }
            & $pythonExe $experimentScript evaluate --run-dir $runDir
            if ($LASTEXITCODE -ne 0) { throw "Extended evaluation failed: $runDir" }
        }
    }
    & $pythonExe $summaryScript --root $ResultsRoot --seeds 42 43 44 45 46 47 48 49 50 51
    if ($LASTEXITCODE -ne 0) { throw "Extended summary failed" }
} else {
    Write-Output "[decision] initial 2x2 is strong; proceeding directly to probe expert"
}

New-Item -ItemType Directory -Path $ProbeRoot -Force | Out-Null
foreach ($seed in @(42, 43, 44)) {
    $sourceDir = Join-Path $ResultsRoot ("seed{0}_R-B" -f $seed)
    foreach ($variant in @("healthy", "random_directional")) {
        foreach ($objective in @("physics_no_balance", "physics_balance", "capability_aware")) {
            $runDir = Join-Path $ProbeRoot ("seed{0}_{1}_{2}" -f $seed, $variant, $objective)
            if (Test-Path -LiteralPath (Join-Path $runDir "test_metrics.json")) { continue }
            Write-Output "[probe] $runDir"
            & $pythonExe $probeScript run --source-dir $sourceDir --output-dir $runDir --variant $variant --objective $objective --steps 300
            if ($LASTEXITCODE -ne 0) { throw "Probe run failed: $runDir" }
        }
    }
}
& $pythonExe $probeScript summarize --root $ProbeRoot
if ($LASTEXITCODE -ne 0) { throw "Probe summary failed" }

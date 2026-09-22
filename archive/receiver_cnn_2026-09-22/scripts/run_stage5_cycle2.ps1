param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$artifacts = Join-Path $projectRoot "artifacts\stage5_cycle2"
$manifest = Join-Path $artifacts "manifest.json"
$references = Join-Path $artifacts "references"
$teacherTargets = Join-Path $artifacts "teacher_targets.npz"
$stage5Checkpoint = Join-Path $projectRoot "artifacts\stage5\training\best.pt"
$stage4Checkpoint = Join-Path $projectRoot "artifacts\stage4\training\best.pt"

if ($PollSeconds -lt 1 -or $PollSeconds -gt 60) {
    throw "PollSeconds must be between 1 and 60"
}

function Invoke-PythonStep {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

$referenceReport = Join-Path $references "report.json"
Write-Host "Waiting for the expanded reference corpus..."
while ($true) {
    if (Test-Path -LiteralPath $referenceReport) {
        $report = Get-Content -Raw -LiteralPath $referenceReport | ConvertFrom-Json
        if ($report.coverage.completed -eq $report.selected_scene_count) {
            break
        }
        Write-Host (
            "References: {0}/{1} complete, {2} accepted" -f
            $report.coverage.completed, $report.selected_scene_count, $report.coverage.accepted
        )
    }
    Start-Sleep -Seconds $PollSeconds
}

$trainAccepted = [int]$report.coverage.by_split.train.converged
$validationAccepted = [int]$report.coverage.by_split.validation.converged
if ($trainAccepted -lt 200 -or $validationAccepted -lt 40) {
    throw "Reference coverage gate failed: train=$trainAccepted validation=$validationAccepted"
}

$search = Join-Path $artifacts "search_w50"
$physics50 = Join-Path $search "physics_targets.npz"
if (-not (Test-Path -LiteralPath $physics50)) {
    Invoke-PythonStep @(
        "-m", "fdtdmesh.physics", "search",
        "--manifest", $manifest,
        "--teacher-targets", $teacherTargets,
        "--checkpoint", $stage5Checkpoint,
        "--baseline-checkpoint", $stage4Checkpoint,
        "--references", $references,
        "--output", $search,
        "--splits", "train", "validation",
        "--physics-weight", "0.5",
        "--beta", "0.02",
        "--perturbations", "2",
        "--device", "cuda"
    )
}

$physics80 = Join-Path $artifacts "physics_targets_w80.npz"
if (-not (Test-Path -LiteralPath $physics80)) {
    Invoke-PythonStep @(
        "-m", "fdtdmesh.physics", "reblend",
        "--manifest", $manifest,
        "--teacher-targets", $teacherTargets,
        "--source-targets", $physics50,
        "--output", $physics80,
        "--physics-weight", "0.8"
    )
}

$trainingRuns = @(
    @{ Name = "w50"; Targets = $physics50 },
    @{ Name = "w80"; Targets = $physics80 }
)
foreach ($run in $trainingRuns) {
    $training = Join-Path $artifacts ("training_" + $run.Name)
    $arguments = @(
        "-m", "fdtdmesh.physics", "distill",
        "--manifest", $manifest,
        "--targets", $run.Targets,
        "--initial-checkpoint", $stage5Checkpoint,
        "--output", $training,
        "--epochs", "60",
        "--batch-size", "8",
        "--learning-rate", "0.0003",
        "--repair-weight", "0.05",
        "--repair-every", "4",
        "--projection-samples", "8",
        "--patience", "10",
        "--min-delta", "0.0000001",
        "--device", "cuda"
    )
    $resume = Join-Path $training "resume.pt"
    $trainingReport = Join-Path $training "training.json"
    if (Test-Path -LiteralPath $resume) {
        $stoppedEarly = $false
        if (Test-Path -LiteralPath $trainingReport) {
            $trainingStatus = Get-Content -Raw -LiteralPath $trainingReport | ConvertFrom-Json
            $stoppedEarly = [bool]$trainingStatus.stopped_early
        }
        $state = & $python -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['epoch'])" $resume
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect $resume" }
        if ([int]$state -lt 60 -and -not $stoppedEarly) {
            $arguments += @("--resume", $resume)
            Invoke-PythonStep $arguments
        }
    } elseif (-not (Test-Path -LiteralPath $trainingReport)) {
        Invoke-PythonStep $arguments
    }
}

$evaluationReports = @()
$checkpoints = @()
foreach ($run in $trainingRuns) {
    $checkpoint = Join-Path $artifacts ("training_" + $run.Name + "\best.pt")
    $evaluation = Join-Path $artifacts ("validation_" + $run.Name)
    $evaluationReport = Join-Path $evaluation "report.json"
    if (-not (Test-Path -LiteralPath $evaluationReport)) {
        Invoke-PythonStep @(
            "-m", "fdtdmesh.physics", "evaluate",
            "--manifest", $manifest,
            "--checkpoint", $checkpoint,
            "--baseline-checkpoint", $stage5Checkpoint,
            "--references", $references,
            "--output", $evaluation,
            "--split", "validation",
            "--device", "cuda"
        )
    }
    $evaluationReports += $evaluationReport
    $checkpoints += $checkpoint
}

$selection = Join-Path $artifacts "validation_selection.json"
if (-not (Test-Path -LiteralPath $selection)) {
    Invoke-PythonStep @(
        "-m", "fdtdmesh.physics", "select",
        "--evaluations", $evaluationReports[0], $evaluationReports[1],
        "--checkpoints", $checkpoints[0], $checkpoints[1],
        "--output", $selection,
        "--beta", "0.02"
    )
}

$selected = Get-Content -Raw -LiteralPath $selection | ConvertFrom-Json
$testOutput = Join-Path $artifacts "test_selected"
if (-not (Test-Path -LiteralPath (Join-Path $testOutput "report.json"))) {
    Invoke-PythonStep @(
        "-m", "fdtdmesh.physics", "evaluate",
        "--manifest", $manifest,
        "--checkpoint", $selected.selected_checkpoint,
        "--baseline-checkpoint", $stage5Checkpoint,
        "--references", $references,
        "--output", $testOutput,
        "--split", "test_iid",
        "--device", "cuda"
    )
}

Write-Host "Stage 5 cycle 2 pipeline completed."

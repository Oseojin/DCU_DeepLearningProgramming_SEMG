# Sequential overnight batch runner for semg_training_v2.
#
#   .\run_batch.ps1
#
# Runs each job one after another and logs to runs\_batch\.
# A failed job does not stop the batch. Windows sleep is suppressed while this window is open.
# Safe to re-run: finished jobs are skipped, half-finished ones are set aside and retried.
# On CUDA out-of-memory a job is retried once at --batch-size 32 --lr 3e-4.

[CmdletBinding()]
param(
    [string]$Manifest = "runs\baseline_densenet161\splits.json",
    [switch]$SkipReport
)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root

# --- resolve the interpreter: prefer this folder's venv over whatever is on PATH -----------
$Python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $Python) { throw "No python found. Create .venv or activate an environment first." }
if (-not (Test-Path (Join-Path $root $Manifest))) {
    throw "Manifest not found: $Manifest  (run the baseline first)"
}
# Absolute paths everywhere, so the child process never depends on its working directory.
$manifestPath = (Resolve-Path (Join-Path $root $Manifest)).Path
$trainScript = Join-Path $root "train.py"

# --- jobs, in order. Edit, reorder or comment out freely. ----------------------------------
# To make a job cheaper, add e.g. "--fold","0" to its Extra array.
$jobs = @(
    [pscustomobject]@{ Name = "cand_densenet121_cwt";       Model = "densenet121_cwt";  EstHours = 4.6; Extra = @() }
    [pscustomobject]@{ Name = "cand_densenet_xs_cwt";       Model = "densenet_xs_cwt";  EstHours = 1.4; Extra = @() }
    [pscustomobject]@{ Name = "cand_efficientnet_b0";       Model = "efficientnet_b0";  EstHours = 1.9; Extra = @() }
    [pscustomobject]@{ Name = "cand_densenet161_newrecipe"; Model = "densenet161";      EstHours = 6.3; Extra = @() }
)

$batchDir = Join-Path $root "runs\_batch"
New-Item -ItemType Directory -Force -Path $batchDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$statusPath = Join-Path $batchDir "status.txt"

# --- keep Windows from sleeping while this script is alive ---------------------------------
# Decimal literals on purpose: Windows PowerShell 5.1 parses 0x80000000 as a negative Int32.
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$awake = $null
try {
    $awake = @(Add-Type -PassThru -Name "BatchAwake" -Namespace "SemgAuth" -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@)[0]
    [void]$awake::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    Write-Host "Sleep suppressed while this window stays open (the display may still turn off)."
} catch {
    Write-Warning "Could not suppress sleep: $($_.Exception.Message)"
}

$totalEst = ($jobs | Measure-Object -Property EstHours -Sum).Sum
Write-Host ""
Write-Host "=== batch $stamp ==="
Write-Host "python   : $Python"
Write-Host "manifest : $Manifest"
Write-Host "jobs     : $($jobs.Count), rough total estimate $totalEst h"
Write-Host "logs     : $batchDir"
Write-Host ""

$batchStart = Get-Date
$results = @()

function Save-Status {
    param([array]$Done, [array]$AllJobs, $Start)
    $elapsed = [math]::Round(((Get-Date) - $Start).TotalHours, 2)
    $lines = @("batch $stamp",
               "updated $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
               "elapsed $elapsed h",
               "")
    foreach ($r in $Done) {
        $lines += ("  {0,-32} {1,-8} {2,7} min" -f $r.Name, $r.Status, $r.Minutes)
    }
    if ($Done.Count -lt $AllJobs.Count) {
        $lines += ""
        for ($i = $Done.Count; $i -lt $AllJobs.Count; $i++) {
            $lines += ("  {0,-32} pending  ~{1} h" -f $AllJobs[$i].Name, $AllJobs[$i].EstHours)
        }
    }
    $lines | Set-Content -Path $statusPath -Encoding UTF8
    $lines | ForEach-Object { Write-Host $_ }
}

function Invoke-Training {
    param($Arguments, $OutLog, $ErrLog)
    $env:PYTHONUNBUFFERED = "1"
    $process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $root `
        -NoNewWindow -PassThru -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
    # Touching Handle caches it; without this ExitCode comes back $null on Windows PowerShell 5.1.
    $null = $process.Handle
    $process.WaitForExit()
    if ($null -eq $process.ExitCode) { return 0 }
    return $process.ExitCode
}

foreach ($job in $jobs) {
    $name = $job.Name
    $runDir = Join-Path $root "runs\$name"
    $index = $results.Count + 1
    $header = "[$index/$($jobs.Count)] $name ($($job.Model))"

    # already finished -> skip; started but unfinished -> set aside and retry
    if (Test-Path $runDir) {
        if (Test-Path (Join-Path $runDir "summary.json")) {
            Write-Host "$header : SKIP (already complete)"
            $results += [pscustomobject]@{ Name = $name; Status = "skipped"; Minutes = 0; Exit = 0 }
            Save-Status -Done $results -AllJobs $jobs -Start $batchStart
            continue
        }
        $parked = "$runDir.incomplete-$stamp"
        Move-Item -Path $runDir -Destination $parked
        Write-Host "$header : previous attempt incomplete, moved to $(Split-Path $parked -Leaf)"
    }

    $out = Join-Path $batchDir "${stamp}_$name.out.log"
    $err = Join-Path $batchDir "${stamp}_$name.err.log"
    $arguments = @($trainScript, "--model", $job.Model, "--fold", "all",
                   "--manifest", $manifestPath, "--run-dir", $runDir) + $job.Extra

    Write-Host "$header : started $(Get-Date -Format 'HH:mm:ss')  (est. $($job.EstHours) h)"
    $jobStart = Get-Date
    $code = Invoke-Training -Arguments $arguments -OutLog $out -ErrLog $err
    $done = (Test-Path (Join-Path $runDir "summary.json"))

    # one automatic retry at a smaller batch if the GPU ran out of memory
    if (-not $done) {
        $errText = if (Test-Path $err) { Get-Content $err -Raw } else { "" }
        if ($errText -match "out of memory") {
            Write-Host "$header : CUDA OOM, retrying at --batch-size 32 --lr 3e-4"
            if (Test-Path $runDir) { Move-Item -Path $runDir -Destination "$runDir.oom-$stamp" }
            $arguments += @("--batch-size", "32", "--lr", "3e-4")
            $out = Join-Path $batchDir "${stamp}_$name.retry.out.log"
            $err = Join-Path $batchDir "${stamp}_$name.retry.err.log"
            $code = Invoke-Training -Arguments $arguments -OutLog $out -ErrLog $err
            $done = (Test-Path (Join-Path $runDir "summary.json"))
        }
    }

    $minutes = [math]::Round(((Get-Date) - $jobStart).TotalMinutes, 1)
    # summary.json is the ground truth for success; the exit code is only reported.
    $status = if ($done) { "ok" } else { "FAILED" }
    Write-Host "$header : $status in $minutes min (exit $code)"
    if ($status -ne "ok") { Write-Host "    see $err" }
    $results += [pscustomobject]@{ Name = $name; Status = $status; Minutes = $minutes; Exit = $code }
    Save-Status -Done $results -AllJobs $jobs -Start $batchStart
}

$totalHours = [math]::Round(((Get-Date) - $batchStart).TotalHours, 2)
Write-Host ""
Write-Host "=== batch finished in $totalHours h ==="
$results | Format-Table -AutoSize

if (-not $SkipReport) {
    $reportPath = Join-Path $batchDir "${stamp}_report.txt"
    & $Python (Join-Path $root "report.py") | Tee-Object -FilePath $reportPath
    Write-Host ""
    Write-Host "Comparison table saved to $reportPath"
}

if ($awake) { try { [void]$awake::SetThreadExecutionState($ES_CONTINUOUS) } catch { } }
Write-Host "Status file: $statusPath"

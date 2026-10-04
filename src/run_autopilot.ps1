# Unattended launcher for autopilot.py (steps 3-6).
#
#   .\run_autopilot.ps1 -DryRun            show the plan and time estimates only
#   .\run_autopilot.ps1                    run with the default 24 h budget
#   .\run_autopilot.ps1 -BudgetHours 10    smaller budget
#
# Suppresses Windows sleep while this window stays open.
# Safe to re-run: finished work is skipped and the run continues where it stopped.

[CmdletBinding()]
param(
    [double]$BudgetHours = 24,
    [string]$Model = "efficientnet_b0",
    [string]$Manifest = "runs\baseline_densenet161\splits.json",
    [switch]$DryRun,
    [switch]$NoTest,
    [switch]$SkipAblation,
    [switch]$SkipSeeds
)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root

$Python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "No python found. Create .venv or activate an environment first." }

$arguments = @((Join-Path $root "autopilot.py"),
               "--model", $Model,
               "--budget-hours", $BudgetHours,
               "--manifest", (Join-Path $root $Manifest))
if ($DryRun)       { $arguments += "--dry-run" }
if ($NoTest)       { $arguments += "--no-test" }
if ($SkipAblation) { $arguments += "--skip-ablation" }
if ($SkipSeeds)    { $arguments += "--skip-seeds" }

# Decimal literals on purpose: Windows PowerShell 5.1 parses 0x80000000 as a negative Int32.
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$awake = $null
if (-not $DryRun) {
    try {
        $awake = @(Add-Type -PassThru -Name "AutopilotAwake" -Namespace "SemgAuth" -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@)[0]
        [void]$awake::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
        Write-Host "Sleep suppressed while this window stays open (the display may still turn off)."
    } catch {
        Write-Warning "Could not suppress sleep: $($_.Exception.Message)"
    }
}

$env:PYTHONUNBUFFERED = "1"
$started = Get-Date
& $Python @arguments
$code = $LASTEXITCODE

if (-not $DryRun) {
    $hours = [math]::Round(((Get-Date) - $started).TotalHours, 2)
    Write-Host ""
    Write-Host "=== autopilot finished in $hours h (exit $code) ==="
    Write-Host "Status : $(Join-Path $root 'runs\_autopilot\status.txt')"
    Write-Host "Report : $(Join-Path $root 'runs\_autopilot\REPORT.md')"
}
if ($awake) { try { [void]$awake::SetThreadExecutionState($ES_CONTINUOUS) } catch { } }

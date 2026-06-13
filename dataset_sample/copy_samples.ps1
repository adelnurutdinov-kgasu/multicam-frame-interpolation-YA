<#
    Builds a small, faithful slice of the full ~320 GB dataset.

    Strategy (walks the whole tree once, top-down):
      * A folder whose name matches the sample-id pattern (e.g. 2025-10-01_..._robb_..._000)
        is a SAMPLE folder.
          - if it is one of the 6 selected ids  -> copy it whole, then stop descending
          - otherwise                            -> skip it entirely (this is the 320 GB we drop)
      * Any FILE that is NOT inside a sample folder is a GLOBAL file
        (TASK.md, config_used.yaml, *.csv summaries, run_meta.json, manifests, logs, ...)
        -> copied whole, preserving its relative path.
      * Any non-sample FOLDER is descended into.

    This keeps every selected sample AND every global/aggregate file, while dropping
    all the other samples. Nothing global is lost.

    Source: %USERPROFILE%\Downloads\cv_dataset
    Dest  : <this folder>\cv_dataset\<same relative paths>

    Run by double-clicking copy_samples.bat, or:
        powershell -ExecutionPolicy Bypass -File copy_samples.ps1
#>

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcRoot   = Join-Path $env:USERPROFILE 'Downloads\cv_dataset'
$DstRoot   = Join-Path $ScriptDir 'cv_dataset'

# One representative sample per target camera, each from a different scene.
$Samples = @(
    '2025-10-01_11_45_46_12_19_35_robb_1759310438799892000__000',     # target: front
    '2025-10-02_12_28_59_12_33_15_luka_1759397924100053000__000',     # target: rear
    '2025-10-01_18_29_52_19_31_28_kynde_1759339701900197000__000',    # target: right_bwd
    '2025-10-10_06_36_32_09_00_01_natelio_1760076111099953000__000',  # target: right_fwd
    '2025-10-11_16_12_04_18_35_14_crozby_1760197023299977000__000',   # target: left_fwd
    '2025-10-12_11_30_56_12_10_38_anabel_1760263528700021000__000'    # target: left_bwd
)

# Folder name that looks like a sample id: 2025-10-01_..._<id>__000
$SampleIdRegex = '^\d{4}-\d{2}-\d{2}_.*__\d{3}$'

Write-Host "Source: $SrcRoot"
Write-Host "Dest  : $DstRoot`n"

if (-not (Test-Path $SrcRoot)) {
    Write-Host "ERROR: source folder not found:`n  $SrcRoot" -ForegroundColor Red
    Write-Host "Edit `$SrcRoot at the top of this script to point at your dataset." -ForegroundColor Yellow
    exit 1
}

$want = @{}
foreach ($s in $Samples) { $want[$s] = $true }

$prefix = $SrcRoot.TrimEnd('\') + '\'
$bySample   = @{}
$globalFiles = 0

function Copy-GlobalFile($file) {
    $rel  = $file.FullName.Substring($script:prefix.Length)
    $dest = Join-Path $script:DstRoot $rel
    $dir  = Split-Path $dest -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
    $script:globalFiles++
}

function Copy-SampleDir($d) {
    $rel = $d.FullName.Substring($script:prefix.Length)
    $dst = Join-Path $script:DstRoot $rel
    robocopy $d.FullName $dst /E /R:1 /W:1 /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -le 7) {
        if (-not $script:bySample.ContainsKey($d.Name)) { $script:bySample[$d.Name] = @() }
        $script:bySample[$d.Name] += (Split-Path $rel -Parent)
        Write-Host ("  [sample] {0}" -f $rel)
    } else {
        Write-Host ("  robocopy error {0} on {1}" -f $LASTEXITCODE, $rel) -ForegroundColor Red
    }
}

Write-Host "Walking dataset tree (samples copied whole, globals preserved, others skipped)...`n"

$stack = New-Object System.Collections.Stack
$stack.Push($SrcRoot)

while ($stack.Count -gt 0) {
    $dir = $stack.Pop()
    foreach ($child in Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue) {
        if ($child.PSIsContainer) {
            if ($child.Name -match $SampleIdRegex) {
                if ($want.ContainsKey($child.Name)) { Copy-SampleDir $child }
                # else: non-selected sample -> skip (do not descend)
            } else {
                $stack.Push($child.FullName)
            }
        } else {
            Copy-GlobalFile $child       # file not inside a sample folder
        }
    }
}

Write-Host "`n--- Samples ---" -ForegroundColor Green
foreach ($s in $Samples) {
    $n = if ($bySample.ContainsKey($s)) { $bySample[$s].Count } else { 0 }
    $flag = if ($n -eq 0) { "  <-- NOT FOUND" } else { "" }
    Write-Host ("{0}  ->  {1} folder(s){2}" -f $s, $n, $flag)
}
Write-Host ("`nGlobal files copied: {0}" -f $globalFiles) -ForegroundColor Green
Write-Host ("Destination: {0}" -f $DstRoot) -ForegroundColor Green
Write-Host "Verify the slice, then the full ~320 GB dataset can be deleted." -ForegroundColor Green

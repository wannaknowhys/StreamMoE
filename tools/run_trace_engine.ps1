param(
    [string]$Engine = 'place-c1c2-exp2',
    [int]$Runs = 1,
    [switch]$Off,
    [string]$Layers = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
    [int]$Prefill = 0,
    [string]$UbRanges = '',
    [int]$Bin = 0,
    [string]$Root = ''
)
# Dump stage tensors of one place engine to temp/<Root>/run_NN/ubNNNN/*.bin.
# Used for the A pre-change byte snapshot; see docs/A_TIER1_FUSION_HANDOFF*.md.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$tag = "$Engine" + $(if ($Off) { '_off' } else { '_on' })
if (-not $Root) { $Root = Join-Path $repo "temp\trace_$tag" }
if (Test-Path (Join-Path $Root 'MANIFEST.txt')) {
    throw "refusing to overwrite snapshot $Root (has MANIFEST.txt); use a new -Root"
}
if (Test-Path $Root) { Remove-Item $Root -Recurse -Force }
New-Item -ItemType Directory -Path $Root | Out-Null

for ($i = 1; $i -le $Runs; $i++) {
    $dir = Join-Path $Root ("run_{0:D2}" -f $i)
    New-Item -ItemType Directory -Path $dir | Out-Null
    $env:STREAM_MOE_TMP_TRACE_DIR = $dir
    $env:STREAM_MOE_TMP_TRACE_LAYERS = $Layers
    $env:STREAM_MOE_TMP_STAGE_TRACE = '1'
    $env:STREAM_MOE_TMP_TRACE_BIN = "$Bin"
    $env:STREAM_MOE_TMP_TRACE_PREFILL = "$Prefill"
    if ($UbRanges) { $env:STREAM_MOE_TMP_TRACE_UB_RANGES = $UbRanges } else { Remove-Item Env:STREAM_MOE_TMP_TRACE_UB_RANGES -ErrorAction SilentlyContinue }
    if ($Off) { $env:STREAM_MOE_TMP_NO_HEAD_FUSE = '1' } else { Remove-Item Env:STREAM_MOE_TMP_NO_HEAD_FUSE -ErrorAction SilentlyContinue }
    $start = Get-Date
    Write-Host "=== $tag run $i/$Runs -> $dir ==="
    & node (Join-Path $repo 'tools\run_bench.js') --flavors StreamMoE_dump_dbg --models olmoe --engines $Engine --tasks en 2>&1 | Tee-Object -FilePath (Join-Path $dir 'bench.log')
    $benchDirs = Get-ChildItem (Join-Path $repo 'temp\bench_outputs') -Directory |
        Where-Object { $_.LastWriteTime -ge $start } | Sort-Object LastWriteTime
    if (-not $benchDirs) { throw "run ${i}: no bench output dir" }
    $txt = Get-ChildItem $benchDirs[-1].FullName -File | Where-Object { $_.Name -like "*$Engine*" } | Select-Object -First 1
    if ($txt) { Copy-Item $txt.FullName (Join-Path $dir 'output.txt') }
    Remove-Item Env:STREAM_MOE_TMP_TRACE_DIR, Env:STREAM_MOE_TMP_TRACE_LAYERS, Env:STREAM_MOE_TMP_TRACE_UB_MAX, Env:STREAM_MOE_TMP_TRACE_UB_RANGES, Env:STREAM_MOE_TMP_TRACE_BIN, Env:STREAM_MOE_TMP_TRACE_PREFILL, Env:STREAM_MOE_TMP_STAGE_TRACE, Env:STREAM_MOE_TMP_NO_HEAD_FUSE -ErrorAction SilentlyContinue
}
Write-Host "all $Runs runs under $Root"

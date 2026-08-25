@echo off
setlocal enabledelayedexpansion
title StreamMoE Pure-LRU Cold-Start Benchmark (No Prior Stats, No KV Snapshot)

echo ===================================================================
echo   StreamMoE Pure-LRU Cold-Start Benchmark
echo   (Eviction: PURE LRU | Zero Prior Knowledge | No KV Snapshot)
echo ===================================================================
echo.

cd /d "%~dp0"

if not exist "bin\stream_moe_server.exe" (
    echo [StreamMoE] Building server binary...
    call build.bat build
    if !ERRORLEVEL! NEQ 0 (
        echo [-] Build failed! Exiting.
        pause
        exit /b 1
    )
)

if not exist "benchmark" mkdir benchmark

set MODEL_PATH=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set DRAFT_PATH=N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf

if not exist "%MODEL_PATH%" (
    echo [-] Model file not found at: %MODEL_PATH%
    echo Please verify the model drive path.
    pause
    exit /b 1
)

echo.
echo Configuration:
echo   - Model:           %MODEL_PATH%
echo   - Eviction Policy: PURE LRU (Zero Historical Knowledge)
echo   - RAM Pool:        71680 MB (70 GB Pinned VirtualLock)
echo   - VRAM Pool:       0 MB
echo   - Preload Policy:  none (Pure Cold Start)
echo   - KV Snapshot:     None (Fresh start without disk snapshot)
echo   - Profile Log:     benchmark\profile_pure_lru.jsonl
echo   - Output Text:     benchmark\conversation_pure_lru.txt
echo   - Report:          benchmark\BENCHMARK_REPORT_PURE_LRU.md
echo.

if exist "benchmark\profile_pure_lru.jsonl" del "benchmark\profile_pure_lru.jsonl"
if exist "benchmark\conversation_pure_lru.txt" del "benchmark\conversation_pure_lru.txt"

start /b "" "bin\stream_moe_server.exe" -m "%MODEL_PATH%" --draft-model "%DRAFT_PATH%" --host 127.0.0.1 --port 8999 --moe-ram-pool 71680 --moe-vram-pool 0 --eviction-policy lru -t 16 --profile-log "benchmark\profile_pure_lru.jsonl" > nul 2>&1

echo Waiting for server startup (approx 2-3 seconds)...
powershell -Command "Start-Sleep -Seconds 3"

echo Running 10-turn multi-turn prompt sequence...
node tools\bench_agent.js --profile-log "benchmark\profile_pure_lru.jsonl" --report-file "benchmark\BENCHMARK_REPORT_PURE_LRU.md" --output-file "benchmark\conversation_pure_lru.txt"

echo Stopping Pure-LRU server...
powershell -Command "Get-Process stream_moe_server -ErrorAction SilentlyContinue | Stop-Process -Force"

echo.
echo ===================================================================
echo  Pure-LRU Benchmark Completed Successfully!
echo ===================================================================
echo.
echo Files Generated:
echo   1. Profile JSONL Log:   benchmark\profile_pure_lru.jsonl
echo   2. Conversation Output: benchmark\conversation_pure_lru.txt
echo   3. Markdown Report:     benchmark\BENCHMARK_REPORT_PURE_LRU.md
echo.
pause
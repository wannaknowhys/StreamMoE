@echo off
setlocal enabledelayedexpansion
title StreamMoE Chinese Long-Horizon Benchmark Suite (Hybrid EST1 vs Pure LRU)

echo ===================================================================
echo    StreamMoE 10-Turn Chinese Long-Horizon Benchmark Suite
echo    [Model: DeepSeek-V4 MoE 150GB+ - 70GB Host RAM Pinned Pool]
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
set PROMPTS_ZH=benchmark\long_horizon_prompts_zh.jsonl

if not exist "%MODEL_PATH%" (
    echo [-] Model file not found at: %MODEL_PATH%
    echo Please verify the model drive path.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo  [TEST 1/2] 10-Turn Chinese Benchmark under HYBRID EST1 Eviction
echo ===================================================================
echo Configuration:
echo   - Prompts:         %PROMPTS_ZH% (10-Turn Chinese Architecture Dialogue)
echo   - Eviction Policy: HYBRID EST1 (Historical Stats + Adaptive Decay)
echo   - RAM Pool:        71680 MB (70 GB Pinned VirtualLock)
echo   - Profile Log:     benchmark\profile_zh_hybrid.jsonl
echo   - Output Text:     benchmark\conversation_zh_hybrid.txt
echo.

if exist "benchmark\profile_zh_hybrid.jsonl" del "benchmark\profile_zh_hybrid.jsonl"
if exist "benchmark\conversation_zh_hybrid.txt" del "benchmark\conversation_zh_hybrid.txt"

start /b "" "bin\stream_moe_server.exe" -m "%MODEL_PATH%" --draft-model "%DRAFT_PATH%" --host 127.0.0.1 --port 8999 --moe-ram-pool 71680 --moe-vram-pool 0 --eviction-policy hybrid -t 16 --profile-log "benchmark\profile_zh_hybrid.jsonl" > nul 2>&1

echo Waiting for server startup...
powershell -Command "Start-Sleep -Seconds 3"

echo Running 10-turn Chinese multi-turn prompt sequence...
node tools\bench_agent.js --prompts "%PROMPTS_ZH%" --profile-log "benchmark\profile_zh_hybrid.jsonl" --report-file "benchmark\BENCHMARK_REPORT_ZH_HYBRID.md" --output-file "benchmark\conversation_zh_hybrid.txt"

echo Stopping Server 1...
powershell -Command "Get-Process stream_moe_server -ErrorAction SilentlyContinue | Stop-Process -Force"
powershell -Command "Start-Sleep -Seconds 2"

echo.
echo ===================================================================
echo  [TEST 2/2] 10-Turn Chinese Benchmark under PURE LRU Eviction
echo ===================================================================
echo Configuration:
echo   - Prompts:         %PROMPTS_ZH% (10-Turn Chinese Architecture Dialogue)
echo   - Eviction Policy: PURE LRU (Zero Historical Knowledge)
echo   - RAM Pool:        71680 MB (70 GB Pinned VirtualLock)
echo   - Profile Log:     benchmark\profile_zh_pure_lru.jsonl
echo   - Output Text:     benchmark\conversation_zh_pure_lru.txt
echo.

if exist "benchmark\profile_zh_pure_lru.jsonl" del "benchmark\profile_zh_pure_lru.jsonl"
if exist "benchmark\conversation_zh_pure_lru.txt" del "benchmark\conversation_zh_pure_lru.txt"

start /b "" "bin\stream_moe_server.exe" -m "%MODEL_PATH%" --draft-model "%DRAFT_PATH%" --host 127.0.0.1 --port 8999 --moe-ram-pool 71680 --moe-vram-pool 0 --eviction-policy lru -t 16 --profile-log "benchmark\profile_zh_pure_lru.jsonl" > nul 2>&1

echo Waiting for server startup...
powershell -Command "Start-Sleep -Seconds 3"

echo Running 10-turn Chinese multi-turn prompt sequence...
node tools\bench_agent.js --prompts "%PROMPTS_ZH%" --profile-log "benchmark\profile_zh_pure_lru.jsonl" --report-file "benchmark\BENCHMARK_REPORT_ZH_PURE_LRU.md" --output-file "benchmark\conversation_zh_pure_lru.txt"

echo Stopping Server 2...
powershell -Command "Get-Process stream_moe_server -ErrorAction SilentlyContinue | Stop-Process -Force"

echo.
echo ===================================================================
echo  All Chinese Benchmark Tests Completed Successfully!
echo ===================================================================
echo.
echo Generated Files:
echo   1. benchmark\conversation_zh_hybrid.txt
echo   2. benchmark\conversation_zh_pure_lru.txt
echo   3. benchmark\BENCHMARK_REPORT_ZH_HYBRID.md
echo   4. benchmark\BENCHMARK_REPORT_ZH_PURE_LRU.md
echo.
pause
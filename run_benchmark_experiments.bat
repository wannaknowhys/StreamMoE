@echo off
setlocal enabledelayedexpansion
title StreamMoE Dual-Tier Long-Horizon Benchmark Suite

echo ===================================================================
echo    StreamMoE Long-Horizon Double-Tier Benchmark Experiments
echo ===================================================================
echo.

cd /d "%~dp0"

:: 1. Ensure binaries are built
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
echo ===================================================================
echo  [EXPERIMENT 1/2] 70GB Host RAM Pool (Single-Tier CPU + NVMe DIO)
echo ===================================================================
echo Configuration:
echo   - RAM Pool:    71680 MB (70 GB Pinned VirtualLock)
echo   - VRAM Pool:   0 MB
echo   - Threads:     16 Physical Cores
echo   - Profile Log: benchmark\profile_70G_ram.jsonl
echo.

if exist "benchmark\profile_70G_ram.jsonl" del "benchmark\profile_70G_ram.jsonl"

start /b "" "bin\stream_moe_server.exe" -m "%MODEL_PATH%" --draft-model "%DRAFT_PATH%" --host 127.0.0.1 --port 8999 --moe-ram-pool 71680 --moe-vram-pool 0 -t 16 --profile-log "benchmark\profile_70G_ram.jsonl" > nul 2>&1

:: Wait for server to initialize
echo Waiting for server startup (approx 2-3 seconds)...
powershell -Command "Start-Sleep -Seconds 3"

echo Running 10-turn multi-turn prompt sequence...
node tools\bench_agent.js

echo Stopping Experiment 1 server...
powershell -Command "Get-Process stream_moe_server -ErrorAction SilentlyContinue | Stop-Process -Force"
powershell -Command "Start-Sleep -Seconds 2"

echo.
echo ===================================================================
echo  [EXPERIMENT 2/2] 70GB Host RAM + 5GB GPU VRAM Dual-Tier Pool
echo ===================================================================
echo Configuration:
echo   - RAM Pool:    71680 MB (70 GB Pinned VirtualLock)
echo   - VRAM Pool:   5120 MB (5 GB GPU VRAM L1 Cache)
echo   - Threads:     16 Physical Cores
echo   - Profile Log: benchmark\profile_70G_ram_5G_vram.jsonl
echo.

if exist "benchmark\profile_70G_ram_5G_vram.jsonl" del "benchmark\profile_70G_ram_5G_vram.jsonl"

start /b "" "bin\stream_moe_server.exe" -m "%MODEL_PATH%" --draft-model "%DRAFT_PATH%" --host 127.0.0.1 --port 8999 --moe-ram-pool 71680 --moe-vram-pool 5120 -t 16 --profile-log "benchmark\profile_70G_ram_5G_vram.jsonl" > nul 2>&1

echo Waiting for server startup...
powershell -Command "Start-Sleep -Seconds 3"

echo Running 10-turn multi-turn prompt sequence...
node tools\bench_agent.js

echo Stopping Experiment 2 server...
powershell -Command "Get-Process stream_moe_server -ErrorAction SilentlyContinue | Stop-Process -Force"

echo.
echo ===================================================================
echo  All Benchmark Experiments Completed Successfully!
echo ===================================================================
echo.
echo Output Profile Logs:
echo   1. benchmark\profile_70G_ram.jsonl
echo   2. benchmark\profile_70G_ram_5G_vram.jsonl
echo.
echo Generated Summary Report:
echo   - benchmark\BENCHMARK_REPORT.md
echo.
pause
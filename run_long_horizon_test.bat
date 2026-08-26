@echo off
setlocal enabledelayedexpansion
rem =====================================================================
rem  StreamMoE Long-Horizon Test Runner (user-invoked, manually watchable)
rem  Usage: run_long_horizon_test.bat [en|zh]
rem  Memory safety: watch Task Manager; kill this window to abort.
rem =====================================================================

title StreamMoE Long-Horizon Test [%1] - watch RAM, close window to abort

set TAG=%1
if "%TAG%"=="" set TAG=en

rem --- kill any stale server on the port first ---
taskkill /F /IM stream_moe_server.exe >nul 2>&1
timeout /t 1 /nobreak >nul

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set PORT=8992
if /i "%TAG%"=="zh" (
    set PROMPTS=benchmark\long_horizon_prompts_zh.jsonl
) else (
    set PROMPTS=benchmark\long_horizon_prompts.jsonl
)

if not exist "bin\stream_moe_server.exe" (
    echo [-] stream_moe_server.exe missing. Run: build.bat build
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [-] Model not found: %MODEL%
    pause
    exit /b 1
)

if exist "benchmark\profile_real_%TAG%.jsonl" del "benchmark\profile_real_%TAG%.jsonl"
if exist "benchmark\conversation_real_%TAG%.txt" del "benchmark\conversation_real_%TAG%.txt"
if exist "benchmark\BENCHMARK_REPORT_REAL_%TAG%.md" del "benchmark\BENCHMARK_REPORT_REAL_%TAG%.md"

echo ===================================================================
echo  Starting server on port %PORT% (RAM pool 71680 MB, n=384, t=16)
echo  MEMORY WATCH: if RAM fills up, close this window NOW.
echo ===================================================================

start "stream_moe_server" /min cmd /c "bin\stream_moe_server.exe -m "%MODEL%" --host 127.0.0.1 --port %PORT% --moe-ram-pool 71680 --moe-vram-pool 0 --kv-placement ram -t 16 --temp 0.6 --top-p 0.95 -n 384 --profile-log benchmark\profile_real_%TAG%.jsonl > temp\server_%TAG%.log 2>&1"

echo Waiting for server readiness (max 120s)...
set READY=0
for /l %%i in (1,1,120) do (
    curl -s --max-time 2 http://127.0.0.1:%PORT%/health >nul 2>&1
    if !ERRORLEVEL!==0 (
        set READY=1
        goto ready
    )
    timeout /t 1 /nobreak >nul
)
:ready
if "%READY%"=="0" (
    echo [-] Server did not become ready in 120s. Aborting.
    taskkill /F /IM stream_moe_server.exe >nul 2>&1
    type temp\server_%TAG%.log 2>nul
    pause
    exit /b 1
)
echo [+] Server ready.

echo ===================================================================
echo  Running 10-turn benchmark (%PROMPTS%)
echo ===================================================================
node tools\bench_agent.js --port %PORT% --prompts "%PROMPTS%" --profile-log "benchmark\profile_real_%TAG%.jsonl" --report-file "benchmark\BENCHMARK_REPORT_REAL_%TAG%.md" --output-file "benchmark\conversation_real_%TAG%.txt"
set BENCH_EXIT=%ERRORLEVEL%

echo [+] Stopping server...
taskkill /F /IM stream_moe_server.exe >nul 2>&1

echo.
echo ===================================================================
echo  DONE. Outputs:
echo    conversation : benchmark\conversation_real_%TAG%.txt
echo    telemetry    : benchmark\profile_real_%TAG%.jsonl
echo    server log   : temp\server_%TAG%.log
echo  memwatch log  : temp\memwatch_*.log (if diagnostic branch)
echo  bench exit code: %BENCH_EXIT%
echo ===================================================================
pause
exit /b %BENCH_EXIT%

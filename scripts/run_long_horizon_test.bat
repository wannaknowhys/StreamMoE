@echo off
setlocal enabledelayedexpansion
rem =====================================================================
rem  StreamMoE Long-Horizon Test Runner (user-invoked, manually watchable)
rem  Usage: scripts\run_long_horizon_test.bat [en|zh] [tag]
rem  Memory safety: watch Task Manager; close this window to abort.
rem =====================================================================

title StreamMoE Long-Horizon Test [%1] - watch RAM, close window to abort

set TAG_ARG=%1
if "%TAG_ARG%"=="" set TAG_ARG=en
set BTAG=%2
if "%BTAG%"=="" set BTAG=main

rem --- kill any stale server on the port first ---
taskkill /F /IM stream_moe_server.exe >nul 2>&1
timeout /t 1 /nobreak >nul

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set PORT=8992
if /i "%TAG_ARG%"=="zh" (
    set PROMPTS=benchmark\prompts\long_horizon_prompts_zh.jsonl
) else (
    set PROMPTS=benchmark\prompts\long_horizon_prompts.jsonl
)
set BIN=build\%BTAG%\bin\stream_moe_server.exe

if not exist "%BIN%" (
    echo [-] %BIN% missing. Run: build.bat build %BTAG%
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [-] Model not found: %MODEL%
    pause
    exit /b 1
)

if exist "benchmark\results\profile_real_%TAG_ARG%.jsonl" del "benchmark\results\profile_real_%TAG_ARG%.jsonl"
if exist "benchmark\results\conversation_real_%TAG_ARG%.txt" del "benchmark\results\conversation_real_%TAG_ARG%.txt"
if exist "benchmark\results\BENCHMARK_REPORT_REAL_%TAG_ARG%.md" del "benchmark\results\BENCHMARK_REPORT_REAL_%TAG_ARG%.md"

echo ===================================================================
echo  Starting server on port %PORT% (RAM pool 71680 MB, n=384, t=16)
echo  MEMORY WATCH: if RAM fills up, close this window NOW.
echo ===================================================================

start "stream_moe_server" /min cmd /c ""%BIN%" -m "%MODEL%" --host 127.0.0.1 --port %PORT% --moe-ram-pool 71680 --moe-vram-pool 0 --kv-placement ram -t 16 --temp 0.6 --top-p 0.95 -n 384 --profile-log benchmark\results\profile_real_%TAG_ARG%.jsonl > build\%BTAG%\server_%TAG_ARG%.log 2>&1"

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
    type build\%BTAG%\server_%TAG_ARG%.log 2>nul
    pause
    exit /b 1
)
echo [+] Server ready.

echo ===================================================================
echo  Running 10-turn benchmark (%PROMPTS%)
echo ===================================================================
node tools\bench_agent.js --port %PORT% --prompts "%PROMPTS%" --profile-log "benchmark\results\profile_real_%TAG_ARG%.jsonl" --report-file "benchmark\results\BENCHMARK_REPORT_REAL_%TAG_ARG%.md" --output-file "benchmark\results\conversation_real_%TAG_ARG%.txt"
set BENCH_EXIT=%ERRORLEVEL%

echo [+] Stopping server...
taskkill /F /IM stream_moe_server.exe >nul 2>&1

echo.
echo ===================================================================
echo  DONE. Outputs (tag=%TAG_ARG%, build=%BTAG%):
echo    conversation : benchmark\results\conversation_real_%TAG_ARG%.txt
echo    telemetry    : benchmark\results\profile_real_%TAG_ARG%.jsonl
echo    server log   : build\%BTAG%\server_%TAG_ARG%.log
echo    memwatch log : %TEMP%\memwatch_*.log (if memwatch patch applied)
echo  bench exit code: %BENCH_EXIT%
echo ===================================================================
pause
exit /b %BENCH_EXIT%

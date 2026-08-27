@echo off
setlocal enabledelayedexpansion
rem =====================================================================
rem  StreamMoE Long-Horizon Test Runner (user-invoked, manually watchable)
rem  Runs the UPSTREAM llama-server (build\<tag>\llama-build\bin). The server
rem  stays UP for the WHOLE run; all turns are chained (multi-turn context)
rem  against the SAME server. After EVERY turn the transcript / telemetry are
rem  written. Then the server is stopped.
rem  Usage: scripts\run_long_horizon_test.bat [en|zh] [tag]
rem  Memory safety: watch Task Manager; close this window to abort.
rem =====================================================================

title StreamMoE Long-Horizon Test [%1] - watch RAM, close window to abort

set TAG_ARG=%1
if "%TAG_ARG%"=="" set TAG_ARG=en
set BTAG=%2
if "%BTAG%"=="" set BTAG=main

rem --- kill any stale server on the port first ---
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 1 /nobreak >nul

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set PORT=8992
if /i "%TAG_ARG%"=="zh" (
    set PROMPTS=benchmark\prompts\long_horizon_prompts_zh.jsonl
) else (
    set PROMPTS=benchmark\prompts\long_horizon_prompts.jsonl
)
set BIN=build\%BTAG%\llama-build\bin\llama-server.exe

if not exist "%BIN%" (
    echo [-] %BIN% missing. Run: build.bat llamalibs %BTAG%
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

echo ===================================================================
echo  Starting server on port %PORT% (RAM pool 71680 MB, t=16, temp=1.0, top_p=0.95)
echo  --no-warmup: skip the empty startup forward pass (MoE: avoids cold-disk
echo                expert loading at boot; experts load on first request)
echo  MEMORY WATCH: if RAM fills up, close this window NOW.
echo ===================================================================

rem Upstream llama-server; --fit off avoids the 162GB-model fit abort.
rem -c 8192 is the short-run default - raise it for long-context runs.
start "llama-server" /min cmd /c ""%BIN%" -m "%MODEL%" --host 127.0.0.1 --port %PORT% -c 8192 -t 16 --temp 1.0 --top-p 0.95 --n-predict 384 --cache-type-k q8_0 --cache-type-v q8_0 --expert-backend --moe-ram-pool 71680 --fit off --no-warmup --no-webui > build\%BTAG%\server_%TAG_ARG%.log 2>&1"

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
    taskkill /F /IM llama-server.exe >nul 2>&1
    type build\%BTAG%\server_%TAG_ARG%.log 2>nul
    pause
    exit /b 1
)
echo [+] Server ready.

echo ===================================================================
echo  Running ALL turns chained against the SAME server (%PROMPTS%)
echo ===================================================================
node tools\bench_agent.js --port %PORT% --prompts "%PROMPTS%" --profile-log "benchmark\results\profile_real_%TAG_ARG%.jsonl" --output-file "benchmark\results\conversation_real_%TAG_ARG%.txt"
set BENCH_EXIT=%ERRORLEVEL%

echo [+] Stopping server...
taskkill /F /IM llama-server.exe >nul 2>&1

echo.
echo ===================================================================
echo  DONE. Outputs (tag=%TAG_ARG%, build=%BTAG%):
echo    conversation : benchmark\results\conversation_real_%TAG_ARG%.txt
echo    telemetry    : benchmark\results\profile_real_%TAG_ARG%.jsonl
echo    server log   : build\%BTAG%\server_%TAG_ARG%.log
echo  bench exit code: %BENCH_EXIT%
echo ===================================================================
pause
exit /b %BENCH_EXIT%

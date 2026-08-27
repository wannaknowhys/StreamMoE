@echo off
rem =====================================================================
rem  StreamMoE Prefill Cross-Validation + Expert-History over a jsonl prompt set.
rem  Uses the UPSTREAM llama-server (build\<tag>\llama-build\bin).
rem  Semantics: a server is started per mode (std, then moe) and STAYS UP for
rem  the WHOLE prompt set - all turns are chained against that same server.
rem  After every turn the server is stopped; verification + simulation run on
rem  the flushed exports.
rem  Usage: scripts\run_prefill_verify.bat [en|zh] [tag]   (default en, main)
rem
rem  NOTE: the prefill/KV/expert-history EXPORT (LLM_EXPORT_DIR + prefill-export
rem  patch) is NOT yet re-wired for the upstream-server architecture. Without it
rem  Phase 3 (verify/simulate) will report missing export files.
rem =====================================================================
setlocal enabledelayedexpansion
set LANG=%~1
if "%LANG%"=="" set LANG=en
set BTAG=%~2
if "%BTAG%"=="" set BTAG=main

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set PORT=8992
if /i "%LANG%"=="zh" (
    set PROMPTS=benchmark\prompts\long_horizon_prompts_zh.jsonl
) else (
    set PROMPTS=benchmark\prompts\long_horizon_prompts.jsonl
)
set BIN=build\%BTAG%\llama-build\bin\llama-server.exe
set STD_DIR=temp\export_%LANG%_std
set MOE_DIR=temp\export_%LANG%_moe

if not exist "%BIN%" ( echo [-] %BIN% missing. Run: build.bat llamalibs %BTAG% & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [-] Model not found. & pause & exit /b 1 )

taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 1 /nobreak >nul
rmdir /s /q "%STD_DIR%" "%MOE_DIR%" 2>nul
mkdir "%STD_DIR%" "%MOE_DIR%" 2>nul

echo ===================================================================
echo  Phase 1: STD server (no --expert-backend) chains ALL prompts once,
echo  then is stopped.
echo ===================================================================
set "LLM_EXPORT_DIR=%CD%\%STD_DIR%"
start "llama-server_std" /min cmd /c ""%BIN%" -m "%MODEL%" --host 127.0.0.1 --port %PORT% -c 8192 -t 16 --temp 1.0 --top-p 0.95 --n-predict 384 --cache-type-k q8_0 --cache-type-v q8_0 --fit off --no-warmup --no-webui > build\%BTAG%\server_%LANG%_std.log 2>&1"
set READY=0
for /l %%i in (1,1,120) do ( curl -s --max-time 2 http://127.0.0.1:%PORT%/health >nul 2>&1 & if !ERRORLEVEL!==0 ( set READY=1 & goto std_ready ) & timeout /t 1 /nobreak >nul )
:std_ready
if "%READY%"=="0" ( echo [-] STD server not ready. Aborting. & taskkill /F /IM llama-server.exe >nul 2>&1 & pause & exit /b 1 )
echo [+] STD server ready.
node tools\bench_agent.js --port %PORT% --prompts "%PROMPTS%" --output-file benchmark\results\conversation_prefill_%LANG%_std.txt --profile-log benchmark\results\profile_prefill_%LANG%_std.jsonl
echo [+] Stopping STD server...
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo ===================================================================
echo  Phase 2: MOE server (--expert-backend) chains ALL prompts once,
echo  then is stopped.
echo ===================================================================
set "LLM_EXPORT_DIR=%CD%\%MOE_DIR%"
start "llama-server_moe" /min cmd /c ""%BIN%" -m "%MODEL%" --host 127.0.0.1 --port %PORT% -c 8192 -t 16 --temp 1.0 --top-p 0.95 --n-predict 384 --cache-type-k q8_0 --cache-type-v q8_0 --expert-backend --moe-ram-pool 71680 --fit off --no-warmup --no-webui > build\%BTAG%\server_%LANG%_moe.log 2>&1"
set READY=0
for /l %%i in (1,1,120) do ( curl -s --max-time 2 http://127.0.0.1:%PORT%/health >nul 2>&1 & if !ERRORLEVEL!==0 ( set READY=1 & goto moe_ready ) & timeout /t 1 /nobreak >nul )
:moe_ready
if "%READY%"=="0" ( echo [-] MOE server not ready. Aborting. & taskkill /F /IM llama-server.exe >nul 2>&1 & pause & exit /b 1 )
echo [+] MOE server ready.
node tools\bench_agent.js --port %PORT% --prompts "%PROMPTS%" --output-file benchmark\results\conversation_prefill_%LANG%_moe.txt --profile-log benchmark\results\profile_prefill_%LANG%_moe.jsonl
echo [+] Stopping MOE server...
taskkill /F /IM llama-server.exe >nul 2>&1

echo.
echo ===================================================================
echo  Phase 3: verify + simulate on flushed exports
echo ===================================================================
if exist "%STD_DIR%\prefill_export.bin" (
    node tools\verify_prefill.js "%STD_DIR%\prefill_export.bin" "%MOE_DIR%\prefill_export.bin"
) else (
    echo [-] no prefill_export.bin - export not re-wired for the upstream server yet.
)
echo.
if exist "%MOE_DIR%\expert_history.bin" (
    node tools\simulate_cache.js "%MOE_DIR%\expert_history.bin" 11008
) else (
    echo no expert_history.bin (export not re-wired yet)
)
echo.
echo  DONE. Results:
echo    conversation : benchmark\results\conversation_prefill_%LANG%_{std,moe}.txt
echo    exports      : %STD_DIR% / %MOE_DIR%
echo ===================================================================
pause
endlocal

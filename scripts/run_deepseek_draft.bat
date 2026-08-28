@echo off
pushd "%~dp0\.."
setlocal
rem =====================================================================
rem  DeepSeek-V4-Flash via the UPSTREAM llama-server + dspark draft + /metrics.
rem  Prefill/trace experiment launcher:
rem    - ctx 1048576 (full), agentic temp=1.0 top_p=0.95, q8_0 KV
rem    - 70GB expert pool, --fit off, --no-warmup
rem    - LLM_EXPORT_DIR = benchmark\trace (persistent trace/stats)
rem  Usage:
rem    1) double-click to start the server (wait for "listening on :8993")
rem    2) node tools\prefill_from_trace.js --tokens 10000 --out benchmark\trace
rem    3) Ctrl+C to stop (writes expert_history_*.bin + prefill_export_*.bin)
rem    4) node tools\verify_lmhead_top.js ... (see benchmark\trace\README.md)
rem  Metrics: curl http://127.0.0.1:8993/metrics
rem =====================================================================
title DeepSeek-V4-Flash server (draft + metrics) - watch RAM, Ctrl+C to stop

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set DRAFT=N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
set BIN=build\main\llama-build\bin\llama-server.exe
set PORT=8993
set "LLM_EXPORT_DIR=%~dp0..\benchmark\trace"
if not exist "%LLM_EXPORT_DIR%" mkdir "%LLM_EXPORT_DIR%"

if not exist "%BIN%" ( echo [-] %BIN% missing. Run: build.bat llamalibs main & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [-] Model not found: %MODEL% & pause & exit /b 1 )
if not exist "%DRAFT%" ( echo [-] Draft model not found: %DRAFT% & pause & exit /b 1 )

echo Loading DeepSeek-V4-Flash + draft (expert pool 70GB, q8 KV, no-warmup, ctx 1M, metrics) ...
rem dspark MTP confidence decays fast (t1~0.99 t2~0.95 t3~0.6-0.8): n_min=1 p_min=0.6
"%BIN%" -m "%MODEL%" --model-draft "%DRAFT%" --spec-draft-n-max 5 --spec-draft-n-min 1 --spec-draft-p-min 0.6 ^
    --expert-backend --moe-ram-pool 71680 ^
    --cache-type-k q8_0 --cache-type-v q8_0 ^
    --fit off --no-warmup -c 1048576 -t 16 --temp 1.0 --top-p 0.95 --metrics ^
    --host 127.0.0.1 --port %PORT% --no-webui

echo.
echo (server exited - was it killed? Ctrl+C sends SIGINT - clean shutdown)
pause
endlocal
popd

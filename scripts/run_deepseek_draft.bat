@echo off
pushd "%~dp0\.."
setlocal
rem =====================================================================
rem  DeepSeek-V4-Flash via the UPSTREAM llama-server with SPECULATIVE
rem  DECODING (draft model = dspark MTP) + /metrics endpoint.
rem
rem  Usage: double-click. Then:
rem    - chat:    node tools\chat_cli.js --port 8993
rem    - metrics: curl http://127.0.0.1:8993/metrics
rem              (prompt/predict tok/s, n_prompt_cached, n_decode,
rem               n_busy_slots, draft stats n_draft_tokens/accepted/
rem               verif_steps/accepted_per_pos)
rem
rem  Conventions (docs/SMOKE_TESTING.md): 70GB pool, --fit off, --no-warmup,
rem  q8_0 KV, agentic temp=1.0 top_p=0.95.
rem  First reply can take minutes (cold N: disk expert loading).
rem =====================================================================
title DeepSeek-V4-Flash server (draft + metrics) - watch RAM, Ctrl+C to stop

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set DRAFT=N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
set BIN=build\main\llama-build\bin\llama-server.exe
set PORT=8993

if not exist "%BIN%" ( echo [-] %BIN% missing. Run: build.bat llamalibs main & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [-] Model not found: %MODEL% & pause & exit /b 1 )
if not exist "%DRAFT%" ( echo [-] Draft model not found: %DRAFT% & pause & exit /b 1 )

echo Loading DeepSeek-V4-Flash + draft (expert pool 70GB, q8 KV, no-warmup, metrics) ...
"%BIN%" -m "%MODEL%" --model-draft "%DRAFT%" --draft-max 5 --draft-min 3 --draft-p-min 0.9 ^
    --expert-backend --moe-ram-pool 71680 ^
    --cache-type-k q8_0 --cache-type-v q8_0 ^
    --fit off --no-warmup -c 8192 -t 16 --temp 1.0 --top-p 0.95 --metrics ^
    --host 127.0.0.1 --port %PORT% --no-webui

echo.
echo (server exited - was it killed? Ctrl+C sends SIGINT -> clean shutdown)
pause
endlocal
popd

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
rem prefill/trace 实验：导出专家历史 + prefill 到持久统计目录（非 temp）
set "LLM_EXPORT_DIR=%~dp0..\benchmark\trace"
if not exist "%LLM_EXPORT_DIR%" mkdir "%LLM_EXPORT_DIR%"

if not exist "%BIN%" ( echo [-] %BIN% missing. Run: build.bat llamalibs main & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [-] Model not found: %MODEL% & pause & exit /b 1 )
if not exist "%DRAFT%" ( echo [-] Draft model not found: %DRAFT% & pause & exit /b 1 )

echo Loading DeepSeek-V4-Flash + draft (expert pool 70GB, q8 KV, no-warmup, metrics, ctx 1M) ...
rem Draft params tuned for dspark MTP: its confidence decays fast (t1~0.99, t2~0.95, t3~0.6-0.8),
rem so n_min=3 + p_min=0.9 cleared every draft (<3 tokens) - n_min=1 p_min=0.6 lets it through.
"%BIN%" -m "%MODEL%" --model-draft "%DRAFT%" --spec-draft-n-max 5 --spec-draft-n-min 1 --spec-draft-p-min 0.6 ^
    --expert-backend --moe-ram-pool 71680 ^
    --cache-type-k q8_0 --cache-type-v q8_0 ^
    --fit off --no-warmup -c 1048576 -t 16 --temp 1.0 --top-p 0.95 --metrics ^
    --host 127.0.0.1 --port %PORT% --no-webui

echo.
echo (server exited - was it killed? Ctrl+C sends SIGINT -> clean shutdown)
pause
endlocal
popd

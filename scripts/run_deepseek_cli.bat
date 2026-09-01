@echo off
pushd "%~dp0\.."
setlocal
rem =====================================================================
rem  DeepSeek-V4-Flash via the UPSTREAM llama-cli + StreamMoE expert pool.
rem  Double-click to run. Type text at the ">" prompt to chat.
rem
rem  Conventions (docs/SMOKE_TESTING.md):
rem    - 70GB pool for big MoE (--moe-ram-pool 71680)
rem    - --fit off (162GB model would abort the auto memory-fit)
rem    - --no-warmup (skip the empty startup pass; experts load on first input)
rem    - short-run ctx 8192; q8_0 KV; agentic temp=1.0 top_p=0.95
rem
rem  NOTE: first reply can take minutes - the expert pool loads from the cold
rem  N: disk on demand. Later replies reuse the pool.
rem =====================================================================
title DeepSeek-V4-Flash CLI (expert pool) - type text to chat, Ctrl+C to quit

set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set BIN=build\main\llama-build\bin\llama-cli.exe

if not exist "%BIN%" (
    echo [-] %BIN% missing. Run: build.bat llamalibs main
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [-] Model not found: %MODEL%
    pause
    exit /b 1
)

echo Loading DeepSeek-V4-Flash (expert pool 70GB, q8_0 KV, no-warmup) ...
"%BIN%" -m "%MODEL%" --expert-backend --moe-ram-pool 71680 ^
    --cache-type-k q8_0 --cache-type-v q8_0 ^
    --fit off --no-warmup -c 8192 -t 16 --temp 1.0 --top-p 0.95

echo.
echo (exited)
pause
endlocal
popd

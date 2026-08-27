@echo off
rem Start the local StreamMoE server for interactive play.
rem Usage: scripts\start_server.bat [extra args...]
rem   Ctrl+C to stop cleanly (SIGINT -> graceful shutdown).
rem   All /v1/chat/completions request bodies are appended to temp\server_prompts.log
rem     (via --prompt-log) so you keep the prompts you sent.
setlocal
set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
set PORT=8992
echo ===================================================================
echo  Starting StreamMoE server on http://127.0.0.1:%PORT%
echo  - Ctrl+C to stop cleanly
echo  - prompts appended to temp\server_prompts.log
echo  - sampling temp=1.0 top_p=0.95 (agentic; see docs\SAMPLING.md)
echo  - max output length -n 384000 (DeepSeek official high/max effort limit;
echo    a request "max_tokens" may only lower it)
echo  - context -c 1048576 (watch the printed KV Cache Memory to see the cap)
echo  - KV cache --cache-type q8_0 --no-swa-full (1M ctx KV: 49.7GB f16+full-size
echo    -> 3.6GB; windowed raw + compressed long-range is the DeepSeek design)
echo ===================================================================
mkdir temp 2>nul
build\main\bin\stream_moe_server.exe -m "%MODEL%" --host 127.0.0.1 --port %PORT% --moe-ram-pool 71680 --kv-placement ram -t 16 --temp 1.0 --top-p 0.95 -c 1048576 -n 384000 --cache-type q8_0 --no-swa-full --prompt-log temp\server_prompts.log %*
endlocal

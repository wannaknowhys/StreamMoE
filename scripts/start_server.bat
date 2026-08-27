@echo off
rem Start the local StreamMoE server for interactive play.
rem Now runs the UPSTREAM llama-server (clang-cl + VS2026 build) - see docs/UPSTREAM_TOOLS_MIGRATION.md.
rem Usage: scripts\start_server.bat [extra args...]
rem   Ctrl+C to stop cleanly.
rem   Prompt backup (--prompt-log) will be re-added in M3 with the route-B args.
rem   Default model is a fast smoke model; switch to DeepSeek-V4-Flash once the
rem   expert-pool subsystem (M3) lands - see docs/SMOKE_TESTING.md.
setlocal
set MODEL=F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf
set PORT=8992
echo ===================================================================
echo  Starting StreamMoE server (upstream llama-server) on http://127.0.0.1:%PORT%
echo  - Ctrl+C to stop cleanly
echo  - sampling temp=1.0 top_p=0.95 (agentic; see docs\SAMPLING.md)
echo  - max output --n-predict 384000 (DeepSeek official high/max effort limit;
echo    a request "max_tokens" may only lower it)
echo  - context -c 1048576 (watch the printed KV cells to see the cap)
echo  - KV cache --cache-type-k q8_0 --cache-type-v q8_0 (upstream params;
echo    SWA cache is windowed by default; --swa-full opts into full-size)
echo  - --no-warmup (skip the empty startup forward pass - for MoE this avoids
echo    triggering on-demand expert loading from the cold disk at boot)
echo  - MODEL = %MODEL%
echo ===================================================================
build\main\llama-build\bin\llama-server.exe -m "%MODEL%" --host 127.0.0.1 --port %PORT% -c 1048576 -t 16 --temp 1.0 --top-p 0.95 --n-predict 384000 --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup --no-webui %*
endlocal

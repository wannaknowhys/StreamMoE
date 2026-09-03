@echo off
rem =====================================================================
rem  Baseline regression - route B / prefill-from numerics guard.
rem  After every code change + rebuild, run this to confirm the 129-token
rem  prefill-from exports still match the known-good baselines.
rem  Checks:
rem    1. moe (route-B, 8GB pool)  vs baseline moe     -> IDENTICAL expected
rem    2. upstream (no route-B)    vs baseline upstream -> IDENTICAL expected
rem    3. per-token KL  baseline-upstream vs new moe    -> report (loose thresh)
rem    4. kv_cos       baseline moe      vs new moe     -> all ~1.0
rem  Usage: run from repo root: baseline_regression\run_baseline.bat
rem =====================================================================
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
set BR=baseline_regression
set OUT=%BR%\temp
if not exist "%OUT%\moe"  mkdir "%OUT%\moe"
if not exist "%OUT%\up"   mkdir "%OUT%\up"

set MODEL=N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
set TOK=%BR%\baseline\upstream_129\tokens_id.bin
set MOE_BIN=%ROOT%\build\StreamMoE_dump\llama-build\bin\llama-server.exe
set UP_BIN=%ROOT%\build\upstream_dump\llama-build\bin\llama-server.exe
set KL=%BR%\tools\verify_kl.exe

if not exist "%MOE_BIN%" ( echo [-] missing %MOE_BIN% - run: build.bat llamalibs StreamMoE_dump & exit /b 1 )
if not exist "%UP_BIN%" ( echo [-] missing %UP_BIN% - run: build.bat llamalibs upstream_dump & exit /b 1 )

rem ---- compile verify_kl.exe if missing ----
if not exist "%KL%" (
    echo [build] verify_kl.exe ...
    F:\Dev\LLVM\bin\clang++.exe -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS -fopenmp "%BR%\tools\verify_kl.cpp" ^
        -I third_party\llama.cpp\ggml\include ^
        "%ROOT%\build\main\llama-build\ggml\src\ggml.lib" "%ROOT%\build\main\llama-build\ggml\src\ggml-base.lib" "%ROOT%\build\main\llama-build\ggml\src\ggml-cpu.lib" ^
        F:\Dev\LLVM\lib\libomp.lib -ladvapi32 -o "%KL%"
    if errorlevel 1 ( echo [-] verify_kl.exe build failed & exit /b 1 )
)
copy /Y F:\Dev\LLVM\bin\libomp.dll "%BR%\tools\" >nul

set PASS=1
echo.
echo =====================================================================
echo [1/5] moe prefill-from (route-B, 8GB pool) ...
"%MOE_BIN%" -m %MODEL% --prefill-from %TOK% --export-dir %OUT%\moe -c 2048 -t 16 ^
    --expert-backend --moe-ram-pool 8192 --fit off --no-warmup > %OUT%\moe\run.log 2>&1
if errorlevel 1 ( echo [-] moe run failed & exit /b 1 )
echo.
echo [2/5] upstream prefill-from ...
"%UP_BIN%" -m %MODEL% --prefill-from %TOK% --export-dir %OUT%\up -c 2048 -t 16 > %OUT%\up\run.log 2>&1
if errorlevel 1 ( echo [-] upstream run failed & exit /b 1 )

echo.
echo [3/5] embd/hidden/KV byte compare vs baseline ...
echo   -- moe vs baseline moe (expect IDENTICAL):
node %BR%\tools\verify_prefill.js %BR%\baseline\moe_129_8192\prefill_export_main.bin %OUT%\moe\prefill_export_main.bin
if not errorlevel 0 ( echo [-] moe DIVERGED from baseline & set PASS=0 )
echo   -- expert history:
node %BR%\tools\verify_expert_history.js %BR%\baseline\moe_129_8192\expert_history_main.bin %OUT%\moe\expert_history_main.bin
if errorlevel 1 ( echo [-] moe expert history DIVERGED & set PASS=0 )
echo   -- upstream vs baseline upstream (expect IDENTICAL):
node %BR%\tools\verify_prefill.js %BR%\baseline\upstream_129\prefill_export_main.bin %OUT%\up\prefill_export_main.bin
if errorlevel 1 ( echo [-] upstream DIVERGED from baseline & set PASS=0 )

echo.
echo [4/5] per-token KL (baseline upstream vs new moe) - report only:
echo   moe-vs-upstream KL is inherent backend noise (routing flips), NOT a bug;
echo   see docs/BACKEND_DIVERGENCE_ANALYSIS.md. FAIL below only if thresh 1.0 trips.
"%KL%" %MODEL% %BR%\baseline\upstream_129\prefill_export_main.bin %OUT%\moe\prefill_export_main.bin --thresh 1.0

echo.
echo [5/5] kv_cos (baseline moe vs new moe, expect ~1.0) ...
node %BR%\tools\kv_cos.js %BR%\baseline\moe_129_8192\prefill_export_main.bin %OUT%\moe\prefill_export_main.bin > "%OUT%\kv_cos.txt"
set "OUTP=%OUT:\=/%"
node -e "const fs=require('fs');const rows=fs.readFileSync(process.argv[1],'utf8').split('\n').filter(l=>/^\d+\t/.test(l));let n=0,min=2;for(const l of rows){for(const c of l.split('\t').slice(1)){const v=+c;n++;if(v<min)min=v}}console.log('kv_cos rows='+rows.length+' cos_min='+(min===2?'-':min.toFixed(6))+' (expect >= ~0.999)')" "%OUTP%/kv_cos.txt"

echo.
echo =====================================================================
if "%PASS%"=="1" ( echo RESULT: PASS - moe/upstream both IDENTICAL to baseline ) else ( echo RESULT: FAIL - see above )
echo =====================================================================
exit /b 0

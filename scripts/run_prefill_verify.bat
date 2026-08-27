@echo off
rem Run prefill cross-validation + expert-history simulation over a jsonl prompt set.
rem Usage: run_prefill_verify.bat [en|zh] [n_prompts]   (default en, 1 prompt)
rem Requires the export build (patches applied, see patches/README.md).
setlocal
set LANG=%~1
if "%LANG%"=="" set LANG=en
set N=%~2
if "%N%"=="" set N=1
echo [run_prefill_verify] lang=%LANG% prompts=%N%
node tools\run_export_batch.js %LANG% %N%
endlocal

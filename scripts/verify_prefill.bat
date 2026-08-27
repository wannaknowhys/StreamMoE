@echo off
rem Quick prefill cross-validation on ONE prompt: std vs --expert-backend export, then verify.
rem Usage: verify_prefill.bat ["prompt text"]   (default: "Say hi.")
rem Requires the export build (patches applied, see patches/README.md).
setlocal
set "PROMPT=%~1"
if "%PROMPT%"=="" set "PROMPT=Say hi."
set MODEL=N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf

rmdir /s /q temp\export_std 2>nul
rmdir /s /q temp\export_moe 2>nul
mkdir temp\export_std 2>nul
mkdir temp\export_moe 2>nul

set "LLM_EXPORT_DIR=%CD%\temp\export_std"
echo [std] running ...
build\main\bin\stream_moe.exe -m %MODEL% --moe-ram-pool 71680 --temp 0 -c 2048 -t 16 -p "%PROMPT%" -n 2

set "LLM_EXPORT_DIR=%CD%\temp\export_moe"
echo [moe] running ...
build\main\bin\stream_moe.exe -m %MODEL% --moe-ram-pool 71680 --expert-backend --temp 0 -c 2048 -t 16 -p "%PROMPT%" -n 2

echo.
echo ===== verify_prefill (std vs moe) =====
node tools\verify_prefill.js temp\export_std\prefill_export.bin temp\export_moe\prefill_export.bin
echo.
echo ===== simulate_cache (moe expert history) =====
if exist temp\export_moe\expert_history.bin (
    node tools\simulate_cache.js temp\export_moe\expert_history.bin 11008
) else (
    echo no expert_history.bin
)
endlocal

@echo off
rem ===========================================================================
rem Converter matrix verification (v2-centric block abstraction, docs sec 9).
rem original (N: source) -> v1/v2/v2chunk baseline, then each of v1/v2/v2chunk
rem as source produces all three formats and is byte-compared to the baseline
rem (tools/cmp_gguf.js streams both files). Stops on first mismatch.
rem
rem usage: verify_convert_matrix.bat <workdir>   (workdir needs ~3x model free)
rem   e.g. on a ramdisk: verify_convert_matrix.bat R:\conv
rem
rem Requires: temp\sm_env.bat (SM_GEMMA_ORIG), temp\stream_moe_convertd.exe
rem (build line in tools\stream_moe_convertd.cpp header).
rem ===========================================================================
setlocal
set DIR=%~1
if "%DIR%"=="" ( echo usage: verify_convert_matrix.bat ^<workdir^> & exit /b 1 )
mkdir "%DIR%" 2>nul
if not exist "%DIR%" ( echo cannot create %DIR% & exit /b 1 )
pushd "%~dp0.."
call temp\sm_env.bat 2>nul
if not exist "%SM_GEMMA_ORIG%" ( echo [verify] SM_GEMMA_ORIG not set/valid in temp\sm_env.bat & exit /b 1 )

set ORIG=%SM_GEMMA_ORIG%
set RATIO=8:9:9:7:9
set CH=5

echo [step 1/4] original -^> v1/v2/v2chunk baseline
node tools\stream_moe_convert.js -m "%ORIG%" -o "%DIR%\v1_base.gguf" --format v1 || goto fail
node tools\stream_moe_convert.js -m "%ORIG%" -o "%DIR%\v2_base.gguf" --format v2 || goto fail
node tools\stream_moe_convert.js -m "%ORIG%" -o "%DIR%\chunk_base" --format v2chunk --chunks %CH% --ratio %RATIO% || goto fail

echo [step 2/4] v1 source -^> v1/v2/v2chunk
node tools\stream_moe_convert.js -m "%DIR%\v1_base.gguf" -o "%DIR%\t2_v1.gguf" --format v1 || goto fail
call :cmp "%DIR%\v1_base.gguf" "%DIR%\t2_v1.gguf" "v1-to-v1" || goto fail
node tools\stream_moe_convert.js -m "%DIR%\v1_base.gguf" -o "%DIR%\t2_v2.gguf" --format v2 || goto fail
call :cmp "%DIR%\v2_base.gguf" "%DIR%\t2_v2.gguf" "v1-to-v2" || goto fail
node tools\stream_moe_convert.js -m "%DIR%\v1_base.gguf" -o "%DIR%\t2_chunk" --format v2chunk --chunks %CH% --ratio %RATIO% || goto fail
call :cmpdir "%DIR%\chunk_base" "%DIR%\t2_chunk" "v1-to-v2chunk" || goto fail

echo [step 3/4] v2 source -^> v1/v2/v2chunk
node tools\stream_moe_convert.js -m "%DIR%\v2_base.gguf" -o "%DIR%\t3_v1.gguf" --format v1 || goto fail
call :cmp "%DIR%\v1_base.gguf" "%DIR%\t3_v1.gguf" "v2-to-v1" || goto fail
node tools\stream_moe_convert.js -m "%DIR%\v2_base.gguf" -o "%DIR%\t3_chunk" --format v2chunk --chunks %CH% --ratio %RATIO% || goto fail
call :cmpdir "%DIR%\chunk_base" "%DIR%\t3_chunk" "v2-to-v2chunk" || goto fail

echo [step 4/4] v2chunk source -^> v1/v2/v2chunk
node tools\stream_moe_convert.js -m "%DIR%\chunk_base\c1.gguf;%DIR%\chunk_base\c2.gguf;%DIR%\chunk_base\c3.gguf;%DIR%\chunk_base\c4.gguf;%DIR%\chunk_base\c5.gguf" -o "%DIR%\t4_v2.gguf" --format v2 || goto fail
node tools\stream_moe_convert.js -m "%DIR%\t4_v2.gguf" -o "%DIR%\t4_v1.gguf" --format v1 || goto fail
call :cmp "%DIR%\v1_base.gguf" "%DIR%\t4_v1.gguf" "v2chunk-to-v1" || goto fail
node tools\stream_moe_convert.js -m "%DIR%\t4_v2.gguf" -o "%DIR%\t4_chunk" --format v2chunk --chunks %CH% --ratio %RATIO% || goto fail
call :cmpdir "%DIR%\chunk_base" "%DIR%\t4_chunk" "v2chunk-to-v2chunk" || goto fail
call :cmp "%DIR%\v2_base.gguf" "%DIR%\t4_v2.gguf" "v2chunk-to-v2" || goto fail

echo.
echo [PASS] all matrix conversions byte-identical
popd & exit /b 0

:cmp
rem %1 = expected file, %2 = candidate file, %3 label; deletes candidate
node tools\cmp_gguf.js "%~1" "%~2"
if errorlevel 1 ( echo   [FAIL] %~3 : %~1  vs  %~2 & exit /b 1 )
echo   [OK] %~3
del /q "%~2" 2>nul
exit /b 0

:cmpdir
rem %1 = baseline dir, %2 = candidate dir, %3 label; deletes candidate dir
for %%f in ("%~2\*.gguf") do call :cmp "%~1\%%~nxf" "%%f" "%~3"
if errorlevel 1 ( exit /b 1 )
rd /s /q "%~2" 2>nul
exit /b 0

:fail
echo.
echo [FAIL] matrix verification stopped
popd & exit /b 1

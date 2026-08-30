@echo off
rem ===========================================================================
rem Converter matrix verification (v2-centric block abstraction, docs §9).
rem original -> v1/v2/v2chunk baseline, then each of v1/v2/v2chunk as source
rem produces all three formats and is byte-compared to the baseline. Stops on
rem the first mismatch, printing which two files differ.
rem
rem usage: verify_convert_matrix.bat <workdir>   (workdir needs ~3x model free)
rem   e.g. on a ramdisk: verify_convert_matrix.bat R:\conv
rem
rem Requires: temp\sm_env.bat (SM_GEMMA_ORIG), temp\stream_moe_convertd.exe
rem (compile line in docs/STREAMMOE_GGUF_FORMAT.md §7).
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

echo [step 1/4] original -^> v1/v2/v2chunk baseline
node tools\stream_moe_convert.js -m "%ORIG%" -o "%DIR%\v1_base.gguf" --format v1 || goto fail
node tools\stream_moe_convert.js -m "%ORIG%" -o "%DIR%\v2_base.gguf" --format v2 || goto fail
call :chunk "%DIR%\v2_base.gguf" "%DIR%\chunk_base" || goto fail

echo [step 2/4] v1 source -^> v1/v2/v2chunk
node tools\stream_moe_convert.js -m "%DIR%\v1_base.gguf" -o "%DIR%\t2_v1.gguf" --format v1 || goto fail
call :cmp "%DIR%\v1_base.gguf" "%DIR%\t2_v1.gguf" "v1->v1" || goto fail
node tools\stream_moe_convert.js -m "%DIR%\v1_base.gguf" -o "%DIR%\t2_v2.gguf" --format v2 || goto fail
call :cmp "%DIR%\v2_base.gguf" "%DIR%\t2_v2.gguf" "v1->v2" || goto fail
call :chunk "%DIR%\t2_v2.gguf" "%DIR%\t2_chunk" || goto fail
call :cmpdir "%DIR%\chunk_base" "%DIR%\t2_chunk" "v1->v2chunk" || goto fail

echo [step 3/4] v2 source -^> v1/v2/v2chunk
node tools\stream_moe_convert.js -m "%DIR%\v2_base.gguf" -o "%DIR%\t3_v1.gguf" --format v1 || goto fail
call :cmp "%DIR%\v1_base.gguf" "%DIR%\t3_v1.gguf" "v2->v1" || goto fail
call :chunk "%DIR%\v2_base.gguf" "%DIR%\t3_chunk" || goto fail
call :cmpdir "%DIR%\chunk_base" "%DIR%\t3_chunk" "v2->v2chunk" || goto fail

echo [step 4/4] v2chunk source -^> v1/v2/v2chunk
call :merge "%DIR%\chunk_base" "%DIR%\t4_v2.gguf" || goto fail
call :cmp "%DIR%\v2_base.gguf" "%DIR%\t4_v2.gguf" "v2chunk->v2" || goto fail
node tools\stream_moe_convert.js -m "%DIR%\t4_v2.gguf" -o "%DIR%\t4_v1.gguf" --format v1 || goto fail
call :cmp "%DIR%\v1_base.gguf" "%DIR%\t4_v1.gguf" "v2chunk->v1" || goto fail
call :chunk "%DIR%\t4_v2.gguf" "%DIR%\t4_chunk" || goto fail
call :cmpdir "%DIR%\chunk_base" "%DIR%\t4_chunk" "v2chunk->v2chunk" || goto fail

echo.
echo [PASS] all matrix conversions byte-identical
popd & exit /b 0

:chunk
rem %1 = v2 file, %2 = out dir (5 strips, ratio %RATIO%)
mkdir "%~2" 2>nul
node tools\convertd_call.js chunk "in=%~1" "out=%~2\c1.gguf,%~2\c2.gguf,%~2\c3.gguf,%~2\c4.gguf,%~2\c5.gguf" "ratio=%RATIO%" >nul
if errorlevel 1 ( echo [verify] chunk failed: %~1 & exit /b 1 )
exit /b 0

:merge
rem %1 = chunk dir, %2 = out v2
node tools\convertd_call.js merge "in=%~1\c1.gguf;%~1\c2.gguf;%~1\c3.gguf;%~1\c4.gguf;%~1\c5.gguf" "out=%~2" >nul
if errorlevel 1 ( echo [verify] merge failed: %~1 & exit /b 1 )
exit /b 0

:cmp
rem %1 %2 files, %3 label
fc /b "%~1" "%~2" >nul 2>&1
if errorlevel 1 ( echo [DIFF] %~3 : %~1  vs  %~2 & exit /b 1 )
echo   [OK] %~3
exit /b 0

:cmpdir
rem %1 baseline dir, %2 candidate dir, %3 label
for %%f in ("%~2\*.gguf") do call :cmp "%~1\%%~nxf" "%%f" "%~3"
exit /b 0

:fail
echo.
echo [FAIL] matrix verification stopped
popd & exit /b 1

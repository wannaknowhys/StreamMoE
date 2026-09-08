@echo off
rem ===========================================================================
rem Converter matrix verification (C++ converter; v1 removed).
rem Baselines: original -to- v2 / v3 / v3chunk. Then the format round-trips that
rem must be byte-identical (v3 canonically reorders dense by category, so
rem v3-to-v2 is NOT byte-identical to original-to-v2; the meaningful invariants
rem are v2-to-v3 == v3, v3-to-v2-to-v3 == v3, v3chunk-to-v3 == v3, etc).
rem
rem usage: verify_convert_matrix.bat ^<workdir^> [model.gguf]
rem   ^<workdir^> needs ~10x model free (baselines + round-trips)
rem   model defaults to SM_GEMMA_ORIG from temp\sm_env.bat
rem
rem Requires: build.bat convert main (build\main\bin\stream_moe_convert.exe)
rem ===========================================================================
setlocal
set DIR=%~1
if "%DIR%"=="" ( echo usage: verify_convert_matrix.bat ^<workdir^> [model.gguf] & exit /b 1 )
mkdir "%DIR%" 2>nul
if not exist "%DIR%" ( echo cannot create %DIR% & exit /b 1 )
pushd "%~dp0.."
call temp\sm_env.bat 2>nul
set CONV=build\main\bin\stream_moe_convert.exe
if not exist "%CONV%" ( echo [verify] build the converter first: build.bat convert main & exit /b 1 )
set ORIG=%~2
if "%ORIG%"=="" set ORIG=%SM_GEMMA_ORIG%
if not exist "%ORIG%" ( echo [verify] model not found: %ORIG% & exit /b 1 )
set RATIO=8:9:9:7:9
set CH=5

echo [step 1/4] baselines (original to v2 / v3 / v3chunk)
"%CONV%" -m "%ORIG%" -o "%DIR%\v2_base.gguf" --format v2 || goto fail
"%CONV%" -m "%ORIG%" -o "%DIR%\v3_base.gguf" --format v3 || goto fail
"%CONV%" -m "%ORIG%" -o "%DIR%\v3c_base" --format v3chunk --chunks %CH% --ratio %RATIO% || goto fail

echo [step 2/4] v2 source
"%CONV%" -m "%DIR%\v2_base.gguf" -o "%DIR%\t_v2.gguf" --format v2 || goto fail
call :cmp "%DIR%\v2_base.gguf" "%DIR%\t_v2.gguf" "v2-to-v2" || goto fail
"%CONV%" -m "%DIR%\v2_base.gguf" -o "%DIR%\t_v3.gguf" --format v3 || goto fail
call :cmp "%DIR%\v3_base.gguf" "%DIR%\t_v3.gguf" "v2-to-v3" || goto fail

echo [step 3/4] v3 source
"%CONV%" -m "%DIR%\v3_base.gguf" -o "%DIR%\t2_v3.gguf" --format v3 || goto fail
call :cmp "%DIR%\v3_base.gguf" "%DIR%\t2_v3.gguf" "v3-to-v3" || goto fail
"%CONV%" -m "%DIR%\v3_base.gguf" -o "%DIR%\t2_v2.gguf" --format v2 || goto fail
"%CONV%" -m "%DIR%\t2_v2.gguf" -o "%DIR%\t2_v3b.gguf" --format v3 || goto fail
call :cmp "%DIR%\v3_base.gguf" "%DIR%\t2_v3b.gguf" "v3-to-v2-to-v3" || goto fail

echo [step 4/4] v3chunk source
set CHUNKS=%DIR%\v3c_base\c1.gguf;%DIR%\v3c_base\c2.gguf;%DIR%\v3c_base\c3.gguf;%DIR%\v3c_base\c4.gguf;%DIR%\v3c_base\c5.gguf
"%CONV%" -m "%CHUNKS%" -o "%DIR%\t3_v3.gguf" --format v3 || goto fail
call :cmp "%DIR%\v3_base.gguf" "%DIR%\t3_v3.gguf" "v3chunk-to-v3" || goto fail
"%CONV%" -m "%CHUNKS%" -o "%DIR%\t3_v2.gguf" --format v2 || goto fail
"%CONV%" -m "%DIR%\t3_v2.gguf" -o "%DIR%\t3_v3b.gguf" --format v3 || goto fail
call :cmp "%DIR%\v3_base.gguf" "%DIR%\t3_v3b.gguf" "v3chunk-to-v2-to-v3" || goto fail

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

:fail
echo.
echo [FAIL] matrix verification stopped
popd & exit /b 1

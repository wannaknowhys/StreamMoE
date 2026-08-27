@echo off
setlocal enabledelayedexpansion
rem =====================================================================
rem  StreamMoE Build Utility (thin dispatcher)
rem  Build rules live in CMakeLists.txt; this file only forwards to
rem  cmake + ninja (Windows). On POSIX use `make` (Makefile dispatches the
rem  same cmake rules).
rem  Usage: build.bat [build|llamalibs|test|clean] [tag]
rem    tag   = build flavor sub-path under build\, default = main
rem  Artifacts per tag: bin\ (exe + libomp.dll), llama-build\ (vendored libllama),
rem                      cmake\ (cmake cache / intermediates).
rem =====================================================================

rem cmake/ninja/clang paths use FORWARD slashes - backslash \D etc. is an
rem invalid escape in CMake string parsing (vendored llamalibs already did this).
set CMAKE=F:/Dev/cmake/bin/cmake.exe
set NINJA=F:/Dev/cmake/bin/ninja.exe
set CLANG=F:/Dev/LLVM/bin/clang.exe
set CLANGXX=F:/Dev/LLVM/bin/clang++.exe
set RC=F:/Dev/LLVM/bin/llvm-rc.exe
set LIBOMP=F:/Dev/LLVM/lib/libomp.lib

set CMD=%1
if "%CMD%"=="" set CMD=build
set TAG=%2
if "%TAG%"=="" set TAG=main
set OUT=build\%TAG%
set LLAMA_BUILD=%OUT%\llama-build

if "%CMD%"=="llamalibs" goto llamalibs
if "%CMD%"=="build" goto build
if "%CMD%"=="test" goto test
if "%CMD%"=="clean" goto clean
echo Unknown command: %CMD%
goto help

:llamalibs
echo [StreamMoE] Building vendored libllama static libs into %LLAMA_BUILD% ...
if not exist "%LLAMA_BUILD%" mkdir "%LLAMA_BUILD%"
"%CMAKE%" -S third_party/llama.cpp -B "%LLAMA_BUILD%" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=%CLANG% ^
    -DCMAKE_CXX_COMPILER=%CLANGXX% ^
    -DCMAKE_RC_COMPILER=%RC% ^
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF ^
    -DLLAMA_CURL=OFF -DGGML_OPENMP=ON -DGGML_NATIVE=ON ^
    -DOpenMP_C_FLAGS=-fopenmp -DOpenMP_CXX_FLAGS=-fopenmp ^
    -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp ^
    -DOpenMP_libomp_LIBRARY=%LIBOMP%
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
"%NINJA%" -C "%LLAMA_BUILD%" llama llama-common-base
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [+] llamalibs done for tag %TAG%
exit /b 0

:build
if not exist "%LLAMA_BUILD%\src\llama.lib" (
    echo [-] libllama libs missing for tag %TAG%. Run first: build.bat llamalibs %TAG%
    exit /b 1
)
echo [StreamMoE] Configuring StreamMoE (cmake+ninja, tag %TAG%) ...
"%CMAKE%" -S . -B "%OUT%\cmake" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=%CLANG% ^
    -DCMAKE_CXX_COMPILER=%CLANGXX% ^
    -DCMAKE_RC_COMPILER=%RC% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    -DLLAMA_BUILD_DIR="%CD:\=/%/%LLAMA_BUILD:\=/%" ^
    -DSTREAMMOE_LIBOMP=%LIBOMP%
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [StreamMoE] Building stream_moe + stream_moe_server ...
"%NINJA%" -C "%OUT%\cmake" stream_moe stream_moe_server
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
copy /Y "%LIBOMP:.lib=.dll%" "%OUT%\bin\libomp.dll" >nul
echo [+] Build SUCCESS: %OUT%\bin\stream_moe.exe, %OUT%\bin\stream_moe_server.exe
exit /b 0

:test
if not exist "%LLAMA_BUILD%\src\llama.lib" (
    echo [-] libllama libs missing for tag %TAG%. Run first: build.bat llamalibs %TAG%
    exit /b 1
)
echo [StreamMoE] Configuring + building tests (tag %TAG%) ...
"%CMAKE%" -S . -B "%OUT%\cmake" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=%CLANG% ^
    -DCMAKE_CXX_COMPILER=%CLANGXX% ^
    -DCMAKE_RC_COMPILER=%RC% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DLLAMA_BUILD_DIR="%CD%\%LLAMA_BUILD%" ^
    -DSTREAMMOE_LIBOMP=%LIBOMP%
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [StreamMoE] Building unit-test executables ...
"%NINJA%" -C "%OUT%\cmake" test_async_dio test_moe_loader test_profiler test_scheduler test_slot
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
copy /Y "%LIBOMP:.lib=.dll%" "%OUT%\bin\libomp.dll" >nul
echo [StreamMoE] Running ctest ...
"%NINJA%" -C "%OUT%\cmake" test
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [+] All unit tests passed for tag %TAG%!
exit /b 0

:clean
echo [StreamMoE] Removing build\ (all tags)...
if exist build rmdir /s /q build
echo [+] Clean complete.
exit /b 0

:help
echo StreamMoE Build Utility (thin dispatcher over cmake+ninja; rules in CMakeLists.txt)
echo Usage:
echo   build.bat build      - Build stream_moe.exe and stream_moe_server.exe
echo   build.bat llamalibs  - Configure and build vendored libllama static libs only
echo   build.bat test       - Build and run all unit tests
echo   build.bat clean      - Remove build\ (all tags)
echo Optional [tag] sub-path for each command (default: main).
echo   build.bat build memwatch   - build under build\memwatch\bin
echo See docs/PROJECT_STRUCTURE.md for the build layout pattern.
exit /b 0

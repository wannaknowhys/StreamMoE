@echo off
setlocal enabledelayedexpansion
rem =====================================================================
rem  StreamMoE Build Utility
rem  Usage: build.bat [build|llamalibs|test|clean] [tag]
rem    tag   = build flavor sub-path under build\, default = main
rem            (e.g. build\main\bin\stream_moe.exe, build\memwatch\bin\...)
rem  Artifacts per tag: bin\ (exe + libomp.dll), obj\ (intermediates),
rem                      llama-build\ (vendored libllama cmake dir)
rem =====================================================================

set CLANG_CXX=F:\Dev\LLVM\bin\clang++.exe
set CLANG_CC=F:\Dev\LLVM\bin\clang.exe
set CXXFLAGS=-std=c++17 -O3 -fopenmp -DUNICODE -D_UNICODE -D_CRT_SECURE_NO_WARNINGS -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -I third_party/llama.cpp/include -Wall -Wextra -Wno-unused-parameter
set LIBOMP=F:\Dev\LLVM\lib\libomp.lib

set CMD=%1
if "%CMD%"=="" set CMD=build
set TAG=%2
if "%TAG%"=="" set TAG=main
set OUT=build\%TAG%
set LLAMA_BUILD=%OUT%\llama-build
set BIN=%OUT%\bin
set OBJ=%OUT%\obj
set LLAMA_LIBS=%LLAMA_BUILD%\src\llama.lib %LLAMA_BUILD%\ggml\src\ggml.lib %LLAMA_BUILD%\ggml\src\ggml-base.lib %LLAMA_BUILD%\ggml\src\ggml-cpu.lib
set LLAMA_INC=-I third_party/llama.cpp/include -I third_party/llama.cpp/vendor

if "%CMD%"=="llamalibs" goto llamalibs
if "%CMD%"=="build" goto build
if "%CMD%"=="test" goto test
if "%CMD%"=="clean" goto clean
echo Unknown command: %CMD%
goto help

:llamalibs
echo [StreamMoE] Building vendored libllama static libs into %LLAMA_BUILD% ...
if not exist "%LLAMA_BUILD%" mkdir "%LLAMA_BUILD%"
"F:\Dev\cmake\bin\cmake.exe" -S third_party/llama.cpp -B "%LLAMA_BUILD%" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=F:/Dev/cmake/bin/ninja.exe ^
    -DCMAKE_C_COMPILER=F:/Dev/LLVM/bin/clang.exe ^
    -DCMAKE_CXX_COMPILER=F:/Dev/LLVM/bin/clang++.exe ^
    -DCMAKE_RC_COMPILER=F:/Dev/LLVM/bin/llvm-rc.exe ^
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF ^
    -DLLAMA_CURL=OFF -DGGML_OPENMP=ON -DGGML_NATIVE=ON ^
    -DOpenMP_C_FLAGS=-fopenmp -DOpenMP_CXX_FLAGS=-fopenmp ^
    -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp ^
    -DOpenMP_libomp_LIBRARY=F:/Dev/LLVM/lib/libomp.lib
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
"F:\Dev\cmake\bin\ninja.exe" -C "%LLAMA_BUILD%" llama llama-common-base
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [+] llamalibs done for tag %TAG%
exit /b 0

:build
echo [StreamMoE] Building StreamMoE binaries for tag [%TAG%] ...
if not exist "%BIN%" mkdir "%BIN%"
if not exist "%OBJ%" mkdir "%OBJ%"

if not exist "%LLAMA_BUILD%\src\llama.lib" (
    echo [-] libllama libs missing for tag %TAG%. Run first: build.bat llamalibs %TAG%
    exit /b 1
)

echo [StreamMoE] Linking %BIN%\stream_moe.exe (real inference core)...
"%CLANG_CXX%" %CXXFLAGS% %LLAMA_INC% ^
    src/backend/moe_backend.cpp ^
    src/backend/scheduler.cpp ^
    src/io/async_dio_win.cpp ^
    src/io/staging_reader.cpp ^
    src/pool/expert_stats.cpp ^
    src/engine/llama_engine.cpp ^
    src/main.cpp ^
    src/profile/profiler.cpp ^
    %LLAMA_LIBS% %LIBOMP% -ladvapi32 -lsynchronization ^
    -o "%BIN%\stream_moe.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Linking %BIN%\stream_moe_server.exe (real inference core)...
"%CLANG_CXX%" %CXXFLAGS% %LLAMA_INC% ^
    src/backend/moe_backend.cpp ^
    src/backend/scheduler.cpp ^
    src/io/async_dio_win.cpp ^
    src/pool/expert_stats.cpp ^
    src/engine/llama_engine.cpp ^
    src/server/http_server.cpp ^
    src/loader/moe_loader.cpp ^
    src/io/staging_reader.cpp ^
    src/profile/profiler.cpp ^
    src/server_main.cpp ^
    %LLAMA_LIBS% %LIBOMP% ^
    -lws2_32 -ladvapi32 -lsynchronization ^
    -o "%BIN%\stream_moe_server.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

copy /Y "F:\Dev\LLVM\bin\libomp.dll" "%BIN%\libomp.dll" >nul
echo [+] Build SUCCESS: %BIN%\stream_moe.exe, %BIN%\stream_moe_server.exe
exit /b 0

:test
echo [StreamMoE] Building Unit Tests for tag [%TAG%] ...
if not exist "%OBJ%" mkdir "%OBJ%"
if not exist temp mkdir temp

if not exist "%LLAMA_BUILD%\src\llama.lib" (
    echo [-] libllama libs missing for tag %TAG%. Run first: build.bat llamalibs %TAG%
    exit /b 1
)

echo [StreamMoE] Compiling test_async_dio...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/io/async_dio_win.cpp ^
    src/io/staging_reader.cpp ^
    tests/test_async_dio.cpp ^
    -o "%OBJ%\test_async_dio.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_moe_loader...
"%CLANG_CXX%" %CXXFLAGS% %LLAMA_INC% ^
    %LLAMA_LIBS% %LIBOMP% -ladvapi32 ^
    src/io/staging_reader.cpp ^
    src/loader/moe_loader.cpp ^
    tests/test_moe_loader.cpp ^
    -o "%OBJ%\test_moe_loader.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_profiler...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/profile/profiler.cpp ^
    tests/test_profiler.cpp ^
    -o "%OBJ%\test_profiler.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_scheduler (route B pool + DIO)...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/backend/scheduler.cpp ^
    src/io/staging_reader.cpp ^
    src/io/async_dio_win.cpp ^
    src/pool/expert_stats.cpp ^
    tests/test_scheduler.cpp ^
    -lsynchronization ^
    -o "%OBJ%\test_scheduler.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_slot (route B control plane)...
"%CLANG_CXX%" %CXXFLAGS% ^
    tests/test_slot.cpp ^
    -lsynchronization ^
    -o "%OBJ%\test_slot.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

rem OpenMP runtime next to test exes (test_moe_loader links ggml-cpu)
copy /Y "F:\Dev\LLVM\bin\libomp.dll" "%OBJ%\libomp.dll" >nul

echo.
echo ========================================================
echo [StreamMoE] Executing Async Direct I/O Tests
echo ========================================================
"%OBJ%\test_async_dio.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing GGUF MoE Loader Tests
echo ========================================================
"%OBJ%\test_moe_loader.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing Profiler Tests
echo ========================================================
"%OBJ%\test_profiler.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing Scheduler Tests (route B pool + DIO)
echo ========================================================
"%OBJ%\test_scheduler.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing Slot Control Plane Tests (route B)
echo ========================================================
"%OBJ%\test_slot.exe"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [+] All unit tests passed for tag %TAG%!
exit /b 0

:clean
echo [StreamMoE] Removing build\ (all tags)...
if exist build rmdir /s /q build
echo [+] Clean complete.
exit /b 0

:help
echo StreamMoE Build Utility
echo Usage:
echo   build.bat build      - Build stream_moe.exe and stream_moe_server.exe
echo   build.bat llamalibs  - Configure and build vendored libllama static libs only
echo   build.bat test       - Build and run all unit tests
echo   build.bat clean      - Remove build\ (all tags)
echo Optional [tag] sub-path for each command (default: main).
echo   build.bat build memwatch   - build under build\memwatch\bin
echo See docs/PROJECT_STRUCTURE.md for the build layout pattern.
exit /b 0

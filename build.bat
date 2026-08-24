@echo off
setlocal enabledelayedexpansion

set CLANG_CXX=F:\Dev\LLVM\bin\clang++.exe
set CXXFLAGS=-std=c++17 -O3 -fopenmp -DUNICODE -D_UNICODE -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/include -Wall -Wextra -Wno-unused-parameter

if "%1"=="" goto help
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="help" goto help

:test
echo [StreamMoE] Building Phase 1 and Phase 2 Unit Tests...
if not exist temp mkdir temp
if not exist build mkdir build

echo [StreamMoE] Compiling test_async_dio...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/io/async_dio_win.cpp ^
    src/io/staging_reader.cpp ^
    tests/test_async_dio.cpp ^
    -o temp/test_async_dio.exe

if %ERRORLEVEL% NEQ 0 (
    echo [-] test_async_dio build failed!
    exit /b %ERRORLEVEL%
)

echo [StreamMoE] Compiling test_expert_pool...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/pool/expert_stats.cpp ^
    src/pool/expert_pool.cpp ^
    tests/test_expert_pool.cpp ^
    -o temp/test_expert_pool.exe

if %ERRORLEVEL% NEQ 0 (
    echo [-] test_expert_pool build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo [StreamMoE] Executing Phase 1: Async Direct I/O Tests
echo ========================================================
temp\test_async_dio.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing Phase 2: Pinned Pool and Eviction Tests
echo ========================================================
temp\test_expert_pool.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [+] All Phase 1 and Phase 2 tests passed!
exit /b 0

:clean
echo [StreamMoE] Cleaning build and temp directories...
if exist temp rmdir /s /q temp
if exist build rmdir /s /q build
echo [+] Clean complete.
exit /b 0

:help
echo StreamMoE Build Utility
echo Usage:
echo   build.bat test   - Build and run all unit tests
echo   build.bat clean  - Clean temporary and build directories
echo   build.bat help   - Show this help message
exit /b 0
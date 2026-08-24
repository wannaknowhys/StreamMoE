@echo off
setlocal enabledelayedexpansion

set CLANG_CXX=F:\Dev\LLVM\bin\clang++.exe
set CLANG_CC=F:\Dev\LLVM\bin\clang.exe
set CXXFLAGS=-std=c++17 -O3 -fopenmp -DUNICODE -D_UNICODE -D_CRT_SECURE_NO_WARNINGS -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -I third_party/llama.cpp/include -Wall -Wextra -Wno-unused-parameter

if "%1"=="" goto help
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="help" goto help

:test
echo [StreamMoE] Building Phase 1, Phase 2 and Phase 3 Unit Tests...
if not exist temp mkdir temp
if not exist build mkdir build

echo [StreamMoE] Compiling GGML support objects...
"%CLANG_CC%" -D_CRT_SECURE_NO_WARNINGS -DGGML_VERSION="\"1.0.0\"" -DGGML_COMMIT="\"dev\"" -c third_party/llama.cpp/ggml/src/ggml.c -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -o temp/ggml.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CC%" -D_CRT_SECURE_NO_WARNINGS -c third_party/llama.cpp/ggml/src/ggml-quants.c -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -o temp/ggml-quants.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CXX%" %CXXFLAGS% -c third_party/llama.cpp/ggml/src/ggml-threading.cpp -o temp/ggml-threading.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CXX%" %CXXFLAGS% -c third_party/llama.cpp/ggml/src/ggml-backend.cpp -o temp/ggml-backend.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CC%" -D_CRT_SECURE_NO_WARNINGS -c third_party/llama.cpp/ggml/src/ggml-alloc.c -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -o temp/ggml-alloc.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CXX%" %CXXFLAGS% -c third_party/llama.cpp/ggml/src/ggml-backend-meta.cpp -o temp/ggml-backend-meta.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CXX%" %CXXFLAGS% -c third_party/llama.cpp/ggml/src/ggml-backend-reg.cpp -o temp/ggml-backend-reg.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

"%CLANG_CXX%" %CXXFLAGS% -c third_party/llama.cpp/ggml/src/ggml-backend-dl.cpp -o temp/ggml-backend-dl.o
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

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

echo [StreamMoE] Compiling test_moe_loader...
"%CLANG_CXX%" %CXXFLAGS% ^
    temp/ggml.o temp/ggml-quants.o temp/ggml-threading.o temp/ggml-backend.o temp/ggml-alloc.o temp/ggml-backend-meta.o temp/ggml-backend-reg.o temp/ggml-backend-dl.o ^
    third_party/llama.cpp/ggml/src/gguf.cpp ^
    src/io/staging_reader.cpp ^
    src/loader/moe_loader.cpp ^
    tests/test_moe_loader.cpp ^
    -o temp/test_moe_loader.exe

if %ERRORLEVEL% NEQ 0 (
    echo [-] test_moe_loader build failed!
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
echo ========================================================
echo [StreamMoE] Executing Phase 3: GGUF MoE Topology and Homogeneity Tests
echo ========================================================
temp\test_moe_loader.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [+] All Phase 1, Phase 2, and Phase 3 tests passed!
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
@echo off
setlocal enabledelayedexpansion

set CLANG_CXX=F:\Dev\LLVM\bin\clang++.exe
set CLANG_CC=F:\Dev\LLVM\bin\clang.exe
set CXXFLAGS=-std=c++17 -O3 -fopenmp -DUNICODE -D_UNICODE -D_CRT_SECURE_NO_WARNINGS -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -I third_party/llama.cpp/include -Wall -Wextra -Wno-unused-parameter
set LIBOMP=F:\Dev\LLVM\lib\libomp.lib

if "%1"=="" goto build
if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="llamalibs" goto llamalibs
if "%1"=="help" goto help

:llamalibs
echo [StreamMoE] Building vendored libllama static libs (CPU backend, OpenMP)...
if not exist temp mkdir temp
& F:\Dev\cmake\bin\cmake.exe -S third_party/llama.cpp -B temp/llama-build -G Ninja ^
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
& F:\Dev\cmake\bin\ninja.exe -C temp/llama-build llama llama-common-base
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
exit /b 0

:build
echo [StreamMoE] Building StreamMoE Binaries (stream_moe.exe, stream_moe_server.exe)...
if not exist temp mkdir temp
if not exist bin mkdir bin

if not exist temp\llama-build\src\llama.lib (
    echo [-] libllama libs missing, building them first...
    call %~f0 llamalibs
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
)

set LLAMA_LIBS=temp\llama-build\src\llama.lib temp\llama-build\ggml\src\ggml.lib temp\llama-build\ggml\src\ggml-base.lib temp\llama-build\ggml\src\ggml-cpu.lib
set LLAMA_INC=-I third_party/llama.cpp/include -I third_party/llama.cpp/vendor

echo [StreamMoE] Linking bin/stream_moe.exe (real inference core)...
"%CLANG_CXX%" %CXXFLAGS% %LLAMA_INC% ^
    src/engine/llama_engine.cpp ^
    src/main.cpp ^
    src/profile/profiler.cpp ^
    %LLAMA_LIBS% %LIBOMP% -ladvapi32 ^
    -o bin/stream_moe.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Linking bin/stream_moe_server.exe (real inference core)...
"%CLANG_CXX%" %CXXFLAGS% %LLAMA_INC% ^
    src/engine/llama_engine.cpp ^
    src/server/http_server.cpp ^
    src/loader/moe_loader.cpp ^
    src/io/staging_reader.cpp ^
    src/profile/profiler.cpp ^
    src/server_main.cpp ^
    third_party/llama.cpp/ggml/src/gguf.cpp ^
    %LLAMA_LIBS% %LIBOMP% ^
    -lws2_32 -ladvapi32 ^
    -o bin/stream_moe_server.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

copy /Y "F:\Dev\LLVM\bin\libomp.dll" "bin\libomp.dll" >nul

echo [+] Build SUCCESS: bin\stream_moe.exe, bin\stream_moe_server.exe
exit /b 0

:test
echo [StreamMoE] Building Unit Tests...
if not exist temp mkdir temp
if not exist bin mkdir bin

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

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_expert_pool...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/pool/expert_stats.cpp ^
    src/pool/expert_pool.cpp ^
    tests/test_expert_pool.cpp ^
    -o temp/test_expert_pool.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_moe_loader...
"%CLANG_CXX%" %CXXFLAGS% ^
    temp/ggml.o temp/ggml-quants.o temp/ggml-threading.o temp/ggml-backend.o temp/ggml-alloc.o temp/ggml-backend-meta.o temp/ggml-backend-reg.o temp/ggml-backend-dl.o ^
    third_party/llama.cpp/ggml/src/gguf.cpp ^
    src/io/staging_reader.cpp ^
    src/loader/moe_loader.cpp ^
    tests/test_moe_loader.cpp ^
    -o temp/test_moe_loader.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_scheduler...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/io/async_dio_win.cpp ^
    src/io/staging_reader.cpp ^
    src/pool/expert_stats.cpp ^
    src/pool/expert_pool.cpp ^
    src/scheduler/moe_scheduler.cpp ^
    src/engine/subgraph_executor.cpp ^
    tests/test_scheduler.cpp ^
    -o temp/test_scheduler.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_state_machine...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/io/async_dio_win.cpp ^
    src/io/staging_reader.cpp ^
    src/pool/expert_stats.cpp ^
    src/pool/expert_pool.cpp ^
    src/scheduler/moe_scheduler.cpp ^
    src/engine/state_machine.cpp ^
    src/engine/speculative_engine.cpp ^
    tests/test_state_machine.cpp ^
    -o temp/test_state_machine.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_kv_cache...
"%CLANG_CXX%" %CXXFLAGS% ^
    temp/ggml.o temp/ggml-quants.o temp/ggml-threading.o temp/ggml-backend.o temp/ggml-alloc.o temp/ggml-backend-meta.o temp/ggml-backend-reg.o temp/ggml-backend-dl.o ^
    third_party/llama.cpp/ggml/src/gguf.cpp ^
    src/io/staging_reader.cpp ^
    src/loader/moe_loader.cpp ^
    src/tokenizer/tokenizer.cpp ^
    src/kv/kv_cache_manager.cpp ^
    src/profile/profiler.cpp ^
    tests/test_kv_cache.cpp ^
    -o temp/test_kv_cache.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [StreamMoE] Compiling test_profiler...
"%CLANG_CXX%" %CXXFLAGS% ^
    src/profile/profiler.cpp ^
    tests/test_profiler.cpp ^
    -o temp/test_profiler.exe

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

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
echo ========================================================
echo [StreamMoE] Executing Phase 4: Dual-Thread Pipeline and Pointer Rebind
echo ========================================================
temp\test_scheduler.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing Phase 5: State Machine and Speculative Decoding
echo ========================================================
temp\test_state_machine.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing KV Cache Persistence and Expansion Tests
echo ========================================================
temp\test_kv_cache.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ========================================================
echo [StreamMoE] Executing High-Resolution Profiler Tests
echo ========================================================
temp\test_profiler.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo.
echo ========================================================
echo [StreamMoE] Executing GGUF Tokenizer Tests
echo ========================================================
temp\test_tokenizer.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo [+] All Phase 1-5, KV Cache, and Profiler tests passed!
exit /b 0

:clean
echo [StreamMoE] Cleaning build and temp directories...
if exist temp rmdir /s /q temp
if exist bin rmdir /s /q bin
echo [+] Clean complete.
exit /b 0

:help
echo StreamMoE Build Utility
echo Usage:
echo   build.bat build      - Build stream_moe.exe and stream_moe_server.exe (libllama core)
echo   build.bat llamalibs  - Configure and build vendored libllama static libs only
echo   build.bat test       - Build and run all unit tests
echo   build.bat clean      - Clean temporary and build directories
echo   build.bat help       - Show this help message
exit /b 0
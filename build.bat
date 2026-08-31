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
set CLANG=F:/Dev/LLVM/bin/clang-cl.exe
set CLANGXX=F:/Dev/LLVM/bin/clang-cl.exe
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
if "%CMD%"=="convertd" goto convertd
echo Unknown command: %CMD%
goto help

:llamalibs
echo [StreamMoE] Building vendored libllama static libs into %LLAMA_BUILD% ...
if not exist "%LLAMA_BUILD%" mkdir "%LLAMA_BUILD%"
rem Device-backend switches are forwarded from the environment (empty = OFF).
rem Set them to ON to enable: GGML_CUDA GGML_HIP GGML_METAL GGML_SYCL GGML_VULKAN
rem (multiple can be ON at once - llama.cpp schedules layers across registered
rem backends via --split-mode/--tensor-split). LLAMA_BUILD_TOOLS controls the
rem upstream llama-cli/llama-server build (default ON).
if "%LLAMA_BUILD_TOOLS%"=="" set LLAMA_BUILD_TOOLS=ON
rem STREAM_MOE feature macros by tag (build.bat llamalibs <tag>):
rem   main            -> -DSTREAM_MOE_ROUTE_B          (route-B full inference, production)
rem   upstream_dump   -> -DSTREAM_MOE_PREFILL_EXPORT   (prefill export, standard upstream baseline)
rem   StreamMoE_dump  -> both                          (full StreamMoE export, vs upstream_dump)
set STREAM_MOE_MACROS=
if "%TAG%"=="main"           set STREAM_MOE_MACROS=-DSTREAM_MOE_ROUTE_B
if "%TAG%"=="upstream_dump"  set STREAM_MOE_MACROS=-DSTREAM_MOE_PREFILL_EXPORT
if "%TAG%"=="StreamMoE_dump" set STREAM_MOE_MACROS=-DSTREAM_MOE_ROUTE_B -DSTREAM_MOE_PREFILL_EXPORT
rem ---- CPU arch + backend by tag ----
rem   main: production route-B. TODO: switch to GGML_CPU_ALL_VARIANTS (official
rem         ggml-cpu-<arch> runtime dispatch, see F:/Dev/computer-use/llama); pinned
rem         to znver3 (5950X) for now.
rem   upstream_dump / StreamMoE_dump: dump tools, local - znver3 + vulkan backend.
set STREAM_MOE_CPU_FLAGS=-march=znver3
set GGML_VULKAN_DEFAULT=OFF
if "%TAG%"=="upstream_dump"  set GGML_VULKAN_DEFAULT=ON
if "%TAG%"=="StreamMoE_dump" set GGML_VULKAN_DEFAULT=ON
if "%GGML_VULKAN%"=="" set GGML_VULKAN=%GGML_VULKAN_DEFAULT%
if "%GGML_CUDA%"==""  set GGML_CUDA=OFF
if "%GGML_HIP%"==""   set GGML_HIP=OFF
if "%GGML_METAL%"=="" set GGML_METAL=OFF
if "%GGML_SYCL%"==""  set GGML_SYCL=OFF
rem vulkan-shaders-gen sub-cmake runs its own configure and inherits the
rem environment: make our ninja + clang visible via PATH. (The toolchain-file
rem hook is cross-compile-only, and the raw VULKAN_SHADER_GEN_CMAKE_ARGS -D is
rem cleared by ggml-vulkan's `set(VULKAN_SHADER_GEN_CMAKE_ARGS "")`.)
if "%GGML_VULKAN%"=="ON" (
    rem split dir-extraction from the PATH assignment: expanding %PATH% inside
    rem `for ... in (...)` breaks when PATH contains "(x86)" (cmd treats the `)`
    rem of (x86) as the end of the for list -> "此时不应有 \VMware\VMware").
    for %%I in ("%NINJA%") do set NINJA_DIR=%%~dpI
    for %%I in ("%CLANG%") do set CLANG_DIR=%%~dpI
    set path="!NINJA_DIR!;!CLANG_DIR!;!PATH!"
    set VULKAN_TOOLCHAIN_ARGS=
) else set VULKAN_TOOLCHAIN_ARGS=
"%CMAKE%" -S third_party/llama.cpp -B "%LLAMA_BUILD%" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=%CLANG% ^
    -DCMAKE_CXX_COMPILER=%CLANGXX% ^
    -DCMAKE_RC_COMPILER=%RC% ^
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=%LLAMA_BUILD_TOOLS% ^
    -DLLAMA_ALL_WARNINGS=OFF ^
    -DLLAMA_CURL=OFF -DGGML_OPENMP=ON -DGGML_NATIVE=ON ^
    -DGGML_VULKAN=%GGML_VULKAN% -DGGML_CUDA=%GGML_CUDA% -DGGML_HIP=%GGML_HIP% ^
    -DGGML_METAL=%GGML_METAL% -DGGML_SYCL=%GGML_SYCL% ^
    -DCMAKE_C_FLAGS="-Wno-cast-qual %STREAM_MOE_CPU_FLAGS%" -DCMAKE_CXX_FLAGS="-Wno-cast-qual /EHsc %STREAM_MOE_CPU_FLAGS% %STREAM_MOE_MACROS%" ^
    -DOpenMP_C_FLAGS=-Xclang;-fopenmp -DOpenMP_CXX_FLAGS=-Xclang;-fopenmp ^
    -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp ^
    -DOpenMP_libomp_LIBRARY=%LIBOMP% %VULKAN_TOOLCHAIN_ARGS%
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
"%NINJA%" -C "%LLAMA_BUILD%" llama llama-cli llama-server
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
copy /Y "%LIBOMP:.lib=.dll%" "%LLAMA_BUILD%\bin\libomp.dll" >nul
echo [+] llamalibs done for tag %TAG% (libllama + llama-cli + llama-server)
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

:convertd
echo [StreamMoE] Building convertd (dumb GGUF TCP service) -> build\convertd\convertd.exe ...
if not exist build\convertd mkdir build\convertd
"%CLANG%" /std:c++17 tools\stream_moe_convertd.cpp /EHsc /MT ^
    -I%CD:\=/%/third_party/llama.cpp/ggml/include ^
    -I%CD:\=/%/third_party/llama.cpp/ggml/src ^
    %CD:\=/%/build/main/llama-build/ggml/src/ggml-base.lib ^
    %LIBOMP% ws2_32.lib /Fe:build\convertd\convertd.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [+] convertd built: build\convertd\convertd.exe
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
echo Optional [tag] sub-path for llamalibs/build (default: main).
echo   llamalibs main           - route-B llama-server (build\main)
echo   llamalibs upstream_dump  - prefill-only export (build\upstream_dump)
echo   llamalibs StreamMoE_dump - route-B + prefill export (build\StreamMoE_dump)
echo   build.bat convertd       - build converter TCP service (build\convertd)
echo   build.bat build memwatch - build under build\memwatch\bin (legacy main-project build)
echo See docs/PROJECT_STRUCTURE.md for the build layout pattern.
exit /b 0

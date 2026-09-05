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
if "%CMD%"=="asan" goto asan
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
rem STREAM_MOE features by tag (build.bat llamalibs <tag>). Feature macros are
rem defined by the vendored root CMakeLists from -DSTREAM_MOE_FEATURES (whole
rem build), not passed via CMAKE_CXX_FLAGS.
rem   main                  -> route_b          (route-B full inference, production)
rem   StreamMoE             -> route_b + vulkan (flagship: route-B with the vram
rem                            device-pool path, NO prefill export - llama-cli
rem                            dialogue / serving; = production main + vulkan)
rem   upstream_dump         -> prefill_export   (prefill export, no vulkan baseline)
rem   upstream_vulkan_dump  -> prefill_export   (prefill export + vulkan, for Vulkan0 comparison)
rem   StreamMoE_dump        -> route_b,prefill_export + vulkan (full StreamMoE export,
rem                            with the Vulkan0 device-pool path; vs upstream_dump)
set STREAM_MOE_FEATURES=
if "%TAG%"=="main"                 set STREAM_MOE_FEATURES=route_b
if "%TAG%"=="StreamMoE"            set STREAM_MOE_FEATURES=route_b
if "%TAG%"=="upstream_dump"        set STREAM_MOE_FEATURES=prefill_export
if "%TAG%"=="upstream_vulkan_dump" set STREAM_MOE_FEATURES=prefill_export
if "%TAG%"=="StreamMoE_dump"       set STREAM_MOE_FEATURES=route_b,prefill_export
if "%TAG%"=="StreamMoE_dump_dbg"   set STREAM_MOE_FEATURES=route_b,prefill_export
rem ---- in-progress compile-time feature switches (default OFF) ------------
rem   StreamMoE_dump_dbg (tag) = StreamMoE_dump + STREAM_MOE_TEMP (temporary
rem   diagnostic code compiled in; see PROJECT_STRUCTURE.md §10). All other tags
rem   build without the macro (production green). Diagnostics are #ifdef
rem   STREAM_MOE_TEMP and must stay gated - the dbg tag is the only way they build.
set STREAM_MOE_TEMP_FLAG=
if "%TAG%"=="StreamMoE_dump_dbg"   set STREAM_MOE_TEMP_FLAG=-DSTREAM_MOE_TEMP
rem ---- CPU arch + backend by tag ----
rem   main: production route-B. TODO: switch to GGML_CPU_ALL_VARIANTS (official
rem         ggml-cpu-<arch> runtime dispatch, see F:/Dev/computer-use/llama); pinned
rem         to znver3 (5950X) for now.
rem   upstream_dump: prefill export, no vulkan (clean upstream baseline).
rem   upstream_vulkan_dump: vulkan backend for Vulkan0 comparison.
rem   StreamMoE_dump / StreamMoE_dump_dbg / StreamMoE: vulkan backend too - the
rem         route-B Vulkan0 device-pool path (M1+) needs the device registered;
rem         runs still default to CPU unless -ngl is given, so numerics land on
rem         the same CPU kernels. StreamMoE = the flagship tag: route_b only (no
rem         prefill export code), for llama-cli dialogue / serving / speed.
set STREAM_MOE_CPU_FLAGS=-march=znver3
set GGML_VULKAN_DEFAULT=OFF
if "%TAG%"=="upstream_vulkan_dump" set GGML_VULKAN_DEFAULT=ON
if "%TAG%"=="StreamMoE_dump"       set GGML_VULKAN_DEFAULT=ON
if "%TAG%"=="StreamMoE_dump_dbg"   set GGML_VULKAN_DEFAULT=ON
if "%TAG%"=="StreamMoE"            set GGML_VULKAN_DEFAULT=ON
rem env GGML_VULKAN=OFF still overrides (e.g. to rebuild a CPU-only StreamMoE_dump
rem for an IDENTICAL baseline regression - see baseline_regression/README.md).
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
    set PATH=!NINJA_DIR!;!CLANG_DIR!;!PATH!
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
    -DCMAKE_C_FLAGS="-Wno-cast-qual %STREAM_MOE_CPU_FLAGS%" -DCMAKE_CXX_FLAGS="-Wno-cast-qual /EHsc %STREAM_MOE_CPU_FLAGS% %STREAM_MOE_TEMP_FLAG%" ^
    -DSTREAM_MOE_FEATURES="%STREAM_MOE_FEATURES%" ^
    -DOpenMP_C_FLAGS=-Xclang;-fopenmp -DOpenMP_CXX_FLAGS=-Xclang;-fopenmp ^
    -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp ^
    -DOpenMP_libomp_LIBRARY=%LIBOMP% %VULKAN_TOOLCHAIN_ARGS%
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
"%NINJA%" -C "%LLAMA_BUILD%" llama llama-cli llama-server
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
rem OpenMP runtime DLL lives in the LLVM bin dir (lib/libomp.dll does not exist) -
rem copy it next to the binaries or llama-server dies with STATUS_DLL_NOT_FOUND.
for %%I in ("%CLANG%") do set LLVM_BIN_DIR=%%~dpI
copy /Y "%LLVM_BIN_DIR%libomp.dll" "%LLAMA_BUILD%\bin\libomp.dll" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [-] failed to copy libomp.dll from "%LLVM_BIN_DIR%"
    exit /b %ERRORLEVEL%
)
echo [+] llamalibs done for tag %TAG% (libllama + llama-cli + llama-server)
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
"%NINJA%" -C "%OUT%\cmake" test_async_dio test_moe_loader test_profiler test_scheduler test_slot test_mix_plan
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
copy /Y "%LIBOMP:.lib=.dll%" "%OUT%\bin\libomp.dll" >nul
echo [StreamMoE] Running ctest ...
"%NINJA%" -C "%OUT%\cmake" test
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [+] All unit tests passed for tag %TAG%!
exit /b 0

:asan
rem ASan (AddressSanitizer) llama-server with route-B, using MSVC cl.exe - see
rem docs/ASAN_BUILD.md. clang-cl ASan conflicts with CMake TryCompile /RTC1, so
rem this deliberately does NOT reuse the clang-cl toolchain above.
echo [StreamMoE] Building ASan llama-server (MSVC cl + /fsanitize=address) -> build\asan\llama-build ...
set "VSVARS=%VS_PATH%"
if "%VSVARS%"=="" set "VSVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSVARS%" (
    echo [-] vcvars64 not found: %VSVARS% ^(set VS_PATH to the vcvars64.bat location^)
    exit /b 1
)
call "%VSVARS%" >nul
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
if not exist build\asan\llama-build mkdir build\asan\llama-build
"%CMAKE%" -S third_party/llama.cpp -B build/asan/llama-build -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TRY_COMPILE_CONFIGURATION=Release ^
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_ALL_WARNINGS=OFF -DLLAMA_CURL=OFF ^
    -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF ^
    -DCMAKE_C_FLAGS="/W0 /fsanitize=address /Zi /O2" ^
    -DCMAKE_CXX_FLAGS="/W0 /EHsc /fsanitize=address /Zi /O2" ^
    -DSTREAM_MOE_FEATURES="route_b"
if errorlevel 1 exit /b 1
"%NINJA%" -C build/asan/llama-build llama-server
if errorlevel 1 exit /b 1
rem ASan runtime dll must match the MSVC version that just configured (vswhere /
rem VCINSTALLDIR is set by vcvars64); copy it next to the exe.
for /d %%m in ("%VCINSTALLDIR%\Tools\MSVC\*") do set MSVC_VER_DIR=%%m
copy /Y "%MSVC_VER_DIR%\bin\Hostx64\x64\clang_rt.asan_dynamic-x86_64.dll" build\asan\llama-build\bin\ >nul
if errorlevel 1 (
    echo [-] failed to copy clang_rt.asan_dynamic-x86_64.dll from "%MSVC_VER_DIR%"
    exit /b 1
)
echo [+] ASan llama-server built: build\asan\llama-build\bin\llama-server.exe
exit /b 0

:convertd
echo [StreamMoE] Building convertd (dumb GGUF TCP service) -> build\convertd\convertd.exe ...
if exist build\convertd\ggml-build\build.ninja goto convertd_build
echo [StreamMoE] Configuring macro-enabled ggml (STREAM_MOE_GGUF_ALIGN) for convertd...
"%CMAKE%" -S third_party/llama.cpp -B build/convertd/ggml-build -G Ninja ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=%CLANG% ^
    -DCMAKE_CXX_COMPILER=%CLANGXX% ^
    -DCMAKE_RC_COMPILER=%RC% ^
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF ^
    -DLLAMA_ALL_WARNINGS=OFF -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF ^
    -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF ^
    -DCMAKE_C_FLAGS="-Wno-cast-qual -DSTREAM_MOE_GGUF_ALIGN" ^
    -DCMAKE_CXX_FLAGS="-Wno-cast-qual /EHsc -DSTREAM_MOE_GGUF_ALIGN"
if errorlevel 1 exit /b 1
:convertd_build
"%NINJA%" -C build/convertd/ggml-build ggml-base
if errorlevel 1 exit /b 1
if not exist build\convertd mkdir build\convertd
"%CLANG%" /std:c++17 tools\stream_moe_convertd.cpp /EHsc /MT -DSTREAM_MOE_GGUF_ALIGN ^
    -I%CD:\=/%/third_party/llama.cpp/ggml/include ^
    -I%CD:\=/%/third_party/llama.cpp/ggml/src ^
    %CD:\=/%/build/convertd/ggml-build/ggml/src/ggml-base.lib ^
    ws2_32.lib /Fe:build\convertd\convertd.exe
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
echo   build.bat llamalibs  - Configure and build vendored libllama static libs only
echo   build.bat test       - Build and run all unit tests
echo   build.bat clean      - Remove build\ (all tags)
echo Optional [tag] sub-path for llamalibs/build (default: main).
echo   llamalibs main           - route-B llama-server (build\main)
echo   llamalibs StreamMoE      - flagship: route-B + vulkan, NO prefill export
echo                             (build\StreamMoE; llama-cli dialogue / serving)
echo   llamalibs upstream_dump  - prefill-only export (build\upstream_dump)
echo   llamalibs StreamMoE_dump - route-B + prefill export (build\StreamMoE_dump)
echo   llamalibs StreamMoE_dump_dbg - StreamMoE_dump + STREAM_MOE_TEMP diagnostic
echo                                code (build\StreamMoE_dump_dbg; debug only)
echo   build.bat convertd       - build converter TCP service (build\convertd)
echo   build.bat asan          - ASan llama-server w/ route-B via MSVC cl (build\asan)
echo See docs/PROJECT_STRUCTURE.md for the build layout pattern.
exit /b 0

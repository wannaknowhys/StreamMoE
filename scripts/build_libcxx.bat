@echo off
rem =====================================================================
rem  Build libc++ (pure-LLVM C++ standard library) for -stdlib=libc++.
rem
rem  Why: clang + MSVC STL (VS2019 <functional>) hits a hard
rem  "casts away qualifiers" error that -Wno-cast-qual cannot suppress
rem  (llama.cpp common/jinja). libc++ replaces the C++ STL entirely, so
rem  the whole toolchain stays LLVM-only (clang-cl + lld + llvm-rc).
rem
rem  Requirements:
rem    - LLVM toolchain at F:\Dev\LLVM (clang-cl.exe, lld-link.exe, llvm-rc.exe)
rem    - cmake + ninja at F:\Dev\cmake
rem    - llvm-project source at F:\Dev\llvm-project (cloned here if missing)
rem
rem  Produces (static, MS C++ ABI - the right ABI on Windows):
rem    F:\Dev\llvm-project\build\lib\libc++.lib
rem
rem  NOT built: libcxxabi / libunwind - they are tied to the Itanium C++
rem  ABI and refuse to build for MSVC targets ("Libunwind doesn't build for
rem  MSVC targets"). Windows only needs libc++ itself.
rem =====================================================================
setlocal
set LLVM_PROJECT=F:\Dev\llvm-project
set LLVM_BIN=F:\Dev\LLVM\bin
set CMAKE=F:/Dev/cmake/bin/cmake.exe
set NINJA=F:/Dev/cmake/bin/ninja.exe

if not exist "%LLVM_PROJECT%\runtimes" (
    echo [.] cloning llvm-project (shallow) ...
    git clone --depth 1 https://github.com/llvm/llvm-project "%LLVM_PROJECT%" || exit /b 1
)

echo [.] configuring libc++ (runtimes / clang-cl / static) ...
"%CMAKE%" -G Ninja -S "%LLVM_PROJECT%\runtimes" -B "%LLVM_PROJECT%\build" ^
    -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_C_COMPILER=%LLVM_BIN%\clang-cl.exe ^
    -DCMAKE_CXX_COMPILER=%LLVM_BIN%\clang-cl.exe ^
    -DCMAKE_RC_COMPILER=%LLVM_BIN%\llvm-rc.exe ^
    -DLLVM_ENABLE_RUNTIMES="libcxx" ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [.] building libc++ ...
"%CMAKE%" --build "%LLVM_PROJECT%\build"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [+] libc++ built:
echo       library:  %LLVM_PROJECT%\build\lib\libc++.lib
echo       headers:  %LLVM_PROJECT%\libcxx\include  +  %LLVM_PROJECT%\build\include  (generated __config_site)
echo.
echo     Reference usage with clang-cl:
echo       clang-cl -stdlib=libc++ -nostdinc++ ^
echo         -I "%LLVM_PROJECT%\libcxx\include" -I "%LLVM_PROJECT%\build\include" ^
echo         source.cpp /link "%LLVM_PROJECT%\build\lib\libc++.lib"
endlocal

# ASan 构建（AddressSanitizer）手法 (ASAN_BUILD.md)

> 用于排查内存越界写（如"params vector 被写坏"这类）。ASan 检测堆/栈越界；`VirtualAlloc` 池内越界测不到（ASan 不拦截 VirtualAlloc）。
> 用 **MSVC cl.exe**（VS2026）构建，因为 clang-cl 的 ASan 与 CMake TryCompile 的 Debug flags（/RTC1）冲突。

## 构建命令（build/asan 独立 tag，不碰 main）

```bat
rem 需要 VS2026 vcvars 环境（MSVC 头/库 + ASan 运行时）
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

set "CXXF=/W0 /EHsc /fsanitize=address /Zi /O2"
set "CF=/W0 /fsanitize=address /Zi /O2"

F:/Dev/cmake/bin/cmake.exe -S third_party/llama.cpp -B build/asan/llama-build -G Ninja ^
  -DCMAKE_MAKE_PROGRAM=F:/Dev/cmake/bin/ninja.exe ^
  -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TRY_COMPILE_CONFIGURATION=Release ^
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_ALL_WARNINGS=OFF -DLLAMA_CURL=OFF ^
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF ^
  -DCMAKE_C_FLAGS="%CF%" -DCMAKE_CXX_FLAGS="%CXXF%"

F:/Dev/cmake/bin/ninja.exe -C build/asan/llama-build llama-cli llama-server

rem 运行时 dll（MSVC 14.51 配套，放 exe 旁）
copy "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\clang_rt.asan_dynamic-x86_64.dll" build\asan\llama-build\bin\
```

## 关键点
- **必须 `/MD`（动态 CRT）**——ASan 需要动态 CRT 拦截分配。不要设 `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`。
- **`CMAKE_TRY_COMPILE_CONFIGURATION=Release`**：避开 CMake ABI 检测用 Debug flags（/RTC1 与 ASan 冲突）。
- **`GGML_OPENMP=OFF`**：libomp 与 ASan 兼容问题，排除噪音。
- **dll 版本要匹配编译器**（MSVC 14.51 用 VS2026 的 dll，不能用 LLVM 22 的——符号不匹配 `_asan_new`）。
- clang-cl 的 ASan：需 `/LIBPATH:F:/Dev/LLVM/lib/clang/22/lib/windows` + 显式 `clang_rt.asan_dynamic-x86_64.lib`，且 TryCompile /RTC1 冲突——**优先用 MSVC cl**。

## 运行
```bat
build\asan\llama-build\bin\llama-server.exe -m <model> --host 127.0.0.1 --port 8997 -c 8192 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup --no-webui
```
ASan 报 `ERROR: AddressSanitizer: heap-buffer-overflow / WRITE of size N` 时带调用栈（帧 #N 符号）。

## 已知成功用例（2026-08-27）
用它排除了"overrides 被 4096 pad 挡住 + 专家池张量读 dummy"问题（最终根因靠逐行 log 定位，ASan 在此例未直接报越界，但确认了不是越界写）。

# ASan 构建（AddressSanitizer）手法 (ASAN_BUILD.md)

> 用于排查内存越界写（如"params vector 被写坏"这类）。ASan 检测堆/栈越界；`VirtualAlloc` 池内越界测不到（ASan 不拦截 VirtualAlloc）。
> 用 **MSVC cl.exe**（VS2026）构建，因为 clang-cl 的 ASan 与 CMake TryCompile 的 Debug flags（/RTC1）冲突。
> 最近更新：2026-09-03（整合进 build.bat `asan` 子命令；route-B 经 `-DSTREAM_MOE_FEATURES=route_b` 启用）。

## 构建（build.bat asan 子命令，独立 build/asan，不碰 main）

```bat
build.bat asan
```

- 内部流程：探测 `VS_PATH`（缺省 `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`）→ `call vcvars64` → cmake（cl.exe + `/fsanitize=address`）+ `-DSTREAM_MOE_FEATURES=route_b` → ninja llama-server → 拷贝 MSVC 配套 `clang_rt.asan_dynamic-x86_64.dll` 到 bin。
- 产物：`build\asan\llama-build\bin\llama-server.exe`。
- 手动等价命令（未走 bat 时）：见 `build.bat` 的 `:asan` 块（vcvars + cmake + ninja 三行本质）。

## 关键点
- **必须 `/MD`（动态 CRT）**——ASan 需要动态 CRT 拦截分配。不要设 `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`。
- **`CMAKE_TRY_COMPILE_CONFIGURATION=Release`**：避开 CMake ABI 检测用 Debug flags（/RTC1 与 ASan 冲突）。
- **`GGML_OPENMP=OFF`**：libomp 与 ASan 兼容问题，排除噪音。
- **dll 版本要匹配编译器**（MSVC 14.51 用 VS2026 的 dll，不能用 LLVM 22 的——符号不匹配 `_asan_new`）。build.bat 用 `%VCINSTALLDIR%` 自动定位配套 dll。
- clang-cl 的 ASan：需 `/LIBPATH:F:/Dev/LLVM/lib/clang/22/lib/windows` + 显式 `clang_rt.asan_dynamic-x86_64.lib`，且 TryCompile /RTC1 冲突——**优先用 MSVC cl**。
- **宏机制**：route-B 经根 CMakeLists features 块（`STREAM_MOE_FEATURES=route_b`）全局定义 `STREAM_MOE_ROUTE_B`——不拼 CXX_FLAGS。需要别的 features 组合就手动加 `-DSTREAM_MOE_FEATURES=...`。

## 运行
```bat
build\asan\llama-build\bin\llama-server.exe -m <model> --host 127.0.0.1 --port 8997 -c 8192 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup --no-webui
```
ASan 报 `ERROR: AddressSanitizer: heap-buffer-overflow / WRITE of size N` 时带调用栈（帧 #N 符号）。

## 已知成功用例（2026-08-27）
用它排除了"overrides 被 4096 pad 挡住 + 专家池张量读 dummy"问题（最终根因靠逐行 log 定位，ASan 在此例未直接报越界，但确认了不是越界写）。

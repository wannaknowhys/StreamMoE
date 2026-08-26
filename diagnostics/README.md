# diagnostics/ - 短期诊断代码（独立编译，不入主构建）

> 规范见 docs/PROJECT_STRUCTURE.md §10。本目录代码可直接 git 跟踪，但不进入 `build.bat build` 主构建。

## trace_dump（每层张量对比，对比基线 vs 专家后端）

用途：把每层 `ffn_norm/ffn_out/attn_*` 输出 + MoE 路由 ids 转储成 SMT1 二进制，用 `tools/compare_trace.js` 对比两次运行，定位首个数值分歧。

编译（`-DSTREAM_MOE_TEMP` 让 llama_engine 接上 cb_eval 钩子）：

```bat
set LIBOMP=F:\Dev\LLVM\lib\libomp.lib
set L=build\main\llama-build
set INC=-I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -I third_party/llama.cpp/include -I third_party/llama.cpp/vendor
F:\Dev\LLVM\bin\clang++.exe -std=c++17 -O1 -fopenmp -DSTREAM_MOE_TEMP -DUNICODE -D_UNICODE -D_CRT_SECURE_NO_WARNINGS %INC% ^
    diagnostics/trace_dump.cpp ^
    src/backend/moe_backend.cpp src/backend/minigraph_exec.cpp src/backend/scheduler.cpp ^
    src/io/async_dio_win.cpp src/io/staging_reader.cpp src/loader/moe_loader.cpp src/pool/expert_stats.cpp ^
    src/engine/llama_engine.cpp src/main.cpp src/profile/profiler.cpp ^
    %L%\src\llama.lib %L%\ggml\src\ggml.lib %L%\ggml\src\ggml-base.lib %L%\ggml\src\ggml-cpu.lib %LIBOMP% ^
    -ladvapi32 -lsynchronization -o temp\stream_moe_trace.exe
copy /Y "F:\Dev\LLVM\bin\libomp.dll" temp >nul
```

跑两遍对比：

```bat
set STREAM_MOE_TRACE_FILE=temp\trace_base.bin
temp\stream_moe_trace.exe -m <model> --moe-ram-pool 71680 --temp 0 -c 2048 -t 16 -p "Say hi." -n 4
set STREAM_MOE_TRACE_FILE=temp\trace_eb.bin
temp\stream_moe_trace.exe -m <model> --moe-ram-pool 71680 --expert-backend --temp 0 -c 2048 -t 16 -p "Say hi." -n 4
node tools\compare_trace.js temp\trace_base.bin temp\trace_eb.bin
```

已知结果：首个分歧在 `ffn_moe_gate-3`（第 3 层，hash 层之后第一层 argsort 路由），maxFloatDiff ~3.4。

# 对 llama.cpp 的修改汇总 (VENDORED_MODIFICATIONS.md)

> 记录对 vendored `third_party/llama.cpp` 的所有改动（当前工作区，patch 记录在 `patches/route-b-inject.patch`）。
> 用途：升级/还原/审阅时对照。所有改动都围绕"route B 专家池注入 + dense/moe 分流 + KV 实测显示"。
> 基线：f280b2698。

## 修改文件总览（6 个）

| 文件 | 改动 | 用途 |
|---|---|---|
| `common/common.h` | `common_params` 加 6 字段 | route B 参数载体（主池 2 + 草稿池 2 + flag + prompt-log）|
| `common/arg.cpp` | 注册 6 个 CLI 参数 | `--expert-backend/--moe-ram-pool/--moe-vram-pool/--moe-draft-ram-pool/--moe-draft-vram-pool/--prompt-log` |
| `common/common.cpp` | `common_init_from_params` 注入 route_b_setup | 主模型加载前初始化专家池 + 挂 tensor_buft_overrides |
| `common/speculative.cpp` | draft 加载前注入 route_b_setup | **多模型池**：draft 挂自己的 overrides（不重用主模型）|
| `common/CMakeLists.txt` | `llama-common` 加父仓库 route B 源 + include + Windows 库 | 把 src/server/route_b_inject + backend/io/loader/pool 编译进 llama-server/cli |
| `tools/server/server-context.cpp` | `llama-ext.h` include + 加载后打印 KV 内存 + 析构打印 spec 统计 | 实际 KV 尺寸 + draft 统计 |

## 逐文件明细

### common/common.h
`common_params` 结构体 `n_predict` 后新增：
```cpp
bool    expert_backend   = false; // route MoE expert tensors to the stream_moe pool
size_t  moe_ram_pool_mb  = 0;     // MAIN expert residency budget in MB (0 = 75% free RAM)
size_t  moe_vram_pool_mb = 0;     // MAIN VRAM budget (GPU pool phase)
size_t  moe_draft_ram_pool_mb  = 0; // DRAFT expert RAM budget (0 = full resident)
size_t  moe_draft_vram_pool_mb = 0; // DRAFT VRAM budget (GPU pool phase)
std::string prompt_log_path;      // append /v1/chat/completions bodies
```

### common/arg.cpp
`--swa-full` 后新增 6 个 `add_opt(common_arg(...))`：
- `--expert-backend`（flag）
- `--moe-ram-pool <MB>` / `--moe-vram-pool <MB>`（主池）
- `--moe-draft-ram-pool <MB>` / `--moe-draft-vram-pool <MB>`（草稿池，`0 = full resident`）
- `--prompt-log <PATH>`（lambda 用 `const std::string &`）

### common/common.cpp
`common_model_params_to_llama`（mparams 构造，`no_host` 后）注入：
```cpp
if (params.expert_backend) {
    auto * ovr = stream_moe::route_b_setup(params.model.path.c_str(),
                    params.moe_ram_pool_mb, params.cpuparams.n_threads, false);
    if (ovr) {
        for (auto * p = ovr; p->pattern != nullptr; ++p) params.tensor_buft_overrides.push_back(*p);
        params.tensor_buft_overrides.push_back({nullptr, nullptr}); // terminator
    }
}
```
（`route_b_setup` 幂等：按模型路径去重，draft/MTP 二次上下文不复用主池。）

### common/speculative.cpp
`common_speculative_init_result` ctor 加载 draft 模型前注入（多模型池）：
```cpp
// draft gets its own pool + buft; NEVER reuse the main-model overrides.
if (params.expert_backend) {
    auto * ovr = stream_moe::route_b_setup(model_path.c_str(),
                    params.moe_draft_ram_pool_mb, params.cpuparams.n_threads, true);
    if (ovr) {
        mparams.tensor_buft_overrides = ovr; // draft pool's own (terminated) array
    }
}
```

### common/CMakeLists.txt
`llama-common` 目标末尾追加：
```cmake
set(STREAM_MOE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)
target_include_directories(${TARGET} PRIVATE ${STREAM_MOE_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
target_sources(${TARGET} PRIVATE
    ${STREAM_MOE_SRC}/server/route_b_inject.cpp
    ${STREAM_MOE_SRC}/backend/moe_backend.cpp
    ${STREAM_MOE_SRC}/backend/minigraph_exec.cpp
    ${STREAM_MOE_SRC}/backend/scheduler.cpp
    ${STREAM_MOE_SRC}/io/async_dio_win.cpp
    ${STREAM_MOE_SRC}/io/staging_reader.cpp
    ${STREAM_MOE_SRC}/loader/moe_loader.cpp
    ${STREAM_MOE_SRC}/pool/expert_stats.cpp)
if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32 advapi32 synchronization)
endif()
```

### tools/server/server-context.cpp
1. include `"../src/llama-ext.h"`（拿 `llama_get_memory_breakdown`）。
2. `load_model()` 里 `vocab = llama_model_get_vocab(model_tgt)` 后：
```cpp
size_t kv_bytes = 0;
for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx_tgt)) kv_bytes += mb.context;
SRV_INF("KV Cache Memory (llama.cpp actual): %.2f MB\n", kv_bytes / 1024.0 / 1024.0);
```

## 生命周期 / 还原

- patch 备份：`patches/route-b-inject.patch`（**必须用 cmd 重定向生成**，PS 5.1 的 `>` 会写 UTF-16，导致 `git apply` 报 "No valid patches"）：
  ```bat
  rem 在父仓库根目录执行（cmd 重定向字节透传，产出纯文本 patch）
  cmd /c "git -C third_party/llama.cpp diff > patches\route-b-inject.patch"
  rem 校验 patch 有效（在已应用的工作区应通过 reverse-check）
  git -C third_party/llama.cpp apply --check -R patches\route-b-inject.patch
  ```
- **已跟踪依赖**：`third_party/llama.cpp/src/tsc_timer.h`（`[TMR]` 启动计时，被 route-b-inject 的
  `src/llama.cpp` / `src/llama-context.cpp` include）。**已提交进 vendored 子模块**（2026-08-28，
  route-b-inject 应用后无需重建）——内容是一个 `sm_tmr::timer`（chrono，析构打印 `[TMR] name dur=xx ms`）。
  缺失会导致编译 `'tsc_timer.h' file not found`。
- 应用：`git -C third_party/llama.cpp apply patches/route-b-inject.patch`（在当前干净基线时）。
- 还原：`git -C third_party/llama.cpp apply -R patches/route-b-inject.patch`。
- 升级子模块前必须：还原 patch → 升级 → 重新 apply 并适配新版本（改动点收敛在 5 个文件）。

## 依赖关系

- `route_b_setup` 依赖父仓库 `src/`（backend/io/loader/pool/server/route_b_inject），已由 common/CMakeLists 编入。
- `server-context.cpp` 的 KV 打印依赖 `llama-ext.h`（vendored src 内部 staging API）。

# llama.cpp mmap 调用点与加载流程 (LLAMA_MMAP_CALLS.md)

> 用途：定位"模型权重在哪里进入 mmap"，为"彻底杜绝 mmap / dense 处理"和 lldb 断点调试提供地图。
> 基于 vendored llama.cpp @ f280b2698 + route-b-inject 改动。

## 1. 模型加载流程（从参数到 mmap）

```text
common_params_parse (arg.cpp)                      <- 参数解析（--expert-backend 等在这里确定）
  -> common_init_from_params (common.cpp:1436+)     <- 咱们的注入点：route_b_setup() 在此被调用，
  |                                                    tensor_buft_overrides 挂到 mparams
  -> llama_model_load_from_file (llama-model.cpp)
  -> llama_model_loader ctor (llama-model-loader.cpp)
  -> llama_model_loader::get_mapping() (llama-model-loader.cpp:1364)
  |    std::make_unique<llama_mmap>(file, prefetch)   <<<<< mmap 创建起点（lldb 断点 #1）
  -> llama_mmap ctor (llama-mmap.cpp:439)
  |    Windows: CreateFileMappingA (543) + MapViewOfFile (550)
  |    POSIX : mmap(NULL, size, PROT_READ, ...) (457)  <<<<< 内核 mmap（lldb 断点 #2）
  -> load_all_data (llama-model-loader.cpp:1556+)
  |    mmap 分支：tensor data 直接指向 mapping->addr() + off（零拷贝）
  |    非默认 buft（含咱们的 expert buft）：set_tensor no-op -> 专家不读 mmap 页
  -> llama_model.cpp:1703  将 mmap 区域映射为 backend buffer（dense 零拷贝）
```

## 2. mmap 调用点表

| 位置 | 功能 | 角色 |
|---|---|---|
| `llama-mmap.cpp:457` | POSIX `mmap()` 内核映射 | **实际 mmap 调用**（lldb 断点）|
| `llama-mmap.cpp:543` | Windows `CreateFileMappingA` | Windows 映射句柄 |
| `llama-mmap.cpp:550` | Windows `MapViewOfFile` | Windows 视图映射 |
| `llama-mmap.cpp:588` | `UnmapViewOfFile` | 释放 |
| `llama-model-loader.cpp:1364` | `make_unique<llama_mmap>`（每分片）| **映射创建点**（lldb 断点）|
| `llama-model-loader.cpp:1367` | `llama_mlock`（mlock 模式）| mlock 化 |
| `llama-model-loader.cpp:817` | use_mmap 平台支持检查 | 开关 |
| `llama-model-loader.cpp:1189` | overrides + mmap 警告 | 诊断 |
| `llama-model-loader.cpp:1212` | mmap 时避免 host buffer | 分配策略 |
| `llama-model-loader.cpp:1556+` | `load_all_data` mmap 分支（tensor 指向映射页）| **零拷贝赋值点** |
| `llama-model.cpp:1131` | `llama_mmaps mappings`（model 持有映射）| 生命周期 |
| `llama-model.cpp:1703` | mmap 区域 -> backend buffer（dense）| 零拷贝 |
| `llama.cpp:56/70` | load_mode 枚举（mmap / mmap+mlock）| 参数 |
| `llama-mmap.h:43` | `llama_mmap` 接口 | 接口 |

## 3. dense vs expert 分流（设计）

- **统一流程**：对本次加载的**每个 GGUF**（主体模型 / draft / mmproj）都做 dense/moe 识别——
  `moe_loader::parse_gguf_topology` 已在 parse 期统计 `dense_total_bytes / expert_total_bytes / dense_tensor_names`（见 `docs/` 说明）。
- **分流**：
  - 主体模型 `expert_total_bytes > 0` → **moe 流程**（route B：`--expert-backend`，expert 走池，dense 走默认）。
  - draft / mmproj 等通常 `expert_total_bytes == 0` → **dense 流程**（不动，`--expert-backend` 对它 no-op）。
  - 判别在 `route_b_setup`（parse 后）做：`g_topo->n_expert == 0` 即 dense。
- 加载起点（要改的地方）：`common_init_from_params` 的 route_b 注入点（已实现）——按 `n_expert` 决定是否挂 overrides。

## 4. lldb 断点方案

```bash
# 启动 server，在 mmap 处下断点，call 即停，bt 看调用链
lldb -- build/main/llama-build/bin/llama-server.exe -m <model> ... --expert-backend --moe-ram-pool 71680 --fit off
(lldb) breakpoint set -n llama_mmap::llama_mmap        # 构造（每分片一次）
(lldb) breakpoint set -n CreateFileMappingA            # 或直接 Windows API
(lldb) breakpoint set -n MapViewOfFile
(lldb) run
(lldb) bt                                            # 看是谁触发的 mmap（应到 llama-model-loader.cpp:1364）
(lldb) thread list / frame select N / frame variable  # 看调用上下文
```

断点 #1（`llama_mmap` 构造）能确认：mmap 只发生在 `llama_model_loader::get_mapping`，且只对 dense 分片（专家 buft 的 `set_tensor` no-op 不 fault 映射页）。

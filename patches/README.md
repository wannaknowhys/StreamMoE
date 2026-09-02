# Patches (StreamMoE vendored patch 体系)

> 用法：vendored `third_party/llama.cpp` 永不 commit；StreamMoE 全部改动以 patch 记录，
> 按 phase 顺序 `git apply` 叠加复现。生成 patch 一律 `cmd /c "git -C third_party/llama.cpp
> diff HEAD -- <文件> > patches\x.patch"`（PS 重定向写 UTF-16，禁）。
> 最近整理：2026-09-03（frag 全主仓库 + features 宏机制 + server-context 纯锚点）。

## 总览（2026-09-03 重构后）

- **vendored HEAD = 纯上游 `f280b2698`**；工作区 = 5 patch 全 apply 态。
- **frag（内容片段）全部在主仓库** `patches/<phase>/...`（单一来源，普通文件随主仓库 commit）：
  - `patches/route-b/common/`：route-b 的 common 层 5 frag + server-context spec 打印 2 frag
  - `patches/prefill-export/common/`：prefill 的 common 层 4 frag
  - `patches/prefill-export/include/`：prefill 的 llama.h 层 3 frag
- **features 宏机制**：`build.bat llamalibs <tag>` 传 `-DSTREAM_MOE_FEATURES`（route_b /
  prefill_export / route_b,prefill_export）→ vendored 根 `CMakeLists.txt` features 块
  用 `add_compile_definitions` **全局定义宏 + `include_directories` 指向主仓库 frag 目录**
  （对当次构建全部 target 生效——防新文件/新 target 忘配宏被静默丢弃）。**宏不拼 CXX_FLAGS**。
- **共享文件 = phase1 只加 include 锚点**（宏保护短名 `#include "xxx.frag"`，编译期靠全局
  include 路径展开主仓库 frag）；**功能 patch 只改专属文件** + 主仓库 frag（不碰共享文件）。
- 无宏构建（features 空 / 不 apply 2a/2b）= 纯上游等价（include 行被预处理跳过）。

## 文件归属（各 patch 各管各的文件，绝不 `git diff >` 全量抄）

### Phase 1（必选，互不依赖）
- `streammoe-macros.patch`：
  - `CMakeLists.txt`（根——features 块）
  - `common/arg.cpp`（route-b/prefill args 锚点）
  - `common/common.cpp` / `common/common.h`（route-b/prefill 锚点）
  - `include/llama.h`（prefill 3 锚点：includes/params/apis——frag 在 `patches/prefill-export/include/`）
  - `tools/server/server-context.cpp`（**3 锚点**：route-b spec slot/dtore + prefill nout）
- `tsc_timer.patch`：`src/tsc_timer.h`（[TMR] `sm_tmr::timer`，析构打印经 `STREAM_MOE_TMR` env 门控）

### Phase 2a（可选）route-b-inject.patch
- `common/CMakeLists.txt`（`STREAM_MOE_SRC` = 主仓库 `src/`，源列表 + PRIVATE include——引擎代码编进 llama-common）
- `common/speculative.cpp` / `common/speculative.h`（route-b draft 池绑定 + 统计）
- `src/llama-model-loader.cpp` / `src/llama-model-loader.h`（bounds check skip）
- `src/llama-model.cpp`、`src/llama.cpp`

### Phase 2b（可选）prefill-export-llama.patch
- `src/llama-context.cpp` / `src/llama-context.h`（prefill 导出 + 专家历史 + cb_eval 图内抓取）
- `src/llama-kv-cache.cpp` / `src/llama-kv-cache.h`
- `tools/server/server.cpp`（/shutdown 端点）

### 独立 gguf-alignment.patch（转换器工具）
- `ggml/include/gguf.h` + `ggml/src/gguf.cpp`（`gguf_set_alignment`——convertd 需要；与推理构建无关，任意时 apply）

> 注意：route-b / prefill **不含任何 frag new-file**（frag 在主仓库常驻）；**不含 server-context**
> 专属改动（锚点全在 phase1 macros）。

## 应用顺序与验证

```
git -C third_party/llama.cpp apply patches\streammoe-macros.patch patches\tsc_timer.patch \
    patches\route-b-inject.patch patches\gguf-alignment.patch patches\prefill-export-llama.patch
```

- Phase 1 必选（顺序可互换）；2a/2b 可选组合；gguf-alignment 独立。
- 验证（A4 做过）：临时 worktree 检出 HEAD → 按序 apply → 与工作区逐字节一致（21 文件 hash）。
- 叠加纪律（README 旧版铁律沿用）：在已有 patch 基础上改代码前先 commit 父仓库 + 快照
  `git -C third_party/llama.cpp diff > temp/patch_backup_<date>/working-tree-full.patch`。
- 每次 vendored 改动收尾：重生成受影响 patch → 临时 worktree apply 验证逐字节一致 → commit。

## 构建变体（tag → features → 输出）

| tag | STREAM_MOE_FEATURES | 宏（根 CMakeLists 定义）| 用途 |
|---|---|---|---|
| `main` | route_b | STREAM_MOE_ROUTE_B | route-B 完整推理（生产）|
| `upstream_dump` | prefill_export | STREAM_MOE_PREFILL_EXPORT | prefill 导出（上游基准，无 vulkan）|
| `upstream_vulkan_dump` | prefill_export | STREAM_MOE_PREFILL_EXPORT | 同 + vulkan（Vulkan0 对比）|
| `StreamMoE_dump` | route_b,prefill_export | 两者 | 完整 StreamMoE 导出 |
| `asan` | route_b | STREAM_MOE_ROUTE_B | ASan（MSVC cl，build.bat asan）|
| convertd | （无 features）| STREAM_MOE_GGUF_ALIGN（独立）| 转换器 |

- 变体隔离：每 tag 独立 `build/<tag>/llama-build`，features 固化在各自 CMakeCache。
- vulkan 构建修复无 patch：`build.bat` 经上游 `VULKAN_SHADER_GEN_CMAKE_ARGS` hook 传工具链。
- POSIX：`Makefile` 薄转发（cmake+ninja），TAG→features 映射同 build.bat（见 Makefile）。

## 主仓库 route-b 引擎源码（不靠 patch）

`src/`（backend/io/loader/pool/profile/server）随主仓库 commit，经 2a 的
`common/CMakeLists.txt`（STREAM_MOE_SRC）编进 llama-common。这是 StreamMoE 引擎本体，
**不属 vendored patch**——单独 git 管理。

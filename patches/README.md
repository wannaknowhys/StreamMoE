# Patches (StreamMoE vendored patch 体系)

> 用法：vendored `third_party/llama.cpp` 永不 commit；StreamMoE 全部改动以 patch 记录，
> 按 phase 顺序 `git apply` 叠加复现。生成 patch 一律 `cmd /c "git -C third_party/llama.cpp
> diff HEAD -- <文件> > patches\x.patch"`（PS 重定向写 UTF-16，禁）。
> 最近整理：2026-09-03（frag 全主仓库 + features 宏机制 + server-context 纯锚点）。

## 总览（2026-09-03 重构后）

- **vendored HEAD = 纯上游 `f280b2698`**；工作区 = 4 patch 全 apply 态。
- **frag（内容片段）全部在主仓库** `patches/<phase>/...`（单一来源，普通文件随主仓库 commit）：
  - `patches/route-b/common/`：route-b 的 common 层 frag（含 `stmoe_routeb_vk_hostmap.frag`——
    ggml-vulkan host map 导出函数体）+ server-context spec 打印 2 frag
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
  - `common/arg.cpp`（route-b/prefill args 锚点 + route-b 参数追踪/placement 检查锚点）
  - `common/common.cpp` / `common/common.h`（route-b/prefill 锚点）
  - `common/preset.cpp`（route-b 参数追踪锚点）
  - `common/speculative.cpp` / `common/speculative.h`（route-b draft 池/统计锚点）
  - `common/CMakeLists.txt`（`include(src/cmake/stmoe_routeb_sources.cmake OPTIONAL)`——引擎源列表搬主仓库）
  - `include/llama.h`（prefill 3 锚点：includes/params/apis——frag 在 `patches/prefill-export/include/`）
  - `src/llama-context.cpp` / `src/llama-context.h`（route-b 5 锚点 + prefill 导出 10 锚点）
  - `src/llama-kv-cache.cpp` / `src/llama-kv-cache.h`（prefill `get_v_storage` 锚点）
  - `tools/server/server.cpp`（prefill prefill-only/shutdown 锚点）
  - `tools/server/server-context.cpp`（**3 锚点**：route-b spec slot/dtore + prefill nout）
  - `ggml/src/ggml-vulkan/ggml-vulkan.cpp`（**1 锚点**：`STREAM_MOE_ROUTE_B` 保护 include
    `stmoe_routeb_vk_hostmap.frag`——函数体在 `patches/route-b/common/`；2026-09 由原
    直插 patch `streammoe-vk-hostmap.patch` 改造，随 macros 走 features 全局 include 路径）
- `tsc_timer.patch`：`src/tsc_timer.h`（[TMR] `sm_tmr::timer`，析构打印经 `STREAM_MOE_TMR` env 门控）

### Phase 2a（可选）route-b-inject.patch

> 2026-10-09 起只含**改既有逻辑行**的部分（frag 化不了的）：`src/llama-model-loader.cpp` / `.h`（bounds check skip）、`src/llama-model.cpp` / `.h`（逻辑/物理设备分离 + dense placement）、`src/llama.cpp`、KV 三文件物理设备选择与冲突拒绝。`common/CMakeLists.txt`（STREAM_MOE_SRC 源列表，已搬 `src/cmake/stmoe_routeb_sources.cmake`）、`common/arg.cpp`、`common/preset.cpp`、`common/speculative.cpp` / `.h`、`src/llama-context.cpp` 全部走 phase1 锚点 + frag。

- `src/llama-model-loader.cpp` / `src/llama-model-loader.h`（bounds check skip）
- `src/llama-model.cpp`、`src/llama.cpp`
- `src/llama-model.h`（`dev_layer_physical` / `route_b_enabled` 声明）
- `src/llama-kv-cache.cpp` / `src/llama-kv-cache-dsv4.cpp` / `src/llama-memory-recurrent.cpp`（物理设备选择与冲突拒绝；kv-cache.cpp 另有 phase1 的 `get_v_storage` 锚点——两 patch 唯一共用文件，hunk 不重叠，顺序 macros → route-b 固定）
- `common/arg.cpp` / `common/preset.cpp`（参数追踪）、`common/speculative.cpp` / `.h`（draft 池绑定 + 统计）、`src/llama-context.cpp`（`route_b_begin_graph()`）、`common/CMakeLists.txt`（STREAM_MOE_SRC 源列表）——**2026-10-09 已全部 frag 化，见 phase1**

### Phase 2b（已消除，2026-10-09）

- `prefill-export-llama.patch` **已删除**：prefill 导出（llama-context.cpp/h + llama-kv-cache.cpp/h + server.cpp）全部搬为 phase1 锚点 + `patches/prefill-export/common/*.frag`（~400 行导出体 = `stmoe_prefill_export_body.frag`）。prefill 功能无变化，`STREAM_MOE_PREFILL_EXPORT` 宏门控不变。

> 注意：route-b / prefill **不含任何 frag new-file**（frag 在主仓库常驻）；**不含 server-context**
> 专属改动（锚点全在 phase1 macros）。转换器不再需要 gguf patch（4K 对齐经内存 seed 上下文实现，
> 见 `src/convert/writer.cpp`）。

## 应用顺序与验证

```
git -C third_party/llama.cpp apply patches\streammoe-macros.patch patches\tsc_timer.patch \
    patches\route-b-inject.patch
```

- Phase 1 必选（顺序可互换）；2a 可选。
- 验证：临时 worktree 检出 HEAD → 按序 apply → 与工作区逐字节一致（2026-10-09：新 3 patch 栈 24 文件逐字节一致，见 `temp/patch_replay_new` 方法）。
- 叠加纪律（README 旧版铁律沿用）：在已有 patch 基础上改代码前先 commit 父仓库 + 快照
  `git -C third_party/llama.cpp diff > temp/patch_backup_<date>/working-tree-full.patch`。
- 每次 vendored 改动收尾：重生成受影响 patch → 临时 worktree apply 验证逐字节一致 → commit。

## 构建变体（tag → features → 输出）

| tag | STREAM_MOE_FEATURES | 宏（根 CMakeLists 定义）| 用途 |
|---|---|---|---|
| `main` | route_b | STREAM_MOE_ROUTE_B | route-B 完整推理（生产，纯 CPU 数值）|
| `StreamMoE` | route_b | STREAM_MOE_ROUTE_B | **旗舰**：route-B + vulkan（vram device-pool 路径），无导出代码——llama-cli 对话/服务 |
| `upstream_dump` | prefill_export | STREAM_MOE_PREFILL_EXPORT | prefill 导出（上游基准，无 vulkan）|
| `upstream_vulkan_dump` | prefill_export | STREAM_MOE_PREFILL_EXPORT | 同 + vulkan（Vulkan0 对比）|
| `StreamMoE_dump` | route_b,prefill_export | 两者 | 完整 StreamMoE 导出 |
| `asan` | route_b | STREAM_MOE_ROUTE_B | ASan（MSVC cl，build.bat asan）|

- 变体隔离：每 tag 独立 `build/<tag>/llama-build`，features 固化在各自 CMakeCache。
- vulkan 构建修复无 patch：`build.bat` 经上游 `VULKAN_SHADER_GEN_CMAKE_ARGS` hook 传工具链。
- POSIX：`Makefile` 薄转发（cmake+ninja），TAG→features 映射同 build.bat（见 Makefile）。

## 主仓库 route-b 引擎源码（不靠 patch）

`src/`（backend/io/loader/pool/profile/server）随主仓库 commit，经 2a 的
`common/CMakeLists.txt`（STREAM_MOE_SRC）编进 llama-common。这是 StreamMoE 引擎本体，
**不属 vendored patch**——单独 git 管理。

## 补丁铁律（添加 2026-09：消灭 phase2a/2b patch 的长线目标）

＞ 长线目标（2026-10-09 进展）：**phase2b 已消除**（prefill-export-llama.patch 删除，导出体全进 frag）；2a 简单部分已消除（arg/preset/speculative/CMake/context 钩子全进锚点+frag，源列表搬 `src/cmake/`）。剩余 2a 逻辑改行（loader/model/KV/llama.cpp）因改的是既有逻辑行而保留 patch——frag 只能缩成 1 行锚点调用，改行动
作本身消灭不掉。只留 streammoe-macros.patch + route-b-inject.patch + tsc_timer.patch。

**vendored 需改动时的处理：**
1. 小插入（钩子/几行调用）→ 不直接 patch vendored：R phase1 → phase1 加 include 锚点→ apply phase1 → 写 frag 内容（主仓库 patches/<phase>/）
2. 大块逻辑 → 倾向主仓库独立 cpp（src/ 下，CMake STREAM_MOE_SRC 编入，像 route_b_chain.cpp），vendored 只锚点调用；或大块也 frag（include 大 frag 可行）
3. **不得不直接改 vendored 时 → 先问用户**（不要自作主张积累欠账）

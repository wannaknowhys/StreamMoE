# build 图入口点 与 output[] 控制

> [English](GRAPH_BUILD_OUTPUT.md) | [简体中文](GRAPH_BUILD_OUTPUT.zh-CN.md)
>
> 下列行号基于**纯上游 `f280b2698`**（vendored 工作区无补丁状态）。应用 phase1/prefill/route-b patch 后行号会偏移——请按符号搜索，勿按行号定位。

## 1. build 图的三个入口

`model.build_graph()` 全库只有三处调用，都在 `src/llama-context.cpp`：

| #   | 调用点（文件:行）                                     | 上层调用链                                                                                                                  | 用途                                                                                              | build 出的图                                                 |
| :-- | :---------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :----------------------------------------------------------- |
| 1   | `graph_reserve`：`src/llama-context.cpp:2431`         | `sched_reserve()`(581) -> 633(PP 最坏情况)、653(TG)、668(再 PP)；`resolve_fused_ops`(513，融合 op 探测)；memory 更新后(830) | **预留**最坏情况图以确定 sched split / compute buffer 大小；**从不执行**                          | 全尺寸 PP 图（n_tokens=min(n_ctx,n_ubatch)）+ 单 seq TG 图   |
| 2   | `process_ubatch`：`src/llama-context.cpp:1358`        | `llama_decode` 主循环(1816，每 ubatch 一次)；`llama_encode`(1463)                                                           | **真实推理**图。先查 `can_reuse`(1339)：参数与上次图相同则复用，否则重建                          | 实际 ubatch 形状（prefill=多 token，decode=n_seqs 个 token） |
| 3   | `llama_encode` 直接路径：`src/llama-context.cpp:3418` | `llama_encode` API                                                                                                          | encode/embedding 路径，**每 ubatch 强制** `res->reset()` + build（不走复用），自带 compute 上下文 | 实际 ubatch 图                                               |

prefill 和 decode 都走**入口 #2**（`process_ubatch`）——区别只在 ubatch 参数（n_tokens/n_seqs/n_outputs），**不存在独立的 prefill/decode 入口**。

## 2. vendored 设置 output[]（batch.logits / ubatch.output）的位置

链路：server 决定哪些 token 要输出 -> `batch.logits[]` -> `llama_decode` -> `ubatch.output[]` -> `n_outputs` -> LM head gather。

| 层                                        | 文件:行                                                                                                                                            | 作用                                               |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| server 决定每 token 是否输出              | `tools/server/server-context.cpp:151-154`（`server_batch::set_output`）——update_slots / process 里的调用方默认只把**最后一个** token 置 true       | 存 `tokens[idx].output`                            |
| batch 渲染 -> `batch.logits[]`            | `common/common.cpp:1851`（`common_batch_add`）：`batch.logits[batch.n_tokens] = logits;`——由 `server_batch::render()`(server-context.cpp:156) 调用 | 把输出标记写进 llama_batch                         |
| `--prefill-from` 模式（prefill patch 改） | `tools/server/server.cpp`：`lg.back() = 1;`                                                                                                        | prefill-from 一次性 decode 强制最后一个 token 输出 |
| llama_batch -> llama_ubatch               | `src/llama-context.cpp`（`llama_decode` 内 ubatch 准备）                                                                                           | 把 `batch.logits` 拷入 `ubatch.output[]`           |
| `n_outputs` 计数                          | `src/llama-context.cpp:1800-1811`                                                                                                                  | `n_outputs = sum(ubatch.output[i])`                |
| 图内 out_ids 张量                         | `src/llama-graph.cpp:2425-2444`（`build_inp_out_ids`）、`199-224`（`set_input` 收集 `ubatch.output[i]` 为 true 的位置）                            | LM head 只算这 `n_outputs` 行输出                  |

**注意**：hidden（`t_h_nextn`）和 embd（`result_norm` = `t_embd`）是**整张图节点张量**——天然覆盖全部 token（layers 对所有 token 算），与 output[] 无关。只有 logits（LM head）才被 n_outputs 剪枝。所以 prefill 导出（prefill-export-llama.patch）即使 server 只置最后 token output，捕获的 embd/hidden 也覆盖全部 token；**只有要全部 token 的 logits 时才需要** `--logits-all` / 全置 output[]。

## 3. 直接 patch 改，还是 phase1 锚点 + frag？（2026-10-09 修订）

判断准则：**纯插入/新代码走 phase1 include 锚点 + 主仓库 frag；只有"改既有逻辑行"才留在功能 patch 里直接改。**

### phase1 锚点 + frag（插入/新代码，任一功能都适用）

- **prefill 导出（2b patch 于 2026-10-09 消除）**：`src/llama-context.cpp`（约 400 行导出体 + 9 处钩子）、`src/llama-context.h`（`export_*` 成员）、`src/llama-kv-cache.h/.cpp`（`get_v_storage`）、`tools/server/server.cpp`（prefill-only + /shutdown）——全部 phase1 锚点 + `patches/prefill-export/common/*.frag`，`#ifdef STREAM_MOE_PREFILL_EXPORT` 门控。
- **route-b 钩子（2a 于 2026-10-09 瘦身）**：`common/arg.cpp` / `common/preset.cpp`（参数追踪）、`common/speculative.cpp/.h`（draft 池绑定 + 统计）、`src/llama-context.cpp`（`route_b_begin_graph()`）、`common/CMakeLists.txt`（源列表搬 `src/cmake/stmoe_routeb_sources.cmake`）——全部 phase1 锚点 + frag。
- 两边都要插字段的共享结构（`common_params` / `llama.h`）沿用原锚点 + 各功能 frag 方案。

### 直接 patch 改（改既有逻辑行——2a 剩余）

- `src/llama-model-loader.cpp/.h`（bounds-check 跳过、route-B 加载分支）、`src/llama-model.cpp/.h`（逻辑/物理设备分离、dense placement）、`src/llama-kv-cache*.cpp` + `llama-memory-recurrent.cpp`（`dev_layer_physical` + no-kv-offload 拒绝）、`src/llama.cpp`（计时器）。frag 只能把这些缩成 1 行锚点调用，改既有行的动作本身消不掉——所以留在 `route-b-inject.patch`。
- `llama-kv-cache.cpp` 是两个 patch 唯一共用的文件（phase1 的 `get_v_storage` 锚点 + route-b 的 ctor 逻辑，hunk 不重叠，顺序固定 macros → route-b）。

### 共享结构示例（`common_params`）

- **Phase 1（streammoe-macros.patch）**只在共享结构里加 include 锚点：

  ```cpp
  struct common_params {
  #ifdef STREAM_MOE_PREFILL_EXPORT
  #include "stmoe_prefill_common_params.frag"
  #endif
      int32_t n_predict = -1;
  #ifdef STREAM_MOE_ROUTE_B
  #include "stmoe_routeb_common_params.frag"
  #endif
      ...
  };
  ```

- 功能只新增 `.frag` 文件——不再改共享文件。
- 宏由 `build.bat llamalibs <tag>` **编译时定义**：`main` -> `-DSTREAM_MOE_ROUTE_B`；`upstream_dump` -> `-DSTREAM_MOE_PREFILL_EXPORT`；`StreamMoE_dump` -> 两者；宏未定义时 include 行被预处理跳过（phase1 单独可编译 = 纯上游等价）。

### 决策清单（2026-10-09）

1. 纯插入 / 新函数 / 新字段 -> **phase1 锚点 + frag**（vendored 不留直接代码）。
2. 改既有逻辑行（分支条件、调用目标、边界检查） -> **route-b-inject.patch 直接 hunk**（`#ifdef STREAM_MOE_ROUTE_B` 门控，保证其他 tag 与上游一致）。
3. 两个 patch 共用文件（目前仅 `llama-kv-cache.cpp`） -> 重生成时按 patch 拆 hunk，顺序固定 macros → route-b。

# build 图入口点 与 output[] 控制

> [English](GRAPH_BUILD_OUTPUT.md) | [简体中文](GRAPH_BUILD_OUTPUT.zh-CN.md)
>
> 下列行号基于**纯上游 `f280b2698`**（vendored 工作区无补丁状态）。应用 phase1/prefill/route-b patch 后行号会偏移——请按符号搜索，勿按行号定位。

## 1. build 图的三个入口

`model.build_graph()` 全库只有三处调用，都在 `src/llama-context.cpp`：

| # | 调用点（文件:行）| 上层调用链 | 用途 | build 出的图 |
|---|---|---|---|---|
| 1 | `graph_reserve`：`src/llama-context.cpp:2431` | `sched_reserve()`(581) -> 633(PP 最坏情况)、653(TG)、668(再 PP)；`resolve_fused_ops`(513，融合 op 探测)；memory 更新后(830) | **预留**最坏情况图以确定 sched split / compute buffer 大小；**从不执行** | 全尺寸 PP 图（n_tokens=min(n_ctx,n_ubatch)）+ 单 seq TG 图 |
| 2 | `process_ubatch`：`src/llama-context.cpp:1358` | `llama_decode` 主循环(1816，每 ubatch 一次)；`llama_encode`(1463) | **真实推理**图。先查 `can_reuse`(1339)：参数与上次图相同则复用，否则重建 | 实际 ubatch 形状（prefill=多 token，decode=n_seqs 个 token）|
| 3 | `llama_encode` 直接路径：`src/llama-context.cpp:3418` | `llama_encode` API | encode/embedding 路径，**每 ubatch 强制** `res->reset()` + build（不走复用），自带 compute 上下文 | 实际 ubatch 图 |

prefill 和 decode 都走**入口 #2**（`process_ubatch`）——区别只在 ubatch 参数（n_tokens/n_seqs/n_outputs），**不存在独立的 prefill/decode 入口**。

## 2. vendored 设置 output[]（batch.logits / ubatch.output）的位置

链路：server 决定哪些 token 要输出 -> `batch.logits[]` -> `llama_decode` -> `ubatch.output[]` -> `n_outputs` -> LM head gather。

| 层 | 文件:行 | 作用 |
|---|---|---|
| server 决定每 token 是否输出 | `tools/server/server-context.cpp:151-154`（`server_batch::set_output`）——update_slots / process 里的调用方默认只把**最后一个** token 置 true | 存 `tokens[idx].output` |
| batch 渲染 -> `batch.logits[]` | `common/common.cpp:1851`（`common_batch_add`）：`batch.logits[batch.n_tokens] = logits;`——由 `server_batch::render()`(server-context.cpp:156) 调用 | 把输出标记写进 llama_batch |
| `--prefill-from` 模式（prefill patch 改）| `tools/server/server.cpp`：`lg.back() = 1;` | prefill-from 一次性 decode 强制最后一个 token 输出 |
| llama_batch -> llama_ubatch | `src/llama-context.cpp`（`llama_decode` 内 ubatch 准备）| 把 `batch.logits` 拷入 `ubatch.output[]` |
| `n_outputs` 计数 | `src/llama-context.cpp:1800-1811` | `n_outputs = sum(ubatch.output[i])` |
| 图内 out_ids 张量 | `src/llama-graph.cpp:2425-2444`（`build_inp_out_ids`）、`199-224`（`set_input` 收集 `ubatch.output[i]` 为 true 的位置）| LM head 只算这 `n_outputs` 行输出 |

**注意**：hidden（`t_h_nextn`）和 embd（`result_norm` = `t_embd`）是**整张图节点张量**——天然覆盖全部 token（layers 对所有 token 算），与 output[] 无关。只有 logits（LM head）才被 n_outputs 剪枝。所以 prefill 导出（prefill-export-llama.patch）即使 server 只置最后 token output，捕获的 embd/hidden 也覆盖全部 token；**只有要全部 token 的 logits 时才需要** `--logits-all` / 全置 output[]。

## 3. 直接 patch 改，还是 phase1 宏包裹？

判断准则：**只有"两个功能 patch 都改同一共享结构"时才需要 phase1 include 锚点方案**，其余全部在功能 patch 里直接改。

### 功能 patch 直接改（单一属主）
- **prefill（prefill-export-llama.patch）独占**：`src/llama-context.cpp`（cb_eval 替换、export_t_* 发布、capture_*、export_token_seq、析构 flush）、`src/llama-context.h`（export_* 成员）、`src/llama-kv-cache.h/.cpp`、`tools/server/server.cpp`、`tools/server/server-context.cpp`（export_dir 映射）。prefill 专属代码在这些文件里包 `#ifdef STREAM_MOE_PREFILL_EXPORT`，保证 route-b-only 构建不编译多余代码。
- **route-b（route-b-inject.patch）独占**：`common/speculative.cpp/.h`、`src/llama-model-loader.cpp`、`src/llama-model.cpp`、`src/llama.cpp`、`tools/server/server-context.cpp`（route_b_setup 注入）、`common/CMakeLists.txt`。**route-b 目前不碰 `llama-context.cpp`**——与 prefill 在此文件零重叠。
- 上述 build 三入口 + output[]/n_outputs/out_ids 逻辑都在 `llama-context.cpp` / `llama-graph.cpp` / `server-context.cpp` / `common.cpp`——目前只被 prefill 改——**加 hook 是直接 patch 改**（包 `#ifdef STREAM_MOE_PREFILL_EXPORT`），不需要 phase1。

### phase1 宏包裹（两个 patch 改同一共享结构）
唯一真正的共享结构冲突是 **`common_params`**（`common/common.h` 声明、`common/common.cpp` / `common/arg.cpp` 解析、`llama.h` 的 params）——两个功能都要加字段：

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
- **功能 patch 只新增 `.frag` 文件**（`common/stmoe_routeb_*.frag`、`common/stmoe_prefill_*.frag`、`include/stmoe_prefill_llama_*.frag`）——**不再改** `common.h/common.cpp/arg.cpp/llama.h`，所以 phase2a/2b 的 apply 顺序无关、永不冲突。
- 宏由 `build.bat llamalibs <tag>` **编译时定义**：`main` -> `-DSTREAM_MOE_ROUTE_B`；`upstream_dump` -> `-DSTREAM_MOE_PREFILL_EXPORT`；`StreamMoE_dump` -> 两者；宏未定义时 include 行被预处理跳过（phase1 单独可编译 = 纯上游等价）。

### 决策清单
1. 该文件/位置是否被 route-b 和 prefill **都**改？
   - 是，且是两边都要插字段的共享结构/头（common_params / llama.h）-> **phase1 锚点 + 功能 .frag**。
   - 是，但位置不重叠（如 llama-context.cpp）-> 各自功能 patch 直接改自己的区域，`#ifdef` 门控；保持 2a/2b hunk 不重叠以维持任意 apply 顺序。
2. 只被一个功能改 -> **直接 patch 改**（建议 `#ifdef STREAM_MOE_*` 门控，保证其他 tag 构建与上游逐字节一致）。

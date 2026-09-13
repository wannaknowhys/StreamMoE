# L2 整层接管 Patch 评审汇总（GPT / Claude / DeepSeek 三方分析合并）

> 本文汇总三份外部 AI 对 `l2_whole_layer.patch`（L2 whole-layer ownership）的评审，
> 去重后按「潜在 bug / 调试手段 / 架构设计建议」三类整理。纯中文。
>
> 来源：
> - [GPT] `chatgpt_Patch分段测试与漏洞_.md`
> - [Claude] `claude_Patch分段启用和测试方案.md`
> - [DeepSeek] `deepseek Patch bug分析.md`
>
> 评审对象：`debug_patch/l2-whole-layer/l2_whole_layer.patch`（commit `6cda201`，`STREAM_MOE_TEMP` 门控的 DEBUG 路径）。

---

## 0. 背景与总体判断

**Patch 做了什么**：把每一层的 dense（attn + norm + residual）节点也收编进 stream_moe 后端，
让一层在同一个 `exec_layer_burst` 里按 `dense_head → MoE burst → dense_tail` 跑完，
从而避免 dense 与 MoE 被 scheduler 切成多个 split。

三方共识：

- **方向正确，已跨过最难的概念门槛**：从「接管 MoE 节点」升级为「接管整个 layer」。GPT 给架构 8/10。
- **当前正确性不足（GPT 约 5/10）**：注释本身已承认 "L2 is not numerically correct yet"（output-token reduction path diverges）。
- **真正危险的不是 MoE burst，而是 dense 节点的重新执行方式 + layer attribution + scheduler ownership 改变后，原有的依赖/alias 语义是否仍成立**（GPT）。
- GPT 明确建议：**不要首先怀疑 `exec_layer_burst_chain_buckets()`，而应优先怀疑「L2 把原本由 scheduler 自然执行的 dense/view/contiguous 节点改成手工 clone 执行后，某个 activation 的 buffer/data/view 语义发生变化」**。

---

## 一、潜在 Bug

按风险等级归类。`[GPT]`/`[Claude]`/`[DeepSeek]` 标注来源；三方都提到或重叠的标 ★。

### A. 高危：可能直接数值错 / 整层漏跑（静默，不 crash）

| # | 问题 | 来源 | 后果 |
|---|------|------|------|
| A1 ★ | `moe_exec_mul_mat_id` 开头 `if (is_alias_op(nodes[0])) return SUCCESS;`。whole-layer 下 split 变大，可能变成 `VIEW/RESHAPE → 真计算`，直接 return 会跳过后面所有计算 | DeepSeek | 整 split 漏执行 |
| A2 ★ | `first_node = ln->front()` 可能是 VIEW/RESHAPE/TRANSPOSE/PERMUTE。alias 节点在 `moe_chain_assign_backend` 里被跳过、不 `set_tensor_backend`，可能根本不在 split 的 `nodes[]` 里 → `has_first=false` → 整层不 burst | DeepSeek / GPT / Claude | 整层不执行 |
| A3 ★ | `ffn_moe_out` 找不到时（名字变、不在 lns 中）`down[]` 全 0 → `dense_tail` 为空 → MoE 之后的 residual/post-norm/dense MLP 全被塞进 `dense_head`，在 burst 之前执行 | DeepSeek / GPT | 数值必错，且不报错 |
| A4 | `dense_head` / `closure` / `dense_tail` 三者互斥且并集 = `lns`，完全靠 `in_closure` + `down[]` 两段逻辑「顺便」保证，无任何显式校验 | Claude | 后续改代码易被静默破坏 |
| A5 | `last_suffixed` 传播停止条件可能漏掉最后一层「最后一个带 `-<il>` 后缀节点之后」的匿名尾部节点（如 MoE 后匿名 residual/add） | DeepSeek / Claude / GPT | capture 漏节点、tail 不完整 |

**对应建议**

- A1：不要只看 `nodes[0]`，应遍历 split，只跳过 alias，继续处理非 alias 节点。
- A2：`first_node` 取该层第一个**非 alias** 节点；debug 打印 `first_node / has_first`，当「layer 出现在 nodes 中但 `!has_first`」时报警。
- A3：`if (!out) { LOG_ERROR(...); return GGML_STATUS_FAILED; }`（至少在 whole-layer 模式）；不要静默清空 tail。
- A4：debug build 加一次性 assert：`head.size() + closure.size() + tail.size() == lns->size()` 且三者两两无交集。
- A5：单独检查最后一层的 `lns` 是否包含所有应有的残差、post-norm、dense MLP 节点。

### B. buffer / 执行语义（手工 clone 执行路径）

| # | 问题 | 来源 | 后果 |
|---|------|------|------|
| B1 ★ | `run_dense_nodes` 跳过 alias op，但 source clone 直接 `lf->data = src->data`，假设 view 的 data 已 materialize；若 `view->data == nullptr` 就把 NULL 带进去 | GPT / DeepSeek | 读空指针 / 静默错 |
| B2 | `run_dense_nodes` 没把 clone 之间建成依赖图，只依赖「A 已执行完，B 的 src->data 有结果」——要求 `nodes` 严格 topological order。这是一个很硬但未 assert 的 invariant | GPT / DeepSeek | stale data，诡异错误 |
| B3 | clone 用 `ggml_new_tensor_4d` 只拷 type/ne/nb/op/op_params，**没拷 `nd->flags`**（如 `GGML_TENSOR_FLAG_OUTPUT`） | Claude | 后续扩展易复发 |
| B4 | `dense_host` 判定只看 source buffer，不看 `nd->buffer`，隐含假设 scheduler 会给 nd 分配正确 host output buffer | GPT / DeepSeek | 依赖隐式行为，多 backend 下不稳 |
| B5 ★ | `moe_dev_supports_buft` 在 `STREAM_MOE_TEMP` 下无条件 `return true`，**没跟 `no_whole` 联动** | Claude / DeepSeek | ① 所有 debug build 调度行为被悄悄改变，`NO_WHOLE_LAYER=1` 对照组不干净；② 可能把设备 buffer 的 tensor 分给 CPU 执行 → 崩/读错 |

**对应建议**

- B1：dump 检查 VIEW 的 `data` 何时被设置；加 `assert(src->data)`；否则 `lf->data = src->data` 会把 NULL 带进去（GPT 认为这是**最值得实际 dump 检查的 bug candidate**）。
- B2：加 assert：若某 src 属于 `dense_nodes`，则它必须出现在更早位置。
- B3：clone 时补拷 `nd->flags`。
- B4：assignment 后断言/打印 `nd`、src buft、dst buft；`assert(nd->buffer == nullptr || buft_is_host(...))`。
- B5：把 override 挂到同一运行时开关，或用「`g_layer_nodes` 是否非空」动态判断；只在明确验证过的 host/CPU 场景开启；测试时检查实际分配的 buft。

### C. 调试脚手架 / 生命周期

| # | 问题 | 来源 | 后果 |
|---|------|------|------|
| C1 | `g_dbg_pos` 是裸全局静态指针，无生命周期管理，非 thread_local；只在「从未设置过」时赋值一次。跨请求/context 重建后可能悬空，`dbg_canary` 解引用 `g_dbg_pos->data` | Claude | UAF |
| C2 | `estimate_scratch` 的 per-node 预算是常数 16KB（`need += n_nodes*16*1024`），没按实际 tensor 大小算。multi-token prefill 下单 dense 节点 `n_embd * n_tokens * sizeof(type)` 很容易远超 16KB | Claude / DeepSeek | 大概率静默写坏内存（数值发散候选） |
| C3 | `g_layer_nodes_all` 在 `moe_chain_assign_backend` 里没 clear；若 `collect_layer_nodes` 内部不清，多次 build graph 会累积旧节点指针 | DeepSeek | 悬垂指针 / 重复执行 / dump 错乱 |
| C4 | `collect_layer_nodes` 实现未在 diff 中出现（可能链接失败）；且它是否复用修好的 `lay[]` 未知 | Claude / DeepSeek | output head 泄漏进最后一层的问题可能只在「层归属查询」层面修好，在「实际 capture」层面还漏 |

**对应建议**

- C1：每次新 decode/graph 开始时重置为 `nullptr`；改 `thread_local`。
- C2：先确认 ctx 是否 `no_alloc`；若会落地数据，按 `ggml_nbytes` 精确算预算。
- C3：确认 `collect_layer_nodes` 内部 `out.clear()`；单独测「连续 build 两次图」检查是否只保留当前图节点。
- C4：第一个去核对的地方——`collect_layer_nodes` 是否直接消费 producer-propagation 里修好的 `lay[]`，还是另一套独立闭包逻辑。

### D. 静默跳过 / 错误处理

| # | 问题 | 来源 |
|---|------|------|
| D1 | 未捕获 `MUL_MAT_ID` 的报错条件收窄：只有 `src[0]` 名含 `_exps` 才 FAILED，否则静默 `continue` | DeepSeek |

**建议**：至少对所有「未归属层的 `MUL_MAT_ID`」报错或打 warning，不要静默跳过。

### E. 归属 heuristic（脆弱点）

| # | 问题 | 来源 |
|---|------|------|
| E1 | layer attribution 是 heuristic（name 后缀 `-<il>` + producer propagation）；跨层匿名节点（producers = {L3, L4}）会被默默选一个 | GPT |
| E2 | `ffn_moe_out` 用 `strstr` substring 匹配，是脆弱 anchor；换模型/上游改名即失效且不报错 | GPT / Claude |
| E3 | `is_view_op()` 与 `is_alias_op()` 语义重叠（`is_view_op(CONT)==true` 但 `is_alias_op(CONT)==false`），未来误用会把 CONT 当 alias | GPT |
| E4 | 「每层所有节点会被切进同一连续 split」是 whole-layer 正确性依赖的隐式假设，但 split 由 scheduler 启发式决定，无人保证 | Claude / GPT |

**对应建议**

- E1：归属做成「验证式」——跨层匿名节点直接报错，不 silent recovery。
- E2：从已有 MoE closure（`moe_layer_exec_t` 的 compute 集合）反推 boundary，而不是从 tensor name 猜。
- E3：正式三分类：`PURE_ALIAS`（VIEW/RESHAPE/TRANSPOSE/PERMUTE）/ `MATERIALIZING`（CONT/DUP/CAST）/ `COMPUTE`，语义不重叠。
- E4：触发与 split 解耦（见架构建议）。

---

## 二、调试手段建议

### 2.1 分阶段开关（分段启用）

现有开关：
- 编译期 `STREAM_MOE_TEMP`：不带 = 生产路径（先确认基线没退化）。
- 运行时 `STREAM_MOE_TMP_NO_WHOLE_LAYER=1`：debug build 但关闭 whole-layer（capture-only）。

建议新增（GPT #10 / Claude #4）：

```
STREAM_MOE_L2_CAPTURE=1        // 只 capture
STREAM_MOE_L2_OWNERSHIP=1      // 只改 ownership，不执行 dense
STREAM_MOE_L2_EXECUTION=1      // 真正执行
STREAM_MOE_L2_DENSE_HEAD=1     // 只跑 head
STREAM_MOE_L2_DENSE_TAIL=1     // 只跑 tail
STREAM_MOE_TMP_WHOLE_LAYER_MAX=N   // 只对 layer <= N 生效（逐层 bisect）
STREAM_MOE_TMP_WHOLE_LAYER=<L>     // 只开单层 L，其余走 baseline
```

### 2.2 推荐测试顺序

```
                    ┌─ NO_WHOLE_LAYER=1 ── baseline
graph build ─ capture ┤
                    └─ whole layer
                          ├─ 不执行 dense（run_dense_nodes 只打印）
                          ├─ 只跑 dense head
                          ├─ head + MoE
                          └─ head + MoE + tail
                                ↓
                             逐层 L0 → L1 → L2 → ...
```

- **Phase 0（最重要）**：只测 capture，不改 execution。用 `NO_WHOLE_LAYER=1`，确认：每层节点数稳定、层间不串、`result_norm/result_output` 没被归进最后一层、匿名 `node_*` 正确归属、`ffn_moe_out` 在正确层、embedding/final output 没被误吞。
- **Phase 1**：开 ownership，但把 `run_dense_nodes()` 换成只打印不执行。
- **Phase 2**：只跑 dense head，做逐节点 A/B。
- **Phase 3**：再开 dense tail，做最终 token-level compare。
- 再逐层打开（L0-only 最有用，因为问题是「某一层输出开始 divergence」而非「整个模型对不对」）。

### 2.3 逐节点 A/B 数值对比

- 利用已有 debug 输出：`[node]` `[dnode]` `[canary]` `[cur]` `[embd]` `[ext]`。
- 用同一 tiny fixture，`NO_WHOLE_LAYER=1` vs whole-layer 两次跑，对比每层每节点数值；重点看 `norm / cur / ffn_moe_in / ffn_moe_out / residual / post_norm`。**不需要先看最终 logits**。
- 正式化成 **whole-layer A/B verifier**：对每节点输出做 hash/compare，直接输出：

```
L17 norm        SAME
L17 Qcur        SAME
L17 attn        SAME
L17 ffn_norm    SAME
L17 moe_in      SAME
L17 moe_out     SAME
L17 residual    DIFFER   ← 直接定位是 residual 被改坏
```

- 固定 fixture：`--chunks 1` + 单 token decode 快速迭代，跑通再上 prefill（注意 C2 的 scratch 预算问题）。

### 2.4 加 invariant / assert（GPT 列了 7 条，认为 1/3/5/6 优先）

1. alias/view 的 `data` 是否已 materialize（**优先**）。
2. dense nodes 是否严格 topological。
3. output anchor `ffn_moe_out` 是否一定存在（**优先**）。
4. anonymous node 是否可能跨 layer。
5. `nd->buffer` 是否真的 host（**优先**）。
6. `first_node` 是否一定出现在负责该 layer 的 split（**优先**）。
7. whole-layer capture 的全局 map 生命周期是否与 scheduler execution 完全一致。

### 2.5 其他

- 打印/校验：实际 buffer type、每个 split 的 `nodes[]`、`first_node`、`has_first`、`[exec] ... layers=` 行（一次调用处理多层时留意）。
- 内存/dump：跑多个 ubatch，确认 `g_dump_scratch` 不越界、`g_layer_nodes_all` 不累积、`g_ubatch>3` 退出逻辑不误触发。
- 单元测试：`is_alias_op`（CONT 为 false）、`run_dense_nodes` 小图（含 VIEW/CONT/MUL_MAT_ID，对比原图直接执行）、`moe_exec_mul_mat_id`（`nodes[0]` 是 alias、后面跟非 alias 的 split）。
- 层划分测试：故意把 `ffn_moe_out` 改名，验证是否报错/fallback；检查最后一层 `lns` 完整性。
- **调试工具与核心控制流分离**（Claude #4 / GPT #16）：`run_dense_nodes` 只负责执行，`dump_dense_nodes` 单独做；把 tracer 做成可选注入接口（不传即 no-op），别让 canary/dump/fprintf 焊死在热路径。

### 2.6 三方最优先排查清单

- **GPT**：① alias/view 的 data 是否 materialize；② `ffn_moe_out` 是否一定存在；③ `nd->buffer` 是否真 host；④ `first_node` 是否一定在该 layer 的 split。优先怀疑手工 clone 后 activation 的 buffer/data/view 语义变化。
- **DeepSeek**：① `moe_exec_mul_mat_id` 开头 `is_alias_op(nodes[0]) return`；② `first_node` 取 `ln->front()` 可能是 alias；③ `ffn_moe_out` 找不到 → `dense_tail` 为空。

---

## 三、架构 / 设计建议（「如果我来实现」）

三方共同主题：**把「抓节点 + 改 scheduler ownership + 手工执行节点」三件事彻底解耦，让 layer 成为真正的执行单位，而不是 scheduler split 的副产品。**

1. **LayerPlan 前置化**（GPT / Claude）
   在 graph-build 阶段一次性算出结构，executor 不再重新推导「谁属于 layer / 谁是 first / 谁是 tail」：

   ```cpp
   struct moe_layer_plan {
       int32_t layer;
       std::vector<ggml_tensor*> all;
       std::vector<ggml_tensor*> head;
       std::vector<ggml_tensor*> moe;
       std::vector<ggml_tensor*> tail;
       ggml_tensor* input;
       ggml_tensor* moe_out;
       ggml_tensor* output;
       bool host_owned;
   };
   ```

   于是 `has_first` / `first_node` / `layers.push_back(...)` 这套 heuristic 全部删掉。

2. **layer 是 execution unit，不是 scheduler split 的副产品**（GPT / Claude）
   `graph build → LayerPlan → scheduler ownership → executor → execute(LayerPlan[0..N])`。
   scheduler 只回答「这些 node 属于 StreamMoE backend」，不该决定「这个 layer 是否已经执行」。

3. **层边界用显式 side-channel 标注，不用字符串反推**（Claude）
   在 `llm_build_context` 构建每层/每个 MoE 子图时，主动注册 `layer_span_t {begin, moe_begin, moe_end, end}`；
   这样 `moe_chain_layer_nodes` / `layer_of_node` / `last_suffixed` 全可删，正确性不依赖模型架构或上游命名约定。

4. **boundary 从已有 MoE closure 反推，而不是从 tensor name 猜**（GPT）
   MoE closure 已经是可靠边界；从 `moe_layer_exec_t` 的 compute 集合反推 head/tail。

5. **op 正式三分类**（GPT）：`PURE_ALIAS`（不 compute，但须保证 data/layout 已正确）/ `MATERIALIZING`（必须执行）/ `COMPUTE`（必须执行）。

6. **尽量不 clone dense graph**（GPT / Claude）
   - 首选：保留原始 `nd` 节点本身，只改变 ownership（`execute_node_range(ctx, backend, plan.head)` 执行原 node）。
   - 如果 llama.cpp scheduler API 不允许，宁可在 scheduler/backend 层增加正式的「this backend owns this node range」机制，也不长期维护 clone graph。
   - 若不得不手工执行：clone「graph structure」（exec graph 的节点指针指向原 `ggml_tensor`），而不是 `ggml_dup_tensor` + 手工填 `data/buffer/view`——后者极易漏掉 layout/buffer/view invariant。

7. **layer output 作为显式 synchronization boundary**（GPT）
   ```
   HEAD → materialize MoE input → pin experts → MOE BURST
        → materialize MoE output → TAIL → LAYER COMPLETE → 下一层
   ```
   只有 `LAYER COMPLETE` 后 Layer N+1 才允许读。layer 正是 cache/expert lifetime + compute + activation lifetime 的自然边界（与 StreamMoE 思路吻合）。

8. **LayerExecutionState 保证一次且仅一次执行**（GPT）
   `{NOT_STARTED, RUNNING, COMPLETE}`；`COMPLETE` 直接 return，`RUNNING` 可 assert「重复并发层执行」。
   correctness 不再依赖「first_node 是否恰好进入这个 split」。

9. **归属「验证式」而非「猜测式」**（GPT）：允许 heuristic，但最后验证匿名节点的所有 layer-attributed producers；出现 `{L3, L4}` 直接报错。

10. **ownership 与 execution 拆成独立 feature flag**（GPT）：见 §2.1。

11. **host-resident 判定升级为 per-layer `LayerCapability`**（GPT）
    ```
    Layer 0: dense=HOST, moe=STREAM_MOE, activation=HOST
    Layer 1: dense=GPU,  moe=STREAM_MOE, activation=HOST
    ```
    不要散落的 `dense_host` boolean；为后续 P100 + RX590 + CPU 多 backend 准备。

12. **触发时机与 split 解耦**（Claude）：做成图构建阶段就确定的显式回调，而不是「从 MUL_MAT_ID 执行入口顺便判断」的反应式方式。

---

## 五、实测定位（2026-09-13）

### 5.1 方法

加逐层开关（`src/backend/route_b_chain.cpp`，`STREAM_MOE_TEMP` 门控）：

```
STREAM_MOE_TMP_WHOLE_LAYER=L     只对第 L 层启用 whole-layer
STREAM_MOE_TMP_WHOLE_LAYER_MAX=N 只对 layer <= N 启用
```

对 olmoe（`-st -p hi -n 24 --temp 0`，greedy）逐层 A/B。

### 5.2 结果

| 配置 | 生成结果 |
|---|---|
| baseline（`NO_WHOLE_LAYER=1`） | `Hello! How can I help you today? ...`（正确） |
| L0–L14 逐个单独接管 | 全部 **SAME** |
| **L15 单独接管** | **DIFF（乱码）** |
| `WHOLE_LAYER_MAX=14`（L0–L14 接管，L15 保持 MoE-only） | **正确** |
| `WHOLE_LAYER_MAX=15`（含最后一层） | 乱码 |

**结论：发散 100% 来自接管最后一层。**

### 5.3 根因

`olmoe.cpp` 只在最后一层插入输出 token 选择：

```cpp
if (il == n_layer - 1 && inp_out_ids) {
    cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);   // whole 模式里 = node_978
    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
}
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
```

- 该 `GET_ROWS` 被 `collect_layer_nodes` 归入最后一层；
- 又被 `run_dense_nodes` 在 dense head 里执行；
- 这就是代码注释里说的 "output-token reduction path diverges"。
- `last_suffixed` guard 挡不住它，因为它出现在**最后一个带后缀节点之前**（`ffn_inp` 之前）。

### 5.4 次要发现

整层接管会**顺带把 attention 从 `FLASH_ATTN_EXT` 全局降级为手动 `kq/kqv`**（baseline L0 = `node_25 FLASH_ATTN_EXT`，whole L0 = `kq/kqv/kqv_out`）。这是 `supports_buft`/ownership 影响图构造的旁证。L0–L14 数值不受影响，**不是**本次发散主因，但需单独确认原因。

---

## 六、Bug 分类

### 6.1 直接就能改（独立、小、风险低）

| id | 问题 | 改法 | 是否动生产路径 |
|---|---|---|---|
| A1 | `is_alias_op(nodes[0])` 直接 return | 遍历 split，只跳 alias | 是（`moe_exec_mul_mat_id`） |
| A2 | `first_node=ln->front()` 可能是 alias | 取首个非 alias 节点 | 是 |
| A3 | `ffn_moe_out` 找不到静默清空 tail | `if(!out) return FAILED` | 否（dbg 才有 lns） |
| A4 | head/closure/tail 互斥无校验 | ~~assert 三者并集=lns~~ **评审前提错**：closure 不总是 layer list 的子集（gemma 实测 closure 有节点不在 lns），只保留诊断打印 | 否 |
| B1 | `lf->data=src->data` 不查空 | assert `src->data` | 否 |
| B2 | clone 依赖严格拓扑序无 assert | assert src 若属本层须更早 | 否 |
| B3 | clone 丢 `nd->flags` | 补拷 flags | 否 |
| B4 | `dense_host` 只看 src buffer | 加 `nd->buffer` 检查 | 否 |
| B5 | `supports_buft` 无条件 true | 挂到 `NO_WHOLE_LAYER` / 动态判断 | 否（`#ifdef STREAM_MOE_TEMP`） |
| C1 | `g_dbg_pos` 悬空/UAF | `thread_local` + 每次重置 | 否 |
| D1 | 未捕获 `MUL_MAT_ID` 静默跳过 | 全部报错/warning | 是 |
| E1 | 跨层匿名节点默默选一个 | 直接报错 | 否 |
| E3 | `is_view_op`/`is_alias_op` 语义重叠 | ~~重命名~~ **归 C**（三分类 PURE_ALIAS/MATERIALIZING/COMPUTE 会一并做，不做一次性改名） | - |
| NEW | DeepSeek 小池 `0xC0000005` 崩溃 | **归 C/D**（是"整图单 buffer"的症状，A 类补不了） | - |

> **落地状态（2026-09-13）**：A 类已完成并推送：
> - `2efbc98` route_b: per-layer whole-layer bisect switches
> - `2c2f1b9` minigraph: harden whole-layer dense execution（B1/B2/B3/C1 + A4 诊断）
> - `f000955` minigraph: whole-layer boundary robustness（A1/A2/A3）
> - `aa94519` route_b: whole-layer capture hardening（B4/E1）
> - `d97f1ec` moe_backend: gate whole-layer buft acceptance（B5）
>
> 验证：dbg build 下 olmoe（baseline/MAX=14 正确，MAX=15 已知乱码）+ gemma 正常；生产 `StreamMoE` 下 olmoe/gemma 正常。
> 评审说错的 C2/C3/C4 未改。E3 与 DeepSeek 崩溃归入 C/D。

### 6.2 评审说错（不用改）

- **C3**（`g_layer_nodes_all` 不 clear）—— 错，`collect_layer_nodes` 第一行 `out.clear()`（`route_b_chain.cpp:636`）。
- **C4**（`collect_layer_nodes` 没实现）—— 错，实现就在 `route_b_chain.cpp:634`。
- **C2**（`estimate_scratch` 常数 16KB 会溢出）—— 不适用：arena 是 `no_alloc`（`minigraph.h:48`），clone tensor 只占结构体，不分配数据，16KB/节点绰绰有余。评审的前提（ctx 可能非 no_alloc）不成立。

### 6.3 适合跟着重构，一次全没

只要做「**LayerPlan 前置化 + 逐层执行 + 显式边界 + 不 clone dense**」：

| id | 为什么自然消失 |
|---|---|
| A5 `last_suffixed` 漏尾部 | 边界显式标注后不需要 suffix/传播启发式 |
| E2 `ffn_moe_out` 名字匹配 | 边界从 MoE closure 反推，不靠名字 |
| E4 触发绑定 split | LayerPlan 是执行单位，不需要 `has_first` |
| A1/A2 | 没有 `first_node`/`has_first` 这套东西 |
| A3 | 边界显式，不存在「找不到 out」 |
| B1/B2/B3 | 不再 clone，直接用原 node 的 buffer/view/flags |
| C1/C2 | tracer 分离；自有 arena 后 scratch 预算问题消失 |
| **olmoe 根因**（最后一层 `inp_out_ids`） | 逐层 + 显式边界后，最后一层的 output reduction 会被正确处理 |
| **DeepSeek 根因**（整图单 backend buffer） | 每层自有 arena、不把 43 层塞进一个 compute buffer |

### 6.4 不会消失、且不好改（必须专门设计）

| 问题 | 为什么躲不掉 |
|---|---|
| **整图单 backend 的 compute buffer 分配** | 本质是 `ggml_backend_sched` 分配策略；节点都归一个 backend 就可能大。要么不全归我们 backend，要么 `no_alloc`+自有 arena——设计选择，不会自动好 |
| **最后一层 `inp_out_ids` 语义** | llama.cpp 图结构；route B 无论怎么重构都必须显式理解并处理「最后一层的 output-token reduction」 |
| **`supports_buft`/`supports_op` 与 scheduler ownership 的耦合** | 用自定义 backend 接管就必须正确声明能力；「怎么声明」是长期设计问题 |
| **dense 执行顺序 / data 就绪的本质约束** | 即使不 clone，「dense 必须在 burst 前 materialize、严格拓扑序」仍是硬约束 |
| **debug 脚手架焊在热路径** | `g_dbg_pos`/dump/fprintf 内嵌在 `run_dense_nodes`/`exec_layer_burst`，需主动抽 tracer |
| **DeepSeek 分配失败的崩溃而非干净报错** | 错误处理问题，需主动补 guard |

### 6.5 执行顺序建议

1. A 类低风险/纯 debug 加固（A4/B1/B2/B3/C1/C2）→ 编译 dbg + 冒烟；
2. A 类 whole-layer 边界健壮性（A1/A2/A3/B4/D1/E1）→ 编译 dbg + olmoe A/B；
3. B5 + DeepSeek 崩溃 guard → 编译 + DeepSeek L2 测试；
4. E3 纯改名 → 编译 + baseline regression；
5. C/D 一起讨论，重构时一并解决。

---

## 七、一句话结论

现有实现的问题不是「哪个函数写错了」，而是**整体依赖了太多「从图的表面特征（名字、split 切分方式）反推结构」的隐式假设**。
三方建议收敛到同一句话：**把这些假设往「图构建阶段的显式契约（LayerPlan / layer_span / 三分类 op / assert）」上收敛**，
并**尽量复用原始 ggml node 语义、不要自建第二套 clone graph 执行语义**。
这样既能定位当前的 output-token reduction 发散，也能让分段启用与后续 layer-aware 专家缓存 / CPU-GPU 混合自然变简单。

# 满宽 / 单设备桶引擎快路径 (BUCKET_FAST_PATH)

[English](BUCKET_FAST_PATH.md) | [简体中文](BUCKET_FAST_PATH.zh-CN.md)

> 状态：设计讨论（2026-09-09），**未实现**。对应 `docs/WORK_IN_PROGRESS.md` P1。
> 目标：在满宽 / 单设备场景下，省掉唯一执行器 `exec_layer_burst_chain_buckets`
> （`src/backend/minigraph_exec.cpp`）里恒等与单目标归约带来的固定开销。
> 本文记录讨论结论、精确判据、转置（CONT）分析、风险与验证口径。

## 1. 背景

- 唯一执行器 = 紧凑桶引擎；round 源 = `build_mix_plan` 的真 token 子集 round。
- 桶引擎的通用形态是"按 round 重建紧凑链 `[d, w_b, n_active]` → 折专家轴 → scatter-add 进 `acc_d`"。
- 当 round 退化/接近满宽时，链里出现大量**恒等置换**与**单目标归约**，纯属开销：
  - `GET_ROWS(cur)` / `GET_ROWS(权重)` 在满宽时是恒等；
  - 单设备时 `ACC` + `host layer_fold` 只是拷贝/零加。
- 注意：**桶只在多 pool 时才产生**。单 pool（全部专家在同一个池）恒为**单桶全宽**
  （每 token 命中该池 n_k 次，hit_count 恒定）；多 pool 时某个 pool（设备）上不同 token
  命中数不同，才会切出多段（详见 §3）。

## 2. 省略条件（精确到代码）

> 判据对象是**当前桶（round）**，与它属于哪个 pool 无关；§3 的 peel 按 pool 计算只是为了
> 说明桶列表怎么来的。

| 省略项 | 条件（每 round / 每 plan） | 挂点 | 理由 |
| :--- | :--- | :--- | :--- |
| `GET_ROWS(cur)` | 每 round `r.n_active == n_t` | `bucket_gather_cur` (minigraph_exec.cpp:665) | `active` 按 t 升序构建（mix_split.cpp:113），`build_scatter_plan` 对全 token 产出单位 `order`（scatter_plan.cpp:84）；与 width 无关 |
| `GET_ROWS(权重)` | 每 round `r.n_active == n_t && r.width == n_k` | `bucket_gather_per_slot` (minigraph_exec.cpp:631) | 必须整个 `(t,k)` 单元都在本 round，目标布局才与源 `[1,n_k,n_t]` 一致 |
| `ACC` 拷贝化 | `single_target && 只有 1 个 round` | acc 循环 (minigraph_exec.cpp:1300) | 多 round 时 ACC 是真累加；`width==n_k && n_active==n_t` 蕴含单 round |
| `host layer_fold` | `single_target`（任意 round 数） | `layer_fold` (minigraph_exec.cpp:465) | 单 target 时 host 侧只剩"把该 target 的 acc 落到 moe_out" |

- `single_target` = `rounds` 里出现的**不同 pool 数 == 1**（不是 `dev_targets.size()<=1`：
  CPU round 与 device round 并存时 target 数是 2）。
- 两类省略**独立**：满宽管展宽 gather；单 target 管归约合并。单 pool 恒为单桶全宽，
  两者同时满足；多 pool 时按每桶独立判。
- `cell_full`（`width==n_k && n_active==n_t`）⇒ 全部 token 命中该 pool 且命中满 n_k
  （即单 pool / 该层等价单池）⇒ 单桶；`tok_full`（`n_active==n_t`）**不**蕴含单桶
  （多 pool 时该 pool 可 token 满而 width<n_k）。

### 2.1 `weights` 是什么（为什么它要 gather）

`build_moe_ffn`（vendored `llama-graph.cpp`）里的 `weights` 是**路由权重（门控概率）**，
不是模型专家权重矩阵：

- gate logits → softmax/sigmoid → `probs` `[n_expert, n_tokens]`（:1991-2008）；
- `selected_experts = argsort_top_k(...)` `[n_k, n_t]`（:2057-2060）= 执行器里的 `ids`；
- `weights = get_rows(probs, selected_experts)` `[1, n_k, n_t]`（:2071）；
- `norm_w` 时 reshape `[n_k,n_t]` → `sum_rows` → clamp → `weights /= weights_sum`
  = `ffn_moe_weights_norm`（:2082-2095），可选 `w_scale` = `_scaled`（:2097-2100）；
- 消费：`experts = experts * weights`（down mm 后，:2267-2270；或 `weight_before_ffn`
  时乘输入 cur，:2107-2111），匿名 fold 再把 n_k 个专家槽相加 → `moe_out`。

所以 `bucket_gather_per_slot` 处理的是 `[1,n_k,n_t]` 的 `ffn_moe_weights_norm`/`_scaled`
外部叶子：compact 桶的逐槽乘法要按桶的 `(t,k)` 子集、按桶 tight 顺序取出（仿射 k 切片
对任意子集不成立，故用 flat index-gather）。满宽时整个 `(t,k)` 单元都在 → 恒等 → 直接
引用源张量。模型专家矩阵（gate_up/down）是池里的 w3d 壳子，与 `weights` 无关。

## 3. 为什么 cur 与 weights 的条件不同

一句话：**桶是一个确定的矩形 `[width, n_active]`，问题是这个矩形是不是源张量的全网格。**
源 `ffn_moe_weights_norm` 是满的 `[1, n_k, n_t]`（连续，flat 下标 `t*n_k+k`）；桶要
`[1, width, n_active]`（flat `a*width+s`）。只有覆盖整个网格才恒等。

| 量 | 直接引用（跳过 gather）条件 | gather 形态 |
| :--- | :--- | :--- |
| `cur` | `n_active == n_t` | 按 token 取整行（行内连续 d 个 float） |
| `weights` | `n_active == n_t && width == n_k` | 按 `(t,k)` 取单元素（flat 下标） |

三种组合：

- `n_active==n_t && width==n_k`：cur 免、weights 免（单 pool 单桶）；
- `n_active==n_t && width<n_k`：cur 免、weights 仍 gather（多 pool 中某 pool 的第一段）；
- `n_active<n_t`（无论 width）：cur、weights 都 gather。

`cur` 只受 token 轴制约（源每 token 一行、与槽无关）；`weights` 受 token 与槽双重制约。

peel 机制（mix_split.cpp:21-35, 108-135）：每 pool 按 hit_count 分桶，`counts` = 非零命中数的
升序去重值；round j 的 `width = v[j]-v[j-1]`（`v[0]=0`），`active = {t : hit_count[t] >= v[j]}`。
两个方向都会解耦：

- **Case A：`n_active==n_t` 但 `width<n_k`**
  该 pool 覆盖全部 token，但 token 命中数不同。
  例：两 pool、n_k=3，某设备 pool 持有部分专家，其 counts=2/3/3 → round1 `width=2`、`active=全部`。
  → cur 恒等（token 轴全）；weights **不**恒等（每 token 只来了 2 个 k 槽）。
- **Case B：`width==n_k` 但 `n_active<n_t`**
  该 pool 里"命中它的 token 都命中满 n_k 次"（即这些 token 的专家全在该 pool），
  但 0 命中的 token 被排除。
  例：两 pool、n_k=2；token A 两个专家都在 Vulkan0、token B 都在 RAM
  → Vulkan0 round1 `width=2=n_k`、`active={A}`。
  → cur / weights 都**不**恒等。
- **Case C：两者同时成立（`cell_full`）**
  全部 token 命中该 pool 且命中满 n_k 次 → 单 pool 单 round → cur 与 weights 都恒等。

推论：`width==n_k` 不蕴含 `n_active==n_t`（Case B）；`n_active==n_t` 不蕴含
`width==n_k`（Case A）。cur 的省略只需 `n_active==n_t`；weights 的省略必须两者都满足。

## 4. 转置（CONT）问题

### 4.1 位置与根因

`append_expert_fold`（minigraph_exec.cpp:608-624）：

```
weighted [d_out, w_b, n_active]
  -> permute(1,0,2,3)            // 视图 [w_b, d_out, n_active]，新 nb0 = d_out*4
  -> ggml_cont                   // :615 真正搬数据（转置）
  -> sum_rows                    // 归约新 ne0 (= w_b) -> [1, d_out, n_active]
  -> ggml_cont_2d                // :619 -> [d_out, n_active]
```

根因：ggml 的 `sum_rows` **只归约连续的 ne0**。

- CPU（`ggml-cpu/ops.cpp:1470-1486`）：`GGML_ASSERT(src0->nb[0] == sizeof(float))`，
  `ggml_vec_sum_f32(ne00, ...)` 按单位步长读。
- Vulkan（`vulkan-shaders/sum_rows.comp:27-33`）：`src_idx = i01*nb01 + i02*nb02 + i03*nb03`，
  求和 `data_a[src_idx + i]`（`i < n_cols = src0->ne[0]`）；push constants 里**没有 nb00**
  （ggml-vulkan.cpp:1972-1998 只有 nb01..03）。

`permute(1,0,2,3)` 后视图的 `nb0 = d_out*4`，两个后端都不能直接吃 → `ggml_cont` 是硬性要求，
不是保守写法。

### 4.2 代价量级

转置张量 = `w_b * d_out * n_active * 4`：

| 场景 | 大小 | 说明 |
| :--- | ---: | :--- |
| prefill 3k（w_b=8, d=2048） | ~196 MB/层 | cont 读+写 ~392MB，sum_rows 再读 196MB → ~600MB/层 |
| decode（n_active=1） | ~64 KB | 可忽略 |

P1b 实测 2-token 层 `CONT 986us` **不可能是带宽**（数据才几十 KB），只能是固定
dispatch/图开销或 perf logger 归因——印证 P2"瓶颈是每层 submit+sync"。

### 4.3 能否省掉

1. **换 per-k view + add 链**（匿名 fold 的形状）：
   `acc = view_2d(weighted, d_out, n_active, weighted->nb[2], 0);`
   `for w: acc = add(acc, view_2d(..., w*weighted->nb[1]))`
   - 不转置、无 196MB 临时、对桶通用（不需要原始 k 映射）；代价 `w_b-1` 个 add 核。
   - 数值是另一种求和顺序（宽松 gate 内）。
2. **满宽时复用 llama 原始匿名 fold**（闭包重放）：零重建，但 `bucket_src_leaf`
   需支持通用 view 形状（现只切 d）——即另一条路径，改动面大。
3. **保留现状**：prefill 的 MoE mm FLOPs 占绝对大头，转置很可能不是瓶颈。

### 4.4 顺手两个点

- `ggml_cont_2d(s, ne0, nt)`（:619）在 sum_rows 输出已连续时是多余拷贝，
  改 `ggml_reshape_2d` 即可（prefill 省 24MB/层）。
- **待确认**：fold scratch（`pc`/`s`/`acc`）走 `bind_fresh(..., false)` 的 device bump，
  但 arena 大小在 round 循环**之前**按 `arena_used(=base+acc) + 32MB` 定
  （minigraph_exec.cpp:1139-1143），build 期 bump 增长不计入；`device_ensure` 循环前只调一次、
  `ensure_buffer` 只在 `need>cap` 时重分配（moe_backend.cpp:424-437）。prefill 单层 `pc`
  就 196MB，远超 32MB slack → device arena 可能写越界。**未复现，P1 需顺手确认。**

## 5. 落地挂点

- 每 round 计算 `tok_full` / `cell_full`，写进 `bucket_build_t`；
  `bucket_gather_cur` 查 `tok_full`、`bucket_gather_per_slot` 查 `cell_full`。
- 满宽早返回：CPU 直接引用 `m->data`+`m->nb`（零拷贝）；device 走 `bucket_upload_leaf`
  （去掉 GET_ROWS kernel，只留一次上传）。
- **device 上传前提**：`tensor_write_host`（tensor_io.h:22）只做连续 memcpy `ggml_nbytes`，
  不认 strides。省略 gather 必须加"源紧凑"条件，否则 strided 源（argsort 层 ids 是 strided，
  weights_norm 待查）会静默传错 → 不满足回退原 gather。CPU 引用不看布局，永远安全。
- `single_target` 时跳过 acc/fold：CPU target 把最终 `per_token->data` 钉到 `moe_out->data`
  （CPU compute 只看 data，§7.4 已验证）；device target 把 `per_token` 留在 arena，sync 后
  一次 `tensor_get` 进 `moe_out`。数值等价（`layer_fold` 里 CPU acc 恒 0，`0+x` 位相同）。
- `gather_cache` 每 round 已 clear（:1279），满宽直接引用与后续子集 gather 不会串。

## 6. 风险与待确认

- `moe_out` 必须连续 `[d_out, n_t]`（`layer_fold` 已假设）；不连续则回退 fold。
- `STREAM_MOE_TMP_BUCKET_ROUNDS`（诊断强制分桶）必须仍走老路径——per-round 条件天然覆盖。
- B38 的 0-token no-op 在建 rounds 前已返回，不受影响。
- device 上传的源紧凑性（见 §5）。
- device arena slack（见 §4.4）。
- `layer_fold` 直接解引用 `moe_out->data`，与"铁律：勿直接解引用 tensor->data"存在张力，
  待确认 moe_out 是否恒为 host。

## 7. 验证口径

- 默认满宽路径：省略的是恒等置换/零运算，应对**当前 HEAD 干净构建逐字节 IDENTICAL**
  （同 flavor、同输入）。
- `STREAM_MOE_TMP_BUCKET_ROUNDS=1`：仍走老路径，宽松 gate（maxAbs ≤ 1e-5 / cos ≈ 1.0）。
- 保留工作区现有 TMR 诊断做前后对拍（设备图节点数、`chain_tail` / `tail_sync` 耗时）。

## 8. 落地任务（设计详见 §9）

- **P1-a** 条件发射：cur/weights gather 恒等跳过 + 连续优化。
- **P1-b** 单 target：ACC 直写 `moe_out`、跳过 host `layer_fold`。
- **P1-c** scratch：per-layer grow-only host scratch，灭执行器保活小 vector。
- **P1-d** `mix_plan` 并入 scratch（flat ids/scatter，rounds 变 span）。
- **P1-e** `twin`/`gather_cache` 换定长数组。
- **P1-f** 回归（默认单 pool 对 HEAD 干净构建 IDENTICAL + 强制分桶宽松 gate + TMR 对拍）。
- **P1b**（后置）CONT 量化 / per-k add 链。
- **P2**（后置）每层 submit+sync。

## 9. 统一路径 + 建图期条件发射（2026-09-09 定）

目标：**单一执行器不变**；单 pool 时靠建图期条件发射，把多余节点**不生成**（效果等价
于执行图优化），而不是另开一条路径。同时本次改动消灭所有保活的小 vector。

### 9.1 条件发射清单（per bucket）

| 节点 | 不生成条件 | 替代 |
| :--- | :--- | :--- |
| `cur` GET_ROWS | `n_active == n_t` | leaf 直接指源 |
| `weights` GET_ROWS | `cell_full` | leaf 直接指源 |
| `weights` GET_ROWS | 单 pool（k 集合统一连续） | k-slice 视图 + token gather |
| `ACC` | `single_target && 单桶` | `sum_rows` 输出直接钉 `moe_out` |
| host `layer_fold` | `single_target` | acc 直接绑输出 / 单次 D2H |

单 pool ⇒ 单桶全宽 ⇒ 图退化为"原始闭包 + 一次专家折叠"，几乎无多余节点。

### 9.2 scratch（灭小 vector）

固定 grow-only host scratch（每层一份，进程生命周期；rebuild 时按当前 `n_t`/`n_k` 定容，
`n_k` 恒定），bump 分配、每 round 偏移：

- i32 段：所有必须活到 `graph_compute` 的索引/ids（`ids_exp`/`ids_slot`/cur idx/weights idx/
  scale ids），替代 `mm_ids_pool`；总上界 `Σ ≤ ~4*n_k*n_t + n_t`，一次分配。
- f32 段：CPU fold 临时，替代 `fold_buf`；device 侧仍走 arena。
- `t_round`/`order`/局部 `idx` 改为 scratch 切片（指针+长度），不再 `push_back`。
- `twin`/`gather_cache`：闭包 ≤ ~20 节点/桶，可换定长小数组线性查，替掉 `unordered_map`。
- **归属/定容（2026-09-09 定）**：不是 per-layer；`thread_local exec_scratch_t`（每执行线程一块，
  单层串行内复用）。层内 **build 时写、之后只读**，跨层 reset + 覆盖。每层按当前 `n_t` / 各轮
  实际 round 算 `i32`/`f32` need，**一次 reserve**，exec 只 bump 不扩容（省 grow；预留后叶子
  data 指针不移动）。**实现坑**：`f32` need 必须含 gather 输出（CPU `bind_fresh(...,false)` 也走
  f32），漏了会在 `sum_rows` 写越界。
- **mix_plan 存储（2026-09-09 定）**：flat `ids`（i32）+ `scatter`（`mix_scatter_t`），
  用量按实际 `Σ width*n_active`；`mix_round_t` 换 `{pool,width,n_active,ids_off,scatter_off}`，
  rounds 用**定长 span 数组**（cap `n_pools*n_k`）。

### 9.3 唯一"不平凡"的节点：fold 的 CONT（P1 保留）

单 pool 时 fold 仍要做（专家轴求和），不是恒等/零；`permute+cont+sum_rows` 里的 CONT 是
`sum_rows` 只吃连续 ne0 的产物，不是数学必需。**决定（2026-09-09）：P1 保留 CONT**，
P1b 单独量化后再决定是否换 per-k view+add 链（`w_b-1` 个 add，无转置；decode 小张量下
add 链节点数可能更亏）。

### 9.4 决定（2026-09-09）

- **fold 不动**：P1 只做条件发射，CONT 保留。
- **`mix_round_t::ids/scatter` 并入 scratch**：`build_mix_plan` 写 flat ids/scatter 到
  scratch，rounds 变 span；同步改 `test_mix_plan`。

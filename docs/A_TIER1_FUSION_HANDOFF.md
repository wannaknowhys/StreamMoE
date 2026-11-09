# A 交接文档：把 B1 融合推广到多设备（c1c2 / Tier-1）

2026-11-06 · 起点提交 `aa12290`（main）· 2026-11-08 回填核实

> 本文是对话口径的汇总；**A 的范围以 `docs/DEVICE_FOLD_HEAD_FUSION.md`（zh-CN 同）§7 为准**。
> 标「未核实」的条目要先验证再信。代码行号可能已漂移，以实际代码为准。

## 0. 开工前先做什么

- 先重读 `docs/CHECKPOINT.md` 和 `docs/DEVICE_FOLD_HEAD_FUSION.md`（zh-CN 版同），尤其是 §7，再读代码。
- 本文件在仓库内，随代码一起维护；在线版不保证同步。

## 1. 现状（截至 `aa12290`）

**B1 已落地。** 单设备（Tier-2）把下一层 head(L+1) 并进 closure+tail 的设备图，不再有 `STREAM_MOE_LATEST` gate，production 默认开。`place-c1c2-exp5` decode 当时测得 11.45 → 13.02 tg/s（+13.7%）。

**B1 开/关的文本差异已有结论，不要再往下追：** 差异等价于 `ffn_norm-13` 输入上一份约 2.4e-7 的差，同一个 MoE 把它放大到约 1e-3。放大机制未明，2.4e-7 的来源未定位，首个翻转 token 未测。文档按「不下执行器/ALU 结论」的口径写，延续这个口径。

**验证体系（places 的新判据）：**

- `baseline_regression\run_baseline.bat baseline_regression\baseline\moe_129_8192_vk`，约 20 分钟量级；第 7、8 步需要环境变量 `SM_OLMOE`，否则跳过。`aa12290` 上结果：cos PASS、IDENTICAL PASS、KL max 0.4028（阈值 4.5）、places 6 PASS / 6 DIFF（PASS：cpu、c1c2-exp5、c2-exp2、c2-exp5、exp2、exp5；DIFF：c1-exp2、c1、c1c2-exp1、c1c2-exp2、c1c2、c2）。
- `[7b/8]` 重复一致性 gate：`baseline_regression/tools/check_place_repeat.js`，4 个 engine（c1-exp2、c1c2-exp1、c1c2-exp2、c1c2-exp5），每个 3 次独立运行、前 2 轮、预热丢弃，输出校验（轮数、无 `[ERROR]`、目录新鲜）。
- 它做两件事：同配置重复跑不一致 = FAIL（退出码 1）；输出 hash（前两轮文本的 sha256 前 16 位，key 为 `flavor:engine`，存 `repeat_baseline.json`）变化或 `[NEW]` = WARN（退出码 2，`run_baseline` 只打 `[WARN]` 行，不置 FAIL）。接受新 hash 用 `--accept`。参数：`--flavor --engines --runs --turns --update-baseline --accept --temperature`。
- 当前基准 hash（`StreamMoE_dump`）：c1-exp2 `67a7d516…`、c1c2-exp1 `a4bef78c…`、c1c2-exp2 `52c93af2…`、c1c2-exp5 `39bea583…`。
- 局限：重复一致性抓非确定性，**抓不到确定性错误**（例如当初的 `ByrEEEE`），那类只靠 hash 变化告警发现；hash 只说明「变了」，不能判好坏；基准是本机（RX 590、当前驱动、当前 submodule）的快照。
- 相关提交：`6898ce8`（撤 gate + 新判据）、`2a3f96d`、`aa12290`（`--accept`、退出码 2、warning 数、4 engine 基准）。前置提交 `b14ee43`：共享 relay helper（`relay_scope_t`、`relay_src_in_set`、`select_relays`，Tier-1 的 lambda 已重构）。

## 2. A 是什么、代码在哪

- 两条路径：**Tier-1** `exec_fused_tail_and_head`（经 `topo_segment_execute` 做 tail+head，c1c2 用，closure 在 host）；**Tier-2** `exec_layer_burst_chain_buckets`（单设备，closure+tail 在 `t.gf`，B1 在这里加了 head(L+1)）。B1 只覆盖 Tier-2。**A = 把这种融合推广到多目标（c1c2，closure 在 host）。**
- 文件：`src/backend/route_b_chain.cpp`（`layout_arena`、`build_layer_plans`、`shell_unit`、xfer 布局）、`src/backend/minigraph_exec.cpp`（closure 执行约 3590 行；dbg dump 约 3358–3421 行）。
- 术语：tail(L) 是层内 `ffn_moe_out` 的前向闭包；head 是其余部分（去掉 MoE closure 本身）。常用张量名：`l_out-L`、`ffn_inp-L`、`ffn_norm-L`、`ids`，head 输入是 `l_out-(L-1)`。

## 3. 前置：先改 xfer 布局（只改布局）

**现状：** xfer shell 是 per-layer 两槽 + 跨层复用。文档把 alias 记为真实隐患，但它不是 B1 那次 bug 的根因。

**历史上同类问题出现过两次：**

- L+1 的 64KB `ffn_moe_weighted` shell 盖住 L 的 8KB `moe_out` shell，当时的补法是拆 tail/head 两个 slot（`route_b_chain.cpp` 里有注释）。
- 实验性的 `fused_units` 布局下，prefill 第一次 layout 是 `fused_units=0`，head 0 的 `inp_tokens` 和 head 1 的 `leaf_6` 落在同一个绝对地址（`0x51240`）。**已核实（2026-11-08）：`fused_units` 在整个 `src/` 中已不存在**，该实验未进 main。

**结论/设计方向：**

- 每次融合改了时间线就补一个手工 slot 启发式，不可持续。更根本的做法是给 shell 一个 liveness 区间（派发点 → 最后一个消费者），在 unit 时间线上做区间打包，类似 scratch 的 `pack_interval`。head L 与 head L+1 不冲突，是区间不相交推出来的，而不是靠 parity。
- **不要再叠「能融合走新序、不能走旧序」的谓词。** layout 里静态算的 `fused_units` 和运行时的 `can_fuse_head` 并存，本身就是失配源。按 unit 规划在不能融合时同样正确，应当无条件使用。

**待核实（已更正）：**

- `layout_arena`（`:225` 定义、`:1512` 调用）与 `build_layer_plans`（`:116` 定义、`:1323` 调用）的**源码顺序本来就是先 plan 后 layout**，不是「需要提前」。
- 真正没核实的是：**首次 prefill build 时这两处是否在同一次调用链里都被执行**，以及 **`layout_arena` 现在是否消费 plan（而不是只消费 `g_layer_exec`）**。第一步应先用 TEMP `fprintf` 在 `build_layer_plans` 和 `layout_arena` 各打印 `g_build_id` 和 plan 数，跑一次首轮 prefill 确认，不要先改代码。
- 读地址要小心：`ddata = off_xfer + xfer_off`；`[xfer-shell]` 里的 `off` 是 region 内偏移，不是绝对地址。

## 4. 验证方案

**改前基线（在 `aa12290` 未改动代码上）：**

- `run_baseline` 全 PASS（§1）。
- 逐张量字节快照：`pwsh -NoProfile -File tools/run_trace_engine.ps1 -Engine place-c1c2-exp5 -Runs 1 -UbRanges 6-6 -Bin 1 -Root temp/snap_a`（脚本已入库：`tools/run_trace_engine.ps1`；输出在 `temp/snap_a/run_01/ub0006/*.bin`；`temp/` 在 `.gitignore` 第 31 行）。`temp/snap_a/MANIFEST.txt` 记录 commit、命令行、ub。
- 同一份代码上取两份独立快照（`snap_a`/`snap_b`），Python 逐文件 sha256 对比，**48/48 逐字节一致**（只说明可复现，不说明灵敏）。

**快照缺口（已核实，2026-11-08）：**

- 48 = 16 层 × 3，分别是 `L*_closure_moe_ffn_moe_out-*`、`L*_head_attn_ffn_inp-*`、`L*_head_cur_ffn_norm-*`。**没有 `l_out`、没有 head 输入 `l_out-(L-1)`、没有 `ids`、没有 closure 读的 weights/residual。**
- 原因（`minigraph_exec.cpp:3403` 起）：`trace_stage` 里 `dump("in", find("l_out-"+(layer-1)))` 和 `dump("out", find("l_out-"+L))` 用的 `find` 在 `moe_chain_layer_nodes_all(L)` 里找不到跨层张量，所以 `in`/`out` 分支不写 BIN。补快照要改取法（从 `g_layer_nodes[L-1]` 或上一层 tail 列表取）。
- **prefill ub 目前取不到 stage dump**：`trace_stage` 第 3363 行 `if (g_trace_nt != 1) return;`，prefill 的 nt>1 直接返回。要补 prefill 快照，先放开这个条件（只对 dump 生效，不影响计数）。
- `ids` 有逐层文本行（`minigraph_exec.cpp` 约 3082 行的 `trace_layer` 路径），可以单独按文本对比，或给 `trace_stage` 加 `ids` 的 BIN 写盘。
- 两份快照一致只说明**可复现**，不说明**灵敏**。要做一个正控：故意改动某个张量（例如临时往 `ffn_norm-13` 加固定噪声），确认快照对比会报 diff。

**两道门，都要过：**

- dbg 的 BIN 逐张量哈希。注意 dump 读回会引入同步，可能掩盖 race，所以字节一致只证明「值没变」。
- production flavor 的重复一致性（4 engine 无 FAIL、无 WARN）+ `run_baseline` 全 PASS。

**判定规则：** consumer 读到的值变了才算发现；只是 shell 里未使用的区域不同不算。出现差异先当发现记录，**不要回退布局去追平**，因为去掉 alias 后可能暴露此前被掩盖的东西（也可能与 2.4e-7 的来源有关）。

**阶段 2（融合本身）：** 数值会合理地变，hash 的 WARN 将没有信息量。硬门改为重复一致性（FAIL）+ `run_baseline` 的 cos/KL；可以考虑把已推迟的 teacher-forcing top-1/KL 做掉。接受新 hash 用 `--accept`，并在提交信息里写明原因。性能用 `tools/run_bench.js --max-turns 2` 对比 c1c2 engine 前后的 decode tg。

## 5. 已知风险与假设

- copy skip：Tier-1 `run_incoming_copies`（`will_flush_same_dev ? dst_dev : ""`）与 Tier-2 incoming（`single_dev_name`）依赖一个假设：同一 backend 上的下一个 compute 会为拷贝排序，其完成会 fence 住 staging。**未验证。**
- 2.4e-7 的来源未知；布局改动若造成字节差异，要和它联系起来看。
- 稳定性证据偏薄：重复一致性 gate 目前是几次连续 PASS，没有 5 次以上的长跑。

## 6. 工具、约定与踩过的坑

- 构建：`.\build.bat llamalibs dual`；dbg：`.\build.bat llamalibs StreamMoE_dump_dbg`（flavor 名 `StreamMoE_dump_dbg`，TEMP 代码放在 `#ifdef STREAM_MOE_TEMP` 里，提交前删干净）。
- 快照：`tools/run_trace_engine.ps1`（已入库）；`temp/snap_*/MANIFEST.txt` 记 commit/命令行/ub。
- 纪律：`third_party/llama.cpp` 子模块永不 commit；`scripts/verify_convert_matrix.bat` 有一份与本线无关的未提交改动，别带进来；每步验证完 commit + push；LF→CRLF 警告无害。
- 坑：批处理在 `if (...)` 括号块里 `echo` 不能带括号；改 JS 模板字符串用 edit 工具，不要用 PowerShell `-replace`（反引号会被吃掉）；比较哈希必须同一个函数、同一输入范围（上次拿 MD5 去比 sha256）；看日志别只看 `-Last 3`（进程内 seed 会推进）；每个新 gate 都要有会 FAIL/WARN 的正控；汇报的结论要附同一口径的证据，没验证的写成「未核实」。

## 7. 建议顺序

1. 重读文档和代码；确认起点 commit，构建，复现 `run_baseline` 与快照。
2. 固化并补全改前基线：manifest（已写）、`l_out`/head 输入/`ids`/weights/residual、prefill ub、快照正控（A 的第一步，需改 dbg trace 代码）。
3. 阶段 1：unit + liveness 区间打包，无条件使用，只改布局；先验证「首次 prefill build 两处都在同一次调用链、`layout_arena` 消费 plan」；过两道门；提交。
4. 阶段 2：c1c2（Tier-1）融合；硬门见 §4；测 decode tg。
5. 更新文档（en / zh / CHECKPOINT）；有意的 hash 变化用 `--accept` 并写明原因；提交。

## 8. 交接时未核实的事项（2026-11-08 更新）

- ~~`temp/run_trace_engine.ps1` 是否在 git 里~~ → **不在**；现已复制到 `tools/run_trace_engine.ps1` 并入库（`$repo` 改为 `Split-Path $PSScriptRoot -Parent`）。快照 manifest 已写 `temp/snap_a/MANIFEST.txt`（不入库）。
- ~~快照是否包含 `l_out` / head 输入~~ → **不包含**（只有 `moe_out`/`ffn_inp`/`ffn_norm`；`ids`/weights/residual 也没有）。prefill ub **取不到**（`g_trace_nt != 1` 提前 return），原因见 §4。
- ~~`fused_units` 实验是否确认已不在 main~~ → **已确认**，整个 `src/` 无 `fused_units`。
- ~~`layout_arena` 与 `build_layer_plans` 的当前先后顺序~~ → 源码顺序是先 plan（`:1323`）后 layout（`:1512`）；**首次 prefill build 是否两者都执行、`layout_arena` 是否消费 plan 仍未核实**，这是阶段 1 第一步。
- copy skip 假设是否成立 → **仍未验证**，留到阶段 2 前。

## 9. 批注回答（2026-11-08）

**问题：阶段 1 是直接做无条件的 unit + liveness 打包，还是先只修 `fused_units=0` 的先后顺序再看？**

**回答：直接做无条件的 unit + liveness 区间打包。**

理由：

1. `fused_units` 已不在 main（整个 `src/` 无该字段），`fused_units=0` 是实验状态，不是活字段；单独修它在 main 上没有落点。
2. 源码顺序本来就是先 plan 后 layout；真正的未知是「首次 prefill build 时两者是否都执行、`layout_arena` 是否消费 plan」。这一步用 TEMP 打印 `g_build_id` + plan 数就能确认，属于阶段 1 的第一个子步骤，不是一条独立岔路。
3. 文档 §3 已定方向：布局无条件使用，不再叠「能融合走新序/不能走旧序」的谓词；中间先修顺序再看，等于写一个会被单元打包覆盖掉的过渡态。
4. 约束：阶段 1 第一步先验证调用链（不改代码），确认 plan 在 layout 前已非空且被消费，然后再做区间打包；否则区间是空的。

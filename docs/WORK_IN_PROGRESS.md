# Work In Progress - patch 体系手术 + 收尾

> **上一轮：全部完成（2026-09-03）**。patch 体系已对齐工作区（干净 apply 逐字节一致）、ASan 整合进 build.bat、文档同步。
> **本轮（2026-09-04）**：v2 块内张量对齐 + SoA pool 布局改造定案并落地（K1-K5）；批量 pin（L1-L6，纯 RAM 0.11s IDENTICAL）；**M 节设计定稿**（驱逐 + move 管线，docs/EXPERT_MOVE_PIPELINE.md）。v1 sections-v1 否决。

## 背景状态（已落地，commit 403a5d2）

- features 机制：build.bat 传 `-DSTREAM_MOE_FEATURES`，vendored 根 CMakeLists features 块全局 `add_compile_definitions` + `include_directories`（3 frag 目录，`../../patches/...` 两级）
- frag 全主仓库：route-b/common、prefill-export/common、prefill-export/include（vendored include/ 已清空）
- llama-common PUBLIC frag include 已撤（保留 STREAM_MOE_SRC）
- server-context.cpp 纯锚点态（3 include 锚点：route-b spec×2 + prefill nout×1，全归 phase1）
- vendored 快照：temp/patch_backup_20260903/working-tree-full*.patch

## 任务清单

### A. patch 手术（patch 对齐工作区）

- [x] A0 盘点：patch 落后清单已确认（macros 缺根 CMakeLists/server-context 锚点；route-b/prefill 含过时 frag new-file + route-b 含旧 server-context 段）
- [x] A1 macros patch 重生成：`git diff HEAD -- CMakeLists.txt common/arg.cpp common/common.cpp common/common.h include/llama.h tools/server/server-context.cpp`（纯锚点/机制，无污染）
- [x] A2 route-b patch 重生成：`git diff HEAD -- common/CMakeLists.txt common/speculative.cpp common/speculative.h src/llama-model-loader.cpp src/llama-model-loader.h src/llama-model.cpp src/llama.cpp`（frag + server-context 自动消失；llama-model-loader.h 补入）
- [x] A3 prefill patch 重生成：`git diff HEAD -- src/llama-context.cpp src/llama-context.h src/llama-kv-cache.cpp src/llama-kv-cache.h tools/server/server.cpp`
- [x] A4 干净 apply 验证：临时 worktree HEAD(f280b2698) → 按序 apply（macros → tsc_timer → route-b → gguf-alignment → prefill）→ **21 文件 hash 与工作区逐字节一致**
- [x] A5 提交 patches + push
- [x] A6 清理残留 spec-stats.patch（已并入 route-b）
- [x] A7 apply.bat —— 暂缓（update_routeb_patch.js 已能文件级更新；干净 apply 已由 A4 验证）

### B. ASan 子命令整合 build.bat

- [x] B1 build.bat 加 asan 子命令（已验证构建成功 + dll copy）
- [x] B2 ASAN_BUILD.md 修过时（build.bat asan + features 说明）

### C. 文档同步

- [x] C1 patches/README.md 重写新体系（frag 主仓库 + features + apply 顺序 + 文件归属）
- [x] C2 docs/CHECKPOINT.md 更新状态段（patch 体系 + features 机制）
- [x] C3 Makefile 注明 Linux route-b 未支持（STREAM_MOE_SRC 硬编码 async_dio_win.cpp，需 posix 源选择后方可启用）

### D. Linux async DIO 真异步化（新开，评估待做）

- [ ] D1 现状已确认：`src/io/async_dio_posix.cpp` 是同步 `pread` 套 async 接口壳（submit_batch 阻塞读；wait_events 无等待语义）——正确性可用、并发/吞吐不合格（对比 win IOCP 真异步）
- [ ] D2 达标设计（按性能底线）：io_uring 主路径 → 探测失败 fallback io_submit/libaio → 再失败同步 pread（现实现降级为它）；需 sqe/cqe ring + O_DIRECT 4K 对齐（v2 直读 slot 已满足）
- [ ] D3 决策点：Linux 是否已是/将成为生产目标（若非——维持占位，仅当 Linux 正经跑 route-b 大 prefill 前做）

### E. features 重构运行时验证（2026-09-03）

- [x] E1 HTTP 推理冒烟（用户手动 curl hi）——OK（features 重构非运行时回归）
- [x] E2 `node tools/run_export.js` hi（moe-temp0/StreamMoE_dump/gemma original）——rc=0，导出产物齐（chat/prefill_export/expert_history/tokens_id/text/meta）
- [x] E3 **--prefill-from 卡死——已定位并修复（6111cc7）**：根因 = 组容量 < 单层专家数——129-token 单次 decode 在 layer 29（group1 仅 76 slots）活跃集超容量，compute 整层 pin（rc>0）不可驱逐 → worker NO_VICTIM 无限 requeue + compute 死等。验证：8192 池跑通（g1→128 slots，rc=0）、71680 池本就不卡、512 池 fail-fast（报 needs>=983MB）。修复 = 每组分池保底 = 一层全量专家 + 剩余按字节比例 + 预算不足 init 即报错退出

## 关键纪律

- patch 生成用 `cmd /c "git -C third_party/llama.cpp diff HEAD -- <文件> > patches\x.patch"`（PS 重定向写 UTF-16，禁）
- 手术前快照 + 主仓库 commit（README 叠加铁律）
- vendored 永不 commit，改动靠 patch 记录

### F. prefill-from 死锁修复 + 分歧验证收尾（2026-09-03）

- [x] F1 死锁修复验证：8192 vs 71680 IDENTICAL（6111cc7 零数值影响）；512 fail-fast
- [x] F2 分歧调查 + CPU-vs-Vulkan 对照：分歧=任意后端固有噪声（同路由 0.9996/翻放大 0.96-0.98/~5% 专家条目）——moe 非分歧来源
- [x] F3 verify_prefill.js KV ne/nb 修复（4bf25d3）；kv_cos.js 验证 OK（自洽全1）
- [x] F4 结论文档 docs/BACKEND_DIVERGENCE_ANALYSIS.md

### G. GPU/Multi-device M1 - CPU-only private-chain skeleton (2026-09 开工) — G3/G4/G5 已被 O 节（设备执行）取代

> **状态（2026-09-08）**：G3/G4/G5 的"收整层 → 重建 → 官方核执行"计划已由 O 节
> `exec_layer_burst_chain_buckets` + per-device 整链设备图落地并取代；G3b stage-1/stage-2 的历史
> 记录保留如下。

- [x] G1 调研完成：gemma4 门控自定义在 gemma4.cpp（attn_out 上 rms_norm+scale+gate_inp_s+mm，dense 域），probs_in 传入 build_moe_ffn；链 = softmax/topk/weights(norm_w) -> fused gate_up MUL_MAT_ID -> view gate/up -> geglu_split -> down MUL_MAT_ID -> mul weights -> 逐expert view/add -> moe_out；每层 2 个 MUL_MAT_ID；post_norm_2 在链外 dense 域（build_moe_ffn 实际调用参数：gate_up merged/up_down_mm_id/norm_w/gating/down 等）+ 现状 minigraph 接口
- [x] G2 定案：执行形态 A = 逐节点 mini-graph + 数据私有（不建影子链/不重建拓扑，拓扑来自主图 -> 无 per-arch 重建器）；私有 arena 中间 / view 指针偏移 / moe_out 直写主图 dst；专家视图接口 = resolve(L,E)->{pool,slot 基址/stride}；mock 点清单见 docs/MOCK.md
- [ ] G3 实现：graph_compute 收整层 -> 重建 -> 官方核执行（中间 arena 不写主图链中间）
- [ ] G4 CPU 伪双 device_pool + ids 分组/汇总逻辑
- [ ] G5 数值对齐官方（v2/prefill 对比）

- [x] G3a verify 安全网落地（494f670）：route_b_chain 模块 + llama-context 三处 build 挂点 + CMake 源；gemma 验证 420 hidden/0 external。**patch 欠账**：vendored llama-context.cpp（route-b 挂点 hunk，与 prefill 同文件需手动 hunk）+ common/CMakeLists.txt（route_b_chain.cpp 源）待补进 route-b patch
- [ ] G3b 执行器（方案已定）：supports_op 接 moe_chain_node_is_privatizable（非 view 计算节点）；moe_exec phase2 对非 MUL_MAT_ID 计算节点（geglu/mul/add/moe_out——无权重、src 主图真写）用**手动 1 节点 cgraph**（ggml_new_graph + nodes[0]=原节点）官方 cpu 核直跑，**无需 per-op 重建**；view 衔接 + MUL_MAT_ID w3d 不变；顺序=split 节点序。第一版真写 dst 对齐数值，再切藏 arena + 链尾 unpin

- [x] G3b stage-1 完成（d9f5bbb）：显式收编（preset 180 链计算节点 -> stream_moe）+ 修 has_moe 早退 + resolve 单池 fallback + 手动 1-node cgraph 执行无权重链节点；**回归 IDENTICAL**（整层链我们执行、真写主图 dst）。debug log 已清。patch 欠账：llama-context 3 处 assign 调用（vendored，route-b patch 待补）
- [x] G3b stage-2 私有化完成（0829d50/双模式）：乒乓+fullalloc 双模式回归 IDENTICAL（前置聚合问题经乒乓共享持久区绕开）。原内容：：前置 = sched 每节点仍单 split（不聚合整层——中间跨 split 必须真写主图），需先查 pass5 为何预设同 backend 仍拆单节点（聚合后中间才可在同 graph_compute 内自己区迭代）；再改 moe_exec 中间 dst -> 私有 arena + 链尾 unpin

- [x] PATCH-DEBT-1（已清，ae990d7）：llama-context 缝 frag 化——macros 4 锚点 + stmoe_routeb_lctx_*.frag + prefill 段重生成，clean-apply 21 文件逐字节一致。原内容：：llama-context.cpp 的 route-b 挂点（顶部 ROUTE_B include route_b_chain.h + 3 处 build_graph 后 verify/assign 块，均 STREAM_MOE_ROUTE_B 宏 gate）需进 route-b-inject.patch——manual hunk against HEAD（行 1358/2431/3418 + include 区）；**连环**：prefill patch 的 llama-context hunk context 需含 route-b 插入行（prefill 后 apply）——两 patch 一起改 + clean-apply 验证
- [x] PATCH-DEBT-2：common/CMakeLists.txt 的 route_b_chain.cpp 源已补进 route-b patch（CMakeLists 段替换完成）

### H. 长线：消灭 phase2a/2b patch（2026-09 定，不立刻整理）

- [ ] 目标：vendored 改动全经 phase1 打桩（include 锚点）+ 主仓库内容（frag / 独立 cpp via STREAM_MOE_SRC）表达，route-b-inject.patch / prefill-export-llama.patch 最终消失，只留 streammoe-macros.patch。纪律写入 patches/README：小插入→锚点+frag；大块→独立 cpp 或大 frag；不得不直接改 vendored→先问用户。

### I. GPU M1 过渡 - 参数 collection 化 + Vulkan 显存池真申请调查（2026-09）

- [x] I1 pool collection 参数落地（cf0e61a）：`--moe-expert-pools <dev>:<MB>[,...]`（main/draft 统一）；route_b_setup 接 pool list；非 RAM 设备排队 lazy alloc（首次 graph_compute 触发——vulkan 注册晚）；回归 IDENTICAL
- [x] I2 vulkan 注册机制查清：ggml_backend_registry **静态注册**（ggml-backend-reg.cpp:129-136，GGML_USE_VULKAN 编译期 + 无 GGML_DISABLE_VULKAN env），非 llama 初始化/惰性。之前拿不到 Vulkan0 = StreamMoE_dump 默认 CPU-only（build.bat:69 设计），需 `GGML_VULKAN=ON` env 重编
- [x] I3 Vulkan0 真分配：2048MB 成功；3G/4G/5G/6G/7G 全 OOM。两层限制：① RX590 驱动 maxBufferSize 偏低（ggml-vulkan.cpp:6377，需 `GGML_VK_FORCE_MAX_BUFFER_SIZE` env 绕过）② 绕过后 vkAllocateMemory 真 OOM——8GB 卡实际空闲 ~2.5GB（-ngl 2 dense + vulkan 运行时 + 系统占用）。M2+ 分段分配规避，不依赖单 buffer
- [x] I4 **新发现待查**：GGML_VULKAN=ON 编入 StreamMoE_dump 后，纯 CPU decode（-ngl 0）结果也分歧（embd token#0 cos 0.985）。**2026-09-03 归因修正**：不是运行时 op_offload 把计算搬 vulkan——GGML_SCHED_DEBUG 实测默认构建 + `--expert-backend`（隐含 no-op-offload）后 **Vulkan0 splits=0**（CPU=1001/STREAMMOE=990），但产物仍与 CPU-only 编 DIVERGED。分歧来自 **GGML_VULKAN=ON 编译形态本身**：model.devices 非空（vulkan 必进）→ llama-context.cpp:442 把 CPU backend 的 buft 换成 Vulkan0 host buft 等 host 内存形态差异 → gate 边界 expert-flip 噪声。**别用 vulkan 编版对 CPU 基线跑回归**（flavor 不匹配）；vulkan 编版自比用 `moe_129_8192_vk` 基线
- [x] I5 div_match 工具整合（63824cb）：`baseline_regression/tools/div_match.js` = 专家翻转 token 集 vs 高散 token 集匹配度。实测 vulkan 版分歧：34 个高散 token 34/34（100%）有专家翻转、0 无翻转——分歧完全由 gemma-4 专家翻转（gate 边界噪声）解释，非 bug；层 29（末层）翻转最频繁（放大最直接）。run_baseline.bat 加 [6/6] div_match 步——DIVERGED 时自动归因（unexplained>0 = bug 信号），IDENTICAL 时一行确认
- [x] I6 build.bat：`StreamMoE_dump` tag **默认 GGML_VULKAN=ON**（route-B Vulkan0 device-pool 路径需设备注册；运行默认仍 CPU 除非 -ngl）。`upstream_dump`/`main` 默认 OFF 保持 baseline/生产语义；env `GGML_VULKAN=OFF` 覆盖（重建 CPU-only StreamMoE_dump 供 CPU 基线回归）
- [x] I7 **--expert-backend 隐含 no-op-offload（3932d33）**：route B 拥有专家放置权——禁 llama 把 host 计算自动 offload 到 GPU（op_offload 默认 true，llama.h:402）。价值：无 -ngl 也省掉 Vulkan0 compute buffer（~1.3G，sched_reserve 分配，context 构造即发生）+ 不空手把计算搬 GPU。GGML_SCHED_DEBUG 实测 Vulkan0 splits=0（见 I4 归因修正）。llama 机制溯源：llama.cpp:224-258 无条件收 GPU 进 model.devices（不查 -ngl）→ llama-context.cpp:364-382 backends 含 vulkan+ACCEL+CPU → ggml_backend_sched_new(op_offload=true) → host 计算节点 2.sup offload。禁用参数对照：`--no-op-offload`（sched 级）vs `--split-mode none -mg -1`（设备级，llama.cpp:291-293 devices.clear()，连 vulkan context 都省）——目标态用前者
- [x] I8 **vk 数值基线 + run_baseline 参数化（5db4a52）**：`baseline_regression/baseline/moe_129_8192_vk`（默认 GGML_VULKAN=ON 构建 + 自动 no-op-offload 产物）；`run_baseline.bat [baseline_moe_dir] [verify_dir]`——CPU 基线 moe_129_8192 / vulkan 基线 moe_129_8192_vk 按构建形态选；upstream 固定 upstream_129；`*_vk` 自动放宽 KL 阈值 1.0→4.5。README 同步。验证：vk 模式双 IDENTICAL PASS；默认（vulkan binary vs CPU 基线）正确报 DIVERGED + 0 unexplained

### J. VRAM 数据层 - 池真驻留 + CPU 读 vram 执行（2026-09）

> 路线 A 落地（数据层先行，执行仍 CPU；为 GPU 阶段铺数据硬前置）。主仓代码直接 commit；唯一 vendored 改动 = ggml-vulkan host map 导出（patch）。

- [x] J1 **host map 通道（ggml-vulkan.cpp phase1 锚点 → `stmoe_routeb_vk_hostmap.frag`，511cc7a 起）**：ggml-vulkan `get_base` 固定返回假 `vk_ptr_base=(void*)0x1000`（:2407）——vulkan 内部靠 `tensor->data - vk_ptr_base` 算偏移（:2411），**不能改**。经 frag 导出 `stmoe_vk_buffer_host_ptr(buffer)` 返回真 `vkMapMemory` ptr（HOST_VISIBLE buffer，:3548-3550）。RX590 分配即 `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`（rebar/BAR1，create_buffer_device :3597-3606）。实测 4MB pattern RW-OK。2026-09 由独立 patch 改造成 macros 锚点 + frag（见 patches/README）。
- [x] J2 **v2 模型开发路径（5265d0a）**：gemma v2 chunk（专家独立化 direct）——moe(v2) 与 moe_129_8192_vk(v1) 基线 **IDENTICAL**（chunk 重排数值无损）；run_baseline 拆 MODEL_MOE(v2)/MODEL_UP(v1，upstream 不认识 v2)
- [x] J3 **vram 区并入槽空间（7f33b00）**：`subpool_t` → per-(group,pool) 区（pool 0=RAM，1+=device）；`init` 收 `vram_region_t{pool,group,base(n=host map),n_slots}`；槽号全局连续、`slot_mem` 覆盖 vram map；RAM carve/驱逐不变；`ram_subpool`/`subpool_of_slot` API。RAM-only 与 RAM+Vulkan0 均 IDENTICAL
- [x] J4 **read_to_vram + CPU 从 vram 读权重执行（b691c90）**：`alloc_or_evict` pool 感知；`accept_requests` 优先该 group 的 device 区（RAM fallback）；DIO（v2 direct）/staging 写进 vram slot（slot_mem = vram map）；目录记 pool 1；minigraph_exec **激活集动态单区**（w3d/ids 区局部化；mixed RAM/VRAM active set 报错 = J6）。实测全 vram "reads pool 1" + IDENTICAL
- [x] J5 **device 驱逐 demote 回 RAM（77b0bbe）**：vram victim（READY+ref0）内容 memcpy 进 group 的 RAM 区（RAM 先丢最冷腾位），目录迁移 pool；RAM victim 仍丢弃（盘为下一层）。Vulkan0:1024（139 槽 group0）129-token 触发 **1987 次 demote 仍 IDENTICAL**——move 数据路径正确
- [x] J6 **mixed 分区执行**（已随唯一桶引擎 + 设备执行落地，2026-09-08）：`build_mix_plan` 按 pool 产出 RAM round + device round，每 round 单独子 mul_mat_id 并折进各自 acc，最后 host fold（WIP O / M2 §7.9）。原"同层激活集跨 RAM/vram 分区"由 per-pool round 天然覆盖。

### K. v2 块内张量对齐 + SoA pool 布局改造（2026-09 定案，替代 b4-3/v1 路线）

> 根因钉死：**ggml-vulkan MUL_MAT_ID 专家步长硬编码 = `ne0*ne1`（单张量紧凑大小），忽略 `nb[2]`**
> （mul_mm.comp:253 batch_stride_a=ne00*ne01；ggml-vulkan.cpp:10385）。CPU mul_mat_id 读 `nb02`
> （ggml-cpu.c:1654）正确。b4-3 arena-clone 已归档（debug_patch/b4-3-arena-clone/）。
> **v1 sections-v1 已否决（2026-09）**：GGUF tensor offset 必须紧凑单调（gguf reader 校验
> `ti.offset==累计 padded size`，gguf.cpp:774-794），无法在张量内做 4K 切片 stride；writeV1 的
> per-expert reflow 产出非法 GGUF（llama 加载报 offset 不匹配）。实测 gemma perExpert：
> gate_up Q4_K 2230272=4096×544.5（非 4K）/ down Q5_1 1486848（4K）/ 末层 Q8_0 2106368（非 4K）；
> deepseek+dspark Q8_0 4456448 全 4K。

**定案（用户 2026-09，两步 + 装载分流）**

- **文件侧（B，保留 expert-blocks 架构）**：v2 块内**每个张量切片独立 4K 对齐**（gate_up 段、
  down 段各自起点 4K；原紧凑拼接 down 起点因 gate_up 2230272 半块错开非 4K）。块内 offset 计算
  每分支起点 align_up(累计,4096)；pad 由 fill 补（块内空洞）。**DIO 源对齐成立**（每个专家每个
  张量切片都可从 4K 对齐位置发起 DIO）。
- **pool 侧（SoA / struct-of-array）**：subpool 内按张量分列——gate_up 列、down 列……每列 = 该
  张量全部专家切片连续，stride = 单张量紧凑 perExpert（无跨张量空洞）。槽语义从"整专家块 AoS"
  变"列内专家切片"。executor 以单张量为权重单位：w3d 壳 data = 列基址 + e×stride、nb[2]=perExpert
  → vulkan 步长天然命中，无需伪 3D。
- **装载（DIO 分流）**：专家 e 的每张量切片各一次 DIO（读 gate_up 切片 → gate_up 列）：
  - perExpert 是 4K 倍数 → 源(4K)+目标槽(4K) 对齐 → **DIO 直写槽**（deepseek/dspark 全命中、gemma down 前29层）
  - perExpert 非 4K → DIO 读 4K 窗口 → **staging move**（gemma gate_up 2230272、末层 down）
  - 异步 ring buffer / pending 聚合 / staging 基础设施已有（async_load_t reqs[pending]）

**任务**

- [x] K1 转换器：v2 write 侧块内分支 offset 每分支 align_up 4K（computeV2Layout 块内布局 + fill pad）；read 侧 buildLayerBranches 对称（branchOff 对齐）。产出合法 v2（tensor_info 占位不变）——57a2838
- [x] K2 跑转换 gemma original→新 v2（v2align，branch_align=1），验证：块内每张量切片 offset%4096==0、llama/convertd 可读、与旧 v2 数值等价（16/16 切片采样逐字节一致）——N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2align.gguf
- [x] **K3 loader/topo（SoA 列描述）**：`moe_loader.h` expert_group_t 加 `columns`（tag/ggml_type/ne/per_expert/per_expert_4k）；topo_builder 按 (tag,type,perExpert) 合并派生。**纯增量**（AoS 不动，数值不变，prefill-from IDENTICAL）。实测 gemma v2align：group0 = gate_up Q4_K[staging]+down Q5_1[direct]；group1 = gate_up[staging]+down Q8_0[staging]。——9ce6e1f
- [x] **K4 scheduler（SoA 槽分配 + 装载目标；D1/D2 已确认 2026-09）**——4604e1c：
  1. `subpool_t` 分列：组内每列 `{tag, off, stride(perExpert)}`；区域列主序 carve（RAM 池/VRAM buffer 连续），列 c 槽 s = base+off+(s-slot_begin)×stride；**控制面零改动**（slot_meta/expert_directory/owner/pin/驱逐照旧）
  2. `start_async_load`/`drain_completions`：**每分支一次 DIO**（read_plan slice 带 column）；`slice.direct` = 源对齐 + perExpert%4096==0 → DIO 直写列槽，否则 staging memcpy 列槽；demote（device→RAM）逐列 memcpy
  3. budget/floor：group.expert_size = 紧凑 ΣperExpert（不再含块 pad）；`slot_mem` → `slot_col_mem(slot,col)`（槽无单址）；`branch_layout` → `column_layout(sp,name)`
- [x] **K5 minigraph_exec（w3d 单张量壳）**——4604e1c：
  1. legacy + burst + exec_mm_vk 三处：resolve 后 `column_layout` 取 (col_off, col_stride)，w3d data = 区域基址+col_off、nb[2]=col_stride（= vulkan 硬编码紧凑步长，天然命中）、ne[2]=n_slots
  2. CPU/vk 同构造；单区限制（mixed RAM/VRAM 激活集报错 J6）沿用
- [x] **K6 数值门**：
  - [x] **CPU/RAM 路径 IDENTICAL**（v2align prefill-from vs moe_129_8192_vk 基线逐字节一致，129-token 全对齐）——SoA 装载/列寻址/分列 staging/direct 全部正确
  - [x] **vulkan 吃 SoA 列**（2026-09-08，WIP O）：设备执行落地后 VRAM round 真在 Vulkan 上跑整链；RAM8G+VRAM256M 设备混跑对同分区 CPU **cos 0.982**（与已知 0.986 冻结基线同量级，**用户决定不追**）。原 demote 风暴由环形驱逐 + DMA 缓解，不再阻塞。
- [ ] K7 文档同步（GGUF_FORMAT / LOADER_FORMATS / MULTI_SUBPOOL / CHECKPOINT / VENDORED）

### L. 批量 pin：bitmap 层请求 + MPSC 就绪位（2026-09 定案）

> **动机**：逐专家阻塞 pin（exec 每个 miss → push 单条 → `wait_version` 死等该专家 → 下一个才入队）
> 使 DIO 全串行 + exec/scheduler 乒乓（scheduler 装完一个队列空、sleep 1ms）。vram 路径跑不起来
> （demote 风暴 + 从不触发 GPU mm）。同时 M2_DEVICE_EXECUTOR §6 的 `total_tokens`+`start_rdtsc`
> profile 字段一直没载体。
>
> **定案（用户 2026-09）**：
>
> 1. **请求 = 整层一条**：`slot_request_t` 从 `{layer,expert,seq}`（12B）改 **96B 定长 POD**（用户定 80B + wake-once 指针 → 96B）：
>    `{layer u32, total_tokens u32, start_rdtsc u64, n_load_target u32, batch_ready ptr, needed[8]=512bit}`。
>    **n_load_target（曾名 seq）从没被读的 per-expert id 重定义为 batch 装载目标数**；`batch_ready` 是 exec 侧 wake-once 计数词。
> 2. **MPSC 队列就绪位化**：ring 元素普通 POD（96B 不可能进 `std::atomic`，12B 已非 lock-free 退化成锁），
>    每槽 publish generation（tail+1）+ release/acquire，**多生产者安全**（tail CAS 预留、consumer 等 generation 才读、无 ABA）。
> 3. **批量 API**：`pin_layer(layer, bitmap, await, out)` 取代 `pin_expert`/`wait_ready` 单专家 API——
>    exec 把整层活跃集一次 push；scheduler pop 后对 bitmap **集体** `alloc_or_evict` +
>    并发装载；exec **只醒一次**（wake-once 计数词）。
> 4. **wake-once（选 B）**：exec 提交 bitmap 后睡在 `batch_ready` 计数词上，scheduler 每处理一个位
>    （含 skip-resident）fetch_add + wake，计数到 n_load_target（=待处理专家数）即醒。**也给了 profile 一个
>    per-batch 完成点**。fail 也 bump（exec 醒后 rescan 重试，防 spin）。
> 5. **pin 生命周期不变**（整专家 = 全列切片 READY 才可用）：async_load_t 本就"整专家 pending 聚合 0
>    才 mark_ready"；批量后 bitmap 位 = 该专家全列载入完成。exec 只在专家 READY 后 pin，层尾全 unpin。
> 6. **total_tokens** = exec 层 `ids->ne[1]`（batch token 数，burst 收 keys 时可得）；`start_rdtsc` =
>    批量请求发起时刻。draft 与 main 同路径。
> 7. **上限校验**：`scheduler::init` 读每层专家数 `n_expert ≤ MAX_EXPERTS_PER_LAYER(512)`，超了 fail-fast。
>    bitmap 容量取 2 的幂只是覆盖主流，非对模型形态假设（非 2 幂只要 ≤ 容量即可）。
>
> **不做**：mixed RAM/VRAM 同层分区执行（J6/M2-3 独立大工程）；vulkan 后端 dll；real-time profile
> 消费端（M2-4，仅补字段载体）。
>
> **任务**
>
> - [x] L1 slot.h：slot_request_t 96B 定长（layer/total_tokens/start_rdtsc/n_load_target/batch_ready/needed[8]，字段曾名 seq=target）；mpsc 队列普通 POD + 每槽 publish generation + release/acquire，多生产者安全——0518153
> - [x] L2 scheduler.h：MAX_EXPERTS_PER_LAYER=512；删 pin_expert/wait_ready 单专家 API；加 pin_layer(layer, bitmap, await, out)（wake-once）+ bit 助手——0518153
> - [x] L3 scheduler.cpp：init n_expert≤512 fail-fast；accept_requests 按 bitmap 集体装载（device-first，per-bit bump）；drain_completions settle 时 bump batch 计数 + wake（成败都 bump 防 spin）——0518153
> - [x] L4 minigraph_exec：burst 整层一次 pin_layer；legacy 角色 split 批量 pin 到 pin_state、down 只确认 pinned——0518153
> - [x] L5 测试：test_scheduler/test_slot 适配新 API + 4-producer mpsc 测试；**5/5 ctest 过**——0518153
> - [x] L6 编译 + 回归：gemma v2align prefill-from vs moe_129_8192_vk **IDENTICAL**；**wall time 几十秒 → 0.11s**（DIO 并发化 + 消乒乓）——0518153
> - [x] **L6b vram GPU 触发**（2026-09-08 解决，见 WIP O / M2 §7.9）：设备执行落地——VRAM round 的整链真在 Vulkan 上跑（device shell + async 提交 + acc_d 回读）。demote 风暴由环形驱逐 + DMA 缓解；RAM8G+VRAM256M 跑通。
> - [x] L7 文档同步（WIP L 节；CHECKPOINT 已引用）

### N. NO_VICTIM 驱逐死锁修复 + 无进展 stall 兜底（2026-09 落地 5d08bb3）

> **根因（lldb 定位，进程 27544）**：`alloc_or_evict` 驱逐扫描是固定下窗 `delta 1..layer`——
> **layer 0 候选集恒空**（delta 无负层），池满后任何新 layer-0 专家 miss 永远无法驱逐 →
> `accept_requests` 收 leftover requeue + `worker_loop` 因 any=true 不 sleep → **单核 100% 自旋**
>
> - exec 在 `batch_await.wait()` 永久睡（此前 `-p hi` decode ~20 轮后卡死在 `*Selected response:*`）。
>   也解释了为何长 decode（>池容量累积）必死、短 `-n` 能过、vram 池场景更易触发。
>
> **修复**：
>
> 1. **驱逐组内 ring**（alloc_or_evict）：从当前层 `(pos + n - k) % n` 环扫本 group 全部层，
>    layer 自身 ref0 旧专家优先（本 token 不用 = 最安全 victim），L0 自然覆盖高层兜底；
>    victim 限本组 slot 区间（顺带消除跨 group 列几何误用）。**数值门 = CPU 基线回归 IDENTICAL PASS**。
> 2. **accept_requests 进展感知**：仅 requeue 不放置 → 返回 false → worker 退避 sleep（不再空转）；
>    ring-full 不算 stall（DIO 背压，drain 自解）。
> 3. **stall 兜底 fail-settle**：同一请求 wall-clock 连续 2s 无放置（no_slot）→ leftover bits
>    逐个 bump（wake-once）唤醒 exec → rescan 仍缺 → pin_layer -1 → GGML_STATUS_FAILED 上抛
>    （链已核实：decode -3 → server "Compute error." send_error → cli 报错退出）。
>
> **验证**：RelWithDebInfo dbg build `-p hi -st` 自然退出 + 正文完整（修复前单核自旋卡死）；
> 纯 RAM 与 Vulkan0:2048 池均跑通；CPU 基线回归 moe/upstream 全 IDENTICAL PASS。
> **与 K6/L6b 关联**：demote 风暴在环形驱逐下不再死锁（能跑通而非卡超时），
> K6 "vulkan 吃 SoA 列数值正确" 的 device 触发路径值得按 L6b 待办重测。

### M. Expert Move Pipeline + (L,E)-Keyed Eviction（2026-09 设计定稿）

> **完整设计见 `docs/EXPERT_MOVE_PIPELINE.md`（EN）/ `.zh-CN.md`** —— 一次工作会话
> 敲定的下一轮 scheduler 重构，建在已落地 L 节批量 pin 之上。本文档含 open questions
> 与构建顺序，改动面大，动码前以它为唯一设计依据。
>
> **动机**：批量 pin 消除了 DIO 串行，暴露 demote 同步 memcpy 成新瓶颈——vram 池
> 场景 ~900+ 次 device→RAM demote 全在调度线程同步做（递归为 RAM 腾位），exec
> 卡在批等待、从未触发 `exec_mm_vk`（L6b）。
>
> **定案要点**：
>
> 1. **Directory 加宽为 (L,E)×pool 生命周期状态表**（A1：state_+slot_ 两个并行原子）：
>    ABSENT/LOADING/READY/MOVING_OUT/MOVING_IN/FAILED——半途状态对 exec 可见，
>    消灭"正在装/在搬"与"缺失"不可分的重复装载窗口。完整 move 描述住任务对象，
>    不进 directory。
> 2. **装载顺序不变量**：先发布 `dir=LOADING(slot)` 再 `begin_reload`（可见点在前）。
> 3. **删 owner_**：8 处编译错误全在 alloc_or_evict（登记×2/打分×2/victim 反查×2），
>    无外部依赖。驱逐改 (L,E) 键后槽侧不再需要"槽里是谁"。
> 4. **驱逐 = (L,E) 键 + 层距偏好**：为 layer-L 批腾 K 槽，从 L-1 往前逐层，最近层
>    先逐（top K/2/层，score 低先逐；score 线估算后置）。枚举用逐层查 entries_
>    （O(256)/层，只扫近层），将来可换 per-(pool,layer) 驻留列表。
> 5. **Move 管线（v2r+r2v 统一异步）**：1 个 copy worker（数量参数化），move_task
>    提交队列 + 完成队列（仿 DIO），worker 只 memcpy、控制面收尾回调度线程；
>    源槽保持 EVICTING 到 cq（不 drop——drop 因状态混乱被否）；排他用槽状态
>    （源 EVICTING/目标 IO_INFLIGHT）**不用 refcount**（refcount 是计算读者语义，
>    驱逐 drain 会自锁）。
> 6. **v2r 不做 GPU/vulkan DMA**：独显 host-visible heap 仍是显存，系统 RAM 不在 GPU
>    地址空间，copy engine 写不到（UMA/APU 才有）。只能 CPU memcpy，故异步 worker。
> 7. **每模型单活跃请求槽（调度侧记账）**：exec 请求 per-model 串行——每 scheduler
>    （本就 per-model）一个 active 槽；drain 是"任何槽变 READY 的唯一地点"，凡 settle
>    且属于 active.still-need 的专家 bump active 一次；归零 wake exec 一次。无 waiter
>    列表、无 "skip 算完成"。多 ctx 并发 decode 需多活跃槽——将来分叉，见设计 §8.4。
> 8. **exec 无状态（替代 6a）**：exec 不认 ABSENT/LOADING——全 try_pin，失败的收进
>    请求、睡到全部 READY 唤醒、醒后重扫，幂等自愈。重试轮 n_load_target = 当轮重测数。
> 9. **预取复用完整 DIO 路径**：预取 = 同一 async_dio_engine 上的 submit，完成走同一
>    drain（mark_ready + dir set + wake + profile delta rdtsc 附加）；预取也先发
>    LOADING；settle 时若属 active.still-need 则 bump——无需专门预取→active 通道。
>
> **任务（详细 build order 见设计文档 §9）**
>
> - [x] M1 directory 加宽（A1 state_+slot_）——6b6300e
> - [x] M2 装载路径：先发 LOADING 再 begin_reload；state-aware accept——dbaff8f
> - [x] M3 驱逐改 (L,E) 层距（alloc_or_evict 内）；删 owner_——45b14de
> - [x] M4 move_task ring + worker（v2r+r2v）+ 完成 drain；v2r 接进驱逐——fdf4982 + **2026-09 DMA 提速**：v2r demote 改 transfer-queue DMA（stmoe_vk_dma_read，
>       ggml_vk_buffer_read 同步路径，cached staging），158ms→~1ms/专家（717bac8）。
>       r2v 无需改（rebar 写 ~8GB/s）。实测见 docs/VRAM_DMA_MOVE.md。
> - [x] M5 调度侧记账（active-slot + exec 单程，2026-09-05）：exec 本地 try_pin 已 READY 的 A；
>       其余 B 作为**一个 active 请求**提交，scheduler 侧 active 槽统一登记 + 现查三态 triage
>       （READY→替 pin / ABSENT→装载 / LOADING→等 drain，closes register-vs-settle 竞态）；
>       drain/move settle 统一替 exec pin B（RAII ledger）+ 归零 wake-once；失败 RAII 回滚 +
>       failed 信号。exec 醒后 scan B 拿 slot 不再重复 pin（单程，去掉两轮 round/rescan）。
>       设计：EXPERT_MOVE_PIPELINE §7.4。验证：CPU 基线回归 IDENTICAL PASS、-p hi -st 自然退出、
>       Vulkan0:2048 池跑通、**DeepSeek-V4-Flash 70G 池（5621 slots, Q8 direct, 5 shard）hi -st
>       跑通**。slot_request_t 96→104B（+failed 指针）。
> - [x] M5b 退出泄漏审计（STREAM_MOE_TEMP / dbg tag，98f4d6d + 8ab6f2a）：~expert_scheduler 析构先
>       静默 500ms（仍注册、worker 排空在飞 prefetch/move）→ stop() → 逐 slot 查 refcount>0 或
>       残留 state（EVICTING/IO_INFLIGHT/FAILED），反查 (L,E) 打 stdout；干净打一行
>       "pool ok - N/M slots used, all refcount 0"。注入测试证实能抓 slot 泄漏；
>       RAM/Vulkan0/deepseek 三场景退出全 0 泄漏。
> - [~] M6 验证：纯 RAM 路径零影响（DMA 代码只在 v2r 触发，RAM 走 memcpy 不变）；VRAM
>   demote 场景 DMA 内容 0 BAD（4-token 64MB，238 demote，列级对比纯 RAM golden）；
>   129-token 64MB 跑通 14.4s（DIO/计算主导，demote 已摊销）。exec_mm_vk 到达 + 新 UT 待补
> - [ ] M7 文档同步（设计文档 §8 open questions 逐条收敛；EXPERT_MOVE_PIPELINE 更新 M4 DMA）
> - [ ] M8 UT：M5 §7.4 四个并发验收用例（登记前 settle / B 含 ABSENT / B 中途失败回滚 / 单活跃）——
>       test_scheduler 目前因 stmoe_vk_dma_read 链接问题不编（既有），M8 需先修 test 链接或改独立测试

- [x] M2-2 真并行骨架：per-device 整链 cgraph + async 提交 + CPU/VK 重叠 + acc_d 回读 + host fold（WIP O / M2 §7.9；overlap 已验证 = 串行，逐层 acc IDENTICAL）
- [x] M2-3 出口 scatter 通用化（随 per-pool round + `layer_fold` 落地）：每 pool 一个 round，折进各自 acc，host 端按参与集 fold；设备 acc 走 `tensor_get` DMA 回读
- [ ] M2-4 profile 埋管：ctx 聚合 + profile ring + 事件 struct + io/device 完成时间戳（消费策略后置）

### M2-5 桶化 append 链 + per-device acc + 最终 host fold（2026-09-06，设计定稿 §7.8）

设计：§7.8 三段。三个 append 函数**一律按当前桶构建**（用户强令，绝不在设计上退让），
紧凑链重建（weightless 吃紧凑前驱，非主图满宽）：

- f1 append_bucket_chain / append_mm_shell / append_op_clone：窄桶列 `[d_out, w_b, n_active]`，
  mm 用桶 ids 子集 + build_cur_sub，输出紧凑 dst（不复用满宽 nd->data）；
- f2 append_expert_fold：折专家 → `[d_out, 1, n_active]`（每 token 专家宽 1）；
- f3 append_scatter_to_fullwidth：**一律**累加进 acc_d（`ggml_acc`，全 token stride=1 同位 add；
  token 子集 = scatter-add）；mock：全宽静默，非全宽 log+exit(1)——mock 不是 no-op，全宽也要真加；
- acc_d：每 device 一个常驻 device buffer `[d_out, n_t]`（build 时随 device 建），跨层复用，
  **层首 `ggml_backend_buffer_clear(acc_d, 0)`**（同步，vulkan 走 vk_buffer_memset + waitFence，
  host 侧 graph_compute 前调用）→ 图内 f3 自然读到清零后 acc_d，无需 GPU 事件节点；
- 最终：acc_d → add_in[k]（§7.8 stage2，GPU = ggml_backend_tensor_copy）→ host fold device_used 槽 → moe_out；
- 单 CPU 模拟多桶：按专家（k 槽）分两桶验证数值。

当前代码：79d0f7c 函数拆分（fold off 全铺 = IDENTICAL 回归基线，保留）；
09c0c52 §7.8 单 device 三段（acc_d→add_in→host fold，device_used=1，数值 = 预期 ulp 分布）。
待做：三函数真桶化（紧凑链重建）+ 桶循环 + 逐层 dump 宽松 gate。

**落地（2026-09-07，STREAM_MOE_TEMP / dbg 构建 / env STREAM_MOE_TMP_CHAIN_BUCKETS）**：
CPU 单 pool 两桶原型引擎已写进 `exec_layer_burst_chain_buckets`（minigraph_exec.cpp）：

- 桶划分 = k-slot 连续段（`tmp_split_blocks` 家族，新增 `cut<N>`/`cutN,M`/`head<N>`；"one"= 单满宽桶走紧凑构建器用于隔离调试）；
- 每桶：`append_mm_bucket`（壳 ids 用 `b.ids_slot` 子集、cur 取链内紧凑孪生/外部共享 leaf、写紧凑 `[d_out,w_b,n_t]` dst）→
  `append_op_bucket`（weightless 紧凑孪生：槽轴只窄 ne1、REPEAT 满宽、GET_ROWS 换桶 ids_exp、外部 per-slot leaf 按 k_lo 推进）
  → `append_expert_fold`（折 w_b）→ **真 acc**：`ggml_acc_inplace` 同位累进常驻 `acc_d`（offset 0；token 子桶仍 exit 1）
  → `chain_exit`（acc_d→add_in→host fold→moe_out，device_used=1）。
- 关键事实：① 槽轴=ne1 由"形状与 n_k 同值"推导不可靠（n_t==n_k 时撞 token 轴），改为**固定 ne1** 判断；② 每桶 ids/孪生缓冲必须 append 进 `mm_ids_pool`/`fold_buf` 保活（图末尾一次 graph_compute，跨桶 leaf 指针不能悬）。
- 验证：129-token prefill-from（v2align，`temp/chain_buckets_gate.bat`）：
  - `one`（紧凑构建器单满宽桶）vs `full`（旧满宽 clone fold 路径）**30 层 moe_out 逐字节一致** → 紧凑孪生引擎几何正确；
  - `cut3`（[0,3)+[3,8) 两桶）L0 maxAbs 7.6e-6 ≤ 1e-5（唯一同输入可比层）；L1+ 因折序 ULP 经层间残差放大 + L3+ argsort 专家翻转发散（已归档噪声类，非 bug）。
- 待做：逆序桶序看 ULP 上限；token 子集 scatter-add（mock 挡）；deepseek 的 clamp/swiglu 紧凑克隆；把紧凑孪生孪生键从"per-bucket 重建"提升到跨桶复用（REPEAT/外部 scale 表与桶无关）。

**scatter_plan 设计（2026-09-07，docs/SCATTER_PLAN.md 双语文档已落，等用户）**：

- 语义确认：tight 重排（如桶 token {0,2,3,4} → {0,2,4,3}）+ acc 等差段（src 连续 len 列 → dst 等差 base/Δ）。
- **ggml_acc 支持 dst 等差间隔（Δ≠1）**：CPU 核 dst[offset+i1*nb1]+=src1[...]，nb1=Δ*d_out*4、offset=base*d_out*4，中间 dst 列不动（ops.cpp:1215-1233 验证）。
- 重排消费在 **cur 拷贝层**（首 mm 前按 tight 序收集 cur → 链列序继承 → per_token 天然分段）。
- per-token 受影响输入盘点：cur / ids / weights_norm（per-slot 路由权重，slot 切片+token 重排）；专家权重与 scale REPEAT 表不受影响。
- 待做：scatter_plan.h/.cpp 纯模块（贪心最大 run 抽取）+ test_scatter_plan.cpp。

**收敛计划（2026-09-07 冻结，未执行——先做 out_off buffer 改造）**：

- 目标：唯一执行引擎 = 紧凑链 buckets 引擎；删 A 全族（exec_split_legacy_impl / exec_mixed_mm /
  exec_one_burst / hide_output / hide_burst / refresh_aliases / tmp_split_* 自测 / exec_round_cpu|vk /
  build_cur_sub / scatter_sub_dst）；B（`exec_layer_burst_chain_buckets` + append_*_bucket 全族 +
  append_expert_fold + chain_exit）移出 `#ifdef STREAM_MOE_TEMP` 无条件编译；exec_layer_burst 只留
  ex 闭包 + input_layouts 指针修复 + pin 段，之后无条件调 buckets（默认单全宽桶 = cut=one 语义，env
  `STREAM_MOE_TMP_CHAIN_BUCKETS` 仍可切多桶）；`moe_exec_mul_mat_id` 非捕获（layer<0）直接报错。
- 前置待解决：deepseek clamp/swiglu 紧凑克隆、token 子集 scatter-add（scatter_plan 接入）、multi-pool。
- 用户拍板记录：只留 buckets 多桶引擎（删 exec_layer_burst_chain/append_mm_shell/append_op_clone/
  append_bucket_chain 单桶参考，buckets 自吞单桶）；legacy 分支也删（非捕获报错）；B 移出 TEMP。

**✅ 删 A 收敛完成（2026-09-07 落地）**：minigraph_exec.cpp 2490 → 1119 行。

- 桶引擎 + tmp_split_blocks/tmp_blk_t + tmp_dump_node dump 族全部移出 `#ifdef STREAM_MOE_TEMP`
  无条件编译（内部 debug/dump 子块保留，env 门控）；删 B 单桶参考（append_mm_shell/append_op_clone/
  append_bucket_chain/append_scatter_to_fullwidth/exec_layer_burst_chain）；exec_layer_burst 只留
  ex + lsum + input_layouts 修复 + pin + set_full_alloc + **无条件调 exec_layer_burst_chain_buckets**
  （无 env/full/one = 单全宽桶）；moe_exec_mul_mat_id layer<0 = 硬错误；删 A 全族 + pin_state 系 +
  tmp_plan/tmp_run/tmp_split/tmp_bucket 自测 + s_hide_flip/s_layer_* 死状态；route_b_chain 删
  pingpong_ok/pingpong_buffer/g_pingpong_ok（fullalloc/set_full_alloc 保留，B 的 twin_out 用）。
- **关键修复（运行时才发现）**：`moe_exec_mul_mat_id` 会收到捕获 producer 的 **view/layout split**
  （如 `ffn_moe_gate-0` VIEW，producer = gate_up mm 输出），改造前由 legacy 兜底指 data；删 legacy 后
  变 un-captured 硬错误 → 加 `is_view_op(first) → SUCCESS`（view 无执行内容，layer burst 已修 data 指针）。
- 验证（StreamMoE_dump_dbg，gemma 129）：cut3 prefill_export（hidden）+ 默认单桶 L0 moe_out 均
  **逐字节一致** vs 删 A 前 arena 版本；kv_cos vs moe_129_8192_vk 全 ~1.0；`build.bat test main` 6/6。
- 遗留（本手术范围外，agent 在纯净 HEAD 复现）：`build.bat llamalibs main`（GGML_VULKAN=OFF）最终 exe
  链接失败 —— 预存在 `stmoe_vk_buffer_host_ptr/stmoe_vk_dma_read` 未定义（moe_backend/route_b_inject
  引用，非 vulkan 构建无 frag）。`StreamMoE_dump_dbg`（VULKAN=ON）链接正常。
- 仍待办（收敛前置）：deepseek clamp/swiglu 紧凑克隆、token 子集 scatter-add（scatter_plan 接入）、
  multi-pool。buckets 引擎现在仅验证过 gemma 全 token 垂直切。

**DeepSeek 删 A 后验证 + 数值基准冻结（2026-09-07）**：

- **能运行**：`upstream_dump`（纯上游 llama CPU）能跑 UD-00001（v2chunk 布局上游原生可读）；
  删 A 后桶引擎 `-p hi -st` 跑通、无崩溃、泄漏审计 0、两次同参重跑逐字一致（确定性正常）。
- **数值 gate（hidden cos 判据）**：upstream `-p hi -st --export-dir` 导出 96-token 生成流 →
  同 tokens_id.bin 分别喂 upstream server 与桶引擎 server prefill-from → verify_prefill：
  hidden cos ≈ **0.99999999**（层输出与上游一致）；embd token#0-2 cos 0.9999999、多数 >0.999；
  expert_history 5958/73530 翻转（8%，gate 边界 expert-flip，累加结果不变 → hidden 仍 cos~1）；
  KV 因布局不同（上游原生 vs 桶自定义）不做字节比。
- **文本差异非 bug**：A 引擎 "Hello!" vs 桶引擎 "Hi there! 😊" = 8% 翻转在解码下游放大
  （gemma 同类已知行为），非删 A 回归。
- **基准冻结**：`baseline_regression/baseline/deepseek_hi_up`（upstream CPU 参照）+
  `deepseek_hi_moe`（桶引擎产物，同 tokens），README 记 DeepSeek gate 判据（cos，非逐字节）。

**out_off arena 改造（2026-09-07 落地，M2 §7.2.1 手动 arena 串行复用）**：

- 孪生输出 data 从"每桶独立 fold_buf heap"改为钉 `fullalloc arena + ex->out_off[闭包索引]`
  （bucket_build_t.twin_out；compact [d,w_b,n_t] ≤ 满宽 [d,n_k,n_t] 同 out_off 区不冲突；
  无 verify 布局时 fallback heap）。多桶在同一 cgraph 按序串行，后桶自动复用前桶 dead 区。
- 验证（StreamMoE_dump_dbg，gemma 129）：`one`/`cut3` L0 moe_out + cut3 全层 prefill_export
  （hidden）**逐字节一致** vs 改造前；CHAIN_DEBUG 计数确认每桶 7 孪生全 arena 命中、0 heap。
- acc_d 即累加器（chain_ctx 持），层首 fill 0 = reset（既有）。fold 中间仍走 fold_buf。

**test 修复（2026-09-07）**：

- **scheduler 后端解耦**（per-pool DMA reader）：scheduler.cpp 删除 `stmoe_vk_dma_read` 前置声明/直调，改 `expert_scheduler::set_pool_dma_read(pool, fn)` + 私有表 `pool_dma_read_[MAX_DEVICE_POOLS=8]`（pool-1 索引）；move_worker 按 `t.src_pool` 查表，未注册回退 host memcpy。route_b_inject 在 vram_seg 注册时按 seg.dev 前缀（Vulkan）`set_pool_dma_read`；CUDA 落地时同点注册 cuda 壳。**test_scheduler 不再需链 vulkan**。
- **test_moe_loader 修复**：CMake 源补 model_builder.cpp/topo_builder.cpp（缺 parse_model_path/build_topology/parse_model）；断言从废弃 `expert_slot_size/staging_size` 改验 `groups[].columns[].per_expert`（SoA 真载体）。deepseek 实模型跑通。
- **test_async_dio 临时禁用**：SoA 重构删了 `sub_tensor_req_t::slot_offset`，其用例3（多 tensor 单 slot 拼装）是 AoS 语义需重写；CMake 注释目标 + build.bat test 列表去之。文件保留待重写。
- 结果：`build.bat test main` 5/5 绿（moe_loader/profiler/scheduler/slot/mix_plan）。

**token-subset round（2026-09-08 定案，docs/BUCKET_EXEC_TOKEN_SUBSET.md + SCATTER_PLAN.md 已同步）**：

- 目标：桶源从"全 token k-slice（`tmp_split_blocks`）"换成 `build_mix_plan` 的**真 token
  子集 round**（`[w_b, n_active]`），接 `scatter_plan` 做累加器写回。`expert_pool[e] =
handle.pool` 直接从 pin 返回的 handle 取（`pin_layer` 本就 per-expert 带 pool），不需要
  新 scheduler 查询；单 RAM 池退化为一个满 round（默认路径必须逐字节 IDENTICAL）。
- 关键修正：
  1. cur gather = 把 `[d,1,n_t]`（ne1==1）**reshape 成 `[d,n_t]` + get_rows + reshape 回**
     （get_rows 只 gather ne1，不 gather ne2）。
  2. 任意 (t,k) 权重（`weights_norm`）= **通用 flat index-gather 节点**：`[1,n_k,n_t]`
     reshape `[1,n_k*n_t]`，i32 leaf `idx[i*w+s]=k+t*n_k`，get_rows，reshape
     `[1,w,n_active]`。取代 `bucket_ext_leaf` 的连续-k 切片。
  3. 测试分桶 = `minigraph_exec.cpp` 内 `static` 助手，**定义 + 调用点都 `#ifdef
STREAM_MOE_TEMP`**（只有 `StreamMoE_dump_dbg` 带宏）；k 奇偶 × t 奇偶 = 4 round
     （刻意非连续 k），验证完即删。不建 `bucket_split.h/.cpp`、不写离线 UT。
  4. 删 `tmp_split_blocks`/`tmp_blk_t`（旧测试 cut 族，现无条件编译）。
- 任务（实施中，principle 14：全写完再统一回归）：
  - [x] 通用 index-gather 助手（cur row_size=d / weights row_size=1）
  - [x] 执行器 round loop 接 `mix_round_t`（expert_pool 从 pins；tight ids；fold→tight per_token）
  - [x] scatter_plan acc 写回（每 seg 一次 `ggml_acc_inplace`）
  - [x] 宏包裹测试强制分桶 + 删 tmp_split_blocks
  - [x] 回归：默认单 round vs **当前 HEAD 干净构建**（StreamMoE_dump，同 v2align 输入）**IDENTICAL**（embd/hidden/KV + expert_history 全同）；`STREAM_MOE_TMP_BUCKET_ROUNDS=1` 强制 k 奇偶 × t 奇偶 = 4 round（w_b=4，n_active=65/64，非连续 k）L0 moe_out vs 默认 **maxAbs=3.8e-6 ≤ 1e-5 / cos=1.0**（宽松 gate，31.6% 元素差 1 ulp）。
- **回归对拍口径（2026-09-08）**：冻结基线 `moe_129_8192_vk` 与当前 HEAD 的 embd/hidden 有**部分 token 不一致**（如 token#0 cos≈0.986）——**已知现象**（K5/删 A 之后引擎与旧冻结基线本就存在 token 级差异，非本次改动引入；用户决定不重建）。因此本次回归改为**对拍当前 HEAD 干净构建**（同输入同 flavor）：默认单 round IDENTICAL，强制分桶 L0 宽松 gate。`run_baseline.bat` 对旧基线的 DIVERGED 属已知，不作本次判据。
- vendored 改动：`common/CMakeLists.txt` STREAM_MOE_SRC 加 `backend/scatter_plan.cpp`（route-b-inject.patch 已重生成 + 反向 check 通过）。

### dump 确认的槽维事实（2026-09-06，CPU 与 Vulkan 一致，见 tmp_dump_l0_vk / tmp_ds_l0）

- **槽维 = 链内所有张量的 ne1**（= 该层 n_k，gemma L0 8 / deepseek 6），贯穿 mm→weightless：
  gemma gate_up/geglu/down/weighted `[d0, 8, 129]`，deepseek gate/up/clamp/swiglu/down/weighted `[d0, 6, 4]`；
- mm 节点走 device 时 dump 前缀 `gpu_i*`（vulkan），weightless 走 `cpu_i*`——当前执行模型 mm 在 device、
  weightless 在 CPU host，形状不变；compact 规则两种 backend 相同；
- 折叠产物（匿名 ADD 树 + moe_out）`[d_out, n_t, 1]` 天然无槽维（ne1 = token）；
- **scale（gemma node_56/57，llama-graph.cpp build_lora_mm_id w_s 分支 1552-1555）**：
  `reshape_3d(w_s,1,n_expert,1) → repeat_4d(…,1,n_expert,n_tokens,1)` = node_56 `[1,n_expert,n_t]` →
  `get_rows(s, ids)` = node_57 `[1,n_k,n_t]`（按 ids 每槽取 scale）→ `mm * s`；
  桶化：get_rows 用**桶 ids 子集** → `[1, w_b, n_t]`；repeat 输入照旧（覆盖桶 ids 引用的 expert rows）；
- deepseek 无独立 scale 节点（scale 权重在 down_exps，weighted 前未展开成节点 dump）——需另查其
  ffn_moe_down/weighted 的乘结构是否内嵌 scale（见 §7.8 外 leaf 表：down 壳 w 3 分片 + scale 合并？）。
- 桶化待处理外部输入：scale（node_57 类）也按桶槽取，不是"链内收缩"。

### O. 设备执行（M2-2 全并行骨架，2026-09-08 开工）

> 设计：docs/M2_DEVICE_EXECUTOR.md §7.9。用户拍板：直接上全并行骨架（per-device 整链
> cgraph + async 提交 + CPU/VK 重叠 + 多设备 fold），不做 mm-only + 回读的过渡形态。

**闭包外 leaf 盘点（verify dump，`baseline_regression/temp/struct_L0.txt` / `struct_ds_L0.txt`）**

- 大权重（pool 驻留）：gemma 2 个（gate_up/down）；deepseek 3 个（gate/up/down 分片）。
- 小张量：cur `[d,1,n_t]`（~1.4MB）；ids `[n_k,n_t]`（几 KB）；per-slot 路由权重
  `ffn_moe_weights_norm`/`_scaled` `[1,n_k,n_t]`（几 KB）；per-expert scale 表
  gemma `ffn_down_exps.scale` `[n_expert,1,1]`=**512 B/层**（deepseek 无）。
- 结论：设备侧非权重开销 MB 级；scale 每设备存全量可忽略；cur 挑 token 已实现
  （`bucket_gather_cur`）。

**已确认的硬点（无阻塞）**

- 设备 tensor 绑定范式（删 A 前 `exec_round_vk`）：`t->buffer=设备 buffer;
t->data=stmoe_vk_buffer_host_offset(buffer, off)`（`vk_ptr_base+off`）。
- vulkan mm_id 步长 = `ne00*ne01`（dim01 连续），命中 SoA `col_stride=perExpert`（K6 目标）。
- vulkan 支持 ACC（op_params 的 nb1/2/3/offset，含 delta 等差）/GET_ROWS/SUM_ROWS/CLAMP。
- 回读走 `ggml_backend_tensor_get`（transfer-queue DMA），不用 rebar host 读 0.02GB/s。

**任务**

- [x] chain_ctx 加 target（pool/backend/arena+stage buffer/host map/bump）+ bind_pool /
      bind_arena / bind_stage 助手；CPU 路径行为不变（`device_target_t` + `chain_ctx.dev`；
      `bucket_ref_leaf`/`bucket_upload_leaf`/`bind_fresh`；`fix_view_buffers` 给 ggml 视图补
      `buffer`——vulkan 从 `tensor->buffer` 解析，不像通用 API 跟随 view_src）
- [x] 设备 staging 上传（cur 全量 / ids / 路由权重 / scale 表）+ 池权重 shell 绑定
      （`t->buffer=sp.dev_buf; t->data=stmoe_vk_buffer_host_offset(sp.dev_buf, col_off)`）
- [x] per-device 整链 cgraph（mm→weightless→fold→acc_d 全在设备；device arena + 设备 acc；
      `acc_d` 在 arena `[result_bytes, +d_out*n_t)`，层首 `tensor_set` 清零）
- [x] 编排：round 按 pool 分区 → 各 target 建图 → 设备图 async 提交 → CPU 图跑 → converge
      （synchronize + `ggml_backend_tensor_get` 回读）→ `layer_fold` → moe_out
- [x] 回归：纯 RAM 默认 vs HEAD 干净构建 **IDENTICAL**；RAM8G+VRAM256M 跑通、VRAM round 真的
      走设备（`dev=1`）、对同分区 force-cpu **cos 0.982**（与已知 0.986 同量级，backend gate）

**踩坑记录**

- **vulkan ACC 的 nb2/nb3 不能为 0**：`acc.comp` 用 `src1_i / p.nb03 / p.nb02` 分解索引（CPU 核
  忽略 2D src1 的 nb2/nb3）。传 0 → 除零 → acc 全零。改成传 `d_out*n_t*4` 后 acc 正常。
- **ggml 视图 buffer**：`ggml_vk_tensor_subbuffer` 直接读 `tensor->buffer->context`（不跟随
  view_src），合成图里 ggml 建的 view（reshape/permute/view）buffer=NULL → `fix_view_buffers`
  从根 view_src 补。
- **CPU/VK 重叠（已核实可用，2026-09-08）**：async 提交设备图 → 调用线程跑 CPU 图 → 层尾
  sync + 回读。**overlap vs 串行（`STREAM_MOE_TMP_NO_OVERLAP`）逐层 acc 与最终产物
  IDENTICAL**（128M/256M 都试），3 次重跑一致。早前一次 0.228 观测是旧 binary/状态残留
  （非重叠问题）——按"先对比每专家张量/累加器/最终 add"的口径逐层 acc 全部 maxAbs=0。
  诊断 env：`STREAM_MOE_TMP_DEVDBG`（acc/arena 范数、gf 节点）、`STREAM_MOE_TMP_FORCE_CPU`
  （设备 round 落回 CPU 对拍分区）、`STREAM_MOE_TMP_ACC_DUMP=<dir>`（逐层 acc dump）、
  `STREAM_MOE_TMP_NO_OVERLAP`（串行对拍）。
- 设备多 round 几何本身已验证：512M 全设备 one-round vs k/t 奇偶 split-4-round **IDENTICAL**
  （ACC 跨 round 累加无问题）。
- **deepseek 设备实测（2026-09-08）**：`RAM:71680 + Vulkan0:1024`（80 槽），`--prefill-from` 95
  token（`deepseek_hi_up/tokens_id.bin`），设备 round `dev=1` 真在 Vulkan 上跑（3 w-shell +
  clamp/swiglu + 6 外 leaf/层）。对 CPU 桶引擎基线 `deepseek_hi_moe` 与上游 `deepseek_hi_up` 均
  **embd cos 0.9999998 / hidden cos 0.99999997**（cos gate 内）；expert_history 6297/73530
  （8.6%）flip（允许，gate 边界）；退出 0 泄漏、无错误。→ 设备路径对 deepseek 闭包成立。

# Work In Progress - patch 体系手术 + 收尾

> 会话任务清单，边做边更新（多 commit，每步成功即提交）。
> **上一轮：全部完成（2026-09-03）**。patch 体系已对齐工作区（干净 apply 逐字节一致）、ASan 整合进 build.bat、文档同步。
> **本轮：Linux async DIO 真异步化（评估待做）仍在。** 另完成：E3 死锁修复验证、verify_prefill.js 修复、backend 分歧分析（docs/BACKEND_DIVERGENCE_ANALYSIS.md）。

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


### G. GPU/Multi-device M1 - CPU-only private-chain skeleton (2026-09 开工)
- [x] G1 调研完成：gemma4 门控自定义在 gemma4.cpp（attn_out 上 rms_norm+scale+gate_inp_s+mm，dense 域），probs_in 传入 build_moe_ffn；链 = softmax/topk/weights(norm_w) -> fused gate_up MUL_MAT_ID -> view gate/up -> geglu_split -> down MUL_MAT_ID -> mul weights -> 逐expert view/add -> moe_out；每层 2 个 MUL_MAT_ID；post_norm_2 在链外 dense 域（build_moe_ffn 实际调用参数：gate_up merged/up_down_mm_id/norm_w/gating/down 等）+ 现状 minigraph 接口
- [ ] G2 私有链重建器设计（输入契约 cur/ids/weights/槽布局 -> 私有 ctx 重建 gemma 层链 -> moe_out）
- [ ] G3 实现：graph_compute 收整层 -> 重建 -> 官方核执行（中间 arena 不写主图链中间）
- [ ] G4 CPU 伪双 device_pool + ids 分组/汇总逻辑
- [ ] G5 数值对齐官方（v2/prefill 对比）

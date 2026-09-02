# Work In Progress - patch 体系手术 + 收尾

> 会话任务清单，边做边更新（多 commit，每步成功即提交）。
> 目标态 = vendored 工作区（features 机制 + frag 全主仓库 + server-context 纯锚点）已编译验证通过；剩余：patch 文件对齐工作区 + ASan 整合 + 文档同步。

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
- [ ] A6 apply.bat（phase 顺序 + --check）—— 可选（update_routeb_patch.js 已能文件级更新）

### B. ASan 子命令整合 build.bat
- [ ] B1 build.bat 加 asan 子命令（vcvars64 探测 + cl.exe + /fsanitize=address /MD + -DSTREAM_MOE_FEATURES=route_b + ninja llama-server）
- [ ] B2 ASAN_BUILD.md 修过时（补 features + 走 build.bat asan 说明 + 改日期）

### C. 文档同步
- [ ] C1 patches/README.md 重写新体系（frag 主仓库 + features + apply 顺序）
- [ ] C2 docs/CHECKPOINT.md 更新状态段（patch 体系 + 构建机制）
- [ ] C3 Makefile 加 features 映射（Linux 若要功能构建）

## 关键纪律
- patch 生成用 `cmd /c "git -C third_party/llama.cpp diff HEAD -- <文件> > patches\x.patch"`（PS 重定向写 UTF-16，禁）
- 手术前快照 + 主仓库 commit（README 叠加铁律）
- vendored 永不 commit，改动靠 patch 记录

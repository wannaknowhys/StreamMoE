# StreamMoE 测试汇总 (TESTING.md)

> 本文档是测试做法的**唯一汇总入口**：单元测试、冒烟、基线回归、导出、bench 笛卡尔积、长程测试的命令、环境与惯例都在这里。
> 分散的专项文档仍然是权威细节：`TEST_FLOW.md`（流程铁律）、`SMOKE_TESTING.md`（冒烟记录）、`BENCHMARK.md`（bench runner 与历史结论）、`CHECKPOINT.md` §3（可跑验证表）、`BUG_TRACKER.md`（bug 与处理方法论）。
> 语言约定：中文说明，代码/命令/路径/标识符保持英文原文。
> 本文档新建于 2026-11-24；“已验证”表示当天在主机上实测或读码确认过，“待核实”表示用户口述或尚未跑通。

---

## 0. 总览：测什么、对照是谁

| 层级 | 入口 | 被测对象 | 对照（control） | 出 PASS/FAIL 吗 |
| :-- | :-- | :-- | :-- | :-- |
| 单元测试 | `build.bat test <tag>` | `src/` 各模块 | ——（自带断言） | 是，ctest |
| 转换器 | `build.bat convert <tag>` + `scripts\verify_convert_matrix.bat` | `stream_moe_convert.exe` | 上游格式 / 已知正确输出 | 转换矩阵脚本出结论 |
| 冒烟 / 单 prompt | `llama-cli` / `llama-server` 手动 | 任一 tag 的二进制 | 人眼读输出 | 否，人眼 |
| 基线回归 | `baseline_regression\run_baseline.bat` | `StreamMoE_dump`（route-B） | `upstream_dump`（见 §1） | 是，`RESULT: PASS/FAIL` |
| 导出 | `tools\run_export.js` / `muliti_test.bat` | 各 flavor | `upstream` engine（同二进制关 route-B）/ `stock-*` | 否，产物落盘后人工/脚本比对 |
| bench 笛卡尔积 | `tools\run_bench.js` | 各 flavor × 模型 × 引擎 × 任务 | `stock-*`（外部 stock 上游二进制） | 否，只记时延 + 状态，流利度人眼查 |
| 长程多轮 | `scripts\run_long_horizon_test.bat` | 指定 tag 的 server | 同次运行的对话转写（人眼） | 否，人眼 |

---

## 1. `main` tag 的状态与对照基线核实（2026-11-24 用户确认 + 读码核实）

### 1.1 `main` 的状态：保留、但用户不管

- **用户决定（2026-11-24 19:09）：`main` 这个 tag 先保留，不关。** 但用户同时说明：纯 CPU 他其实是看 upstream 的，`main` 事实上他不管了；GitHub CI 以后再说。
- 因此 `main` 的状态记为：**保留、但用户不管；纯 CPU 对照是 upstream；CI 待定。**
- 本次任务**没有关 `main`，没有动 `main` 的任何配置**（`build.bat` 里 `main` 的 tag 定义、`:llamalibs` / `:test` / `:convert` 行为均未改；只改了 `:test` 与 `:convert` 两处共用的 CRT flag，见 §4）。

### 1.2 核实结论：测试脚本的对照是 upstream，不是 `main`

2026-11-24 逐项读码核实的结果：**所有成脚本的比较（基线回归、bench、导出）都是 StreamMoE 分支 vs upstream 分支；`main` 没有出现在任何一个 flavor、任何一次 `--flavors` 调用、任何一个 runner 的二进制解析里。**

逐项证据：

| 脚本 / 位置 | 被测（StreamMoE 侧） | 对照（upstream 侧） | `main` 出现吗 |
| :-- | :-- | :-- | :-- |
| `baseline_regression\run_baseline.bat` | `MOE_BIN = build\StreamMoE_dump\llama-build\bin\llama-server.exe` | `UP_BIN = build\upstream_dump\llama-build\bin\llama-server.exe` | 只在 `verify_kl.exe` 缺失重编时链 `build\main\llama-build\ggml\src\*.lib`（链接输入，不是比较对象）；`verify_kl.exe` 已存在，不触发 |
| `tools\run_bench.js` / `tools\run_export.js` | 默认 flavor `StreamMoE`；`binPath(run)` = `build\<bin \| flavor>\llama-build\bin\llama-server.exe` | flavor `upstream_dump` / `upstream_vulkan_dump`；engine `stock-cpu` / `stock-vulkan` 用 `binPath: ${SM_UPSTREAM_BIN}`（外部 stock 上游二进制，见 §2） | 全仓 grep：`run_bench.js`、`run_export.js`、`bench_agent.js`、`baseline_regression\tools\*.js` 均无 `main` 作 flavor/bin 的引用 |
| `tools\run_specs\flavors\`（6 个） | `StreamMoE`、`StreamMoE_dump`、`StreamMoE_latest`、`StreamMoE_dump_dbg`（均为 `hasRouteB: true`） | `upstream_dump`（`hasRouteB: false`，“Stock llama.cpp baseline (prefill_export, CPU only)”）、`upstream_vulkan_dump`（`hasRouteB: false`，Vulkan baseline） | **没有 `main.json`** |
| `tools\run_specs\engines\` | `moe.json`（`--expert-backend --moe-ram-pool ${pool}`）、`place-*.json`（`--dense-placement …`） | `upstream.json`（`--n-gpu-layers 0`，同一 flavor 二进制关 route-B 跑）、`stock-cpu.json` / `stock-vulkan.json`（`binPath: ${SM_UPSTREAM_BIN}`） | 无 |
| `tools\smoke_cli.js` | 默认 `--bin build\StreamMoE\llama-build\bin\llama-cli.exe` | —— | 无 |
| `baseline_regression\tools\verify_places.js` / `check_place_repeat.js` | `--flavor StreamMoE` / `--flavor StreamMoE_dump` | —— | 无 |

### 1.3 `main` 还剩的三个非对照用途（均已确认，暂不动）

1. **文档里的随手命令默认二进制**：`TEST_FLOW.md` 单 prompt 与 `SMOKE_TESTING.md` 冒烟命令写的是 `build\main\llama-build\bin\llama-cli.exe` / `llama-server.exe`；`tools\chat_cli.js` 第 5 行注释示例也是 `build\main\…\llama-server.exe`。这是历史习惯：`main` 是 CPU-only 的 route-B 完整推理版，随手跑最顺。将来清理时可把这些命令改指 `StreamMoE` 或 `upstream_dump`，本次不动。
2. **`verify_kl.exe` 缺失时重编的链接输入**：`run_baseline.bat` 在 `verify_kl.exe` 不存在时用 clang++ 编它，链 `build\main\llama-build\ggml\src\ggml.lib / ggml-base.lib / ggml-cpu.lib`。`verify_kl.exe` 现已存在，不触发；若将来 `main` 被删，这一行要改指别的 tag。
3. **`CHECKPOINT.md` §3 的 build 检查行**：“route-b 完整推理 | `build.bat llamalibs main`”。只是一行 build 检查，不是比较对照。注意该行还写着“当前缺 vk 符号桩，链接失败见 BUG_TRACKER B33”——**已过时**，B33 已于 2026-11-24 随 Phase 1 关闭（见 §4），CHECKPOINT.md 本身尚未同步更新。

---

## 2. 回归环境：`temp\sm_env.bat` 与模型文件

- 模板：`scripts\sm_env.example.bat`（空值）。**本地实值文件是 `temp\sm_env.bat`（`temp/` gitignored，永不提交）**；`private\env.bat` 是另一候选位置，`run_bench.js` / `run_export.js` 按 `temp\sm_env.bat` → `private\env.bat` 顺序加载。
- `run_baseline.bat` 开头 `call "%ROOT%\temp\sm_env.bat"`；`muliti_test.bat` 同样。
- 变量（`scripts\sm_env.example.bat` 所列）：`SM_GEMMA_ORIG` / `SM_GEMMA_V2` / `SM_GEMMA_V1` / `SM_DS_MAIN` / `SM_DS_DRAFT` / `SM_DS_V1` / `SM_OLMOE` / `SM_JSONL_CN` / `SM_JSONL_EN` / `SM_PREFILL_10K` / `SM_OUT_ROOT` / `SM_UPSTREAM_BIN`。`SM_BENCH_OUT` 由 `run_bench.js` 默认 `benchmark/results/bench_<ts>.jsonl`，也可在 `sm_env.bat` 里覆盖。

模型文件核实（2026-11-24，`temp\sm_env.bat` 所指路径逐个 `Test-Path` + 大小确认）：

| 变量 | 所指模型 | 核实结果 |
| :-- | :-- | :-- |
| `SM_GEMMA_ORIG` | `N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | ✅ 存在，15.78 GiB |
| `SM_GEMMA_V2` | `N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf` | ✅ 存在，15.79 GiB |
| `SM_DS_MAIN` | `N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf` | ✅ 存在（第 1 分片 5,257,408 字节 header；分片 2–4 各约 49 GB，分片 5 约 13.5 GB，总计约 162 GB） |
| `SM_DS_DRAFT` | `N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-DeepSeek-V4-Flash-0731-Q8_0-v2.gguf`（草稿模型） | ✅ 存在，10.15 GiB（2026-11-24 核实：`sm_env.bat` 实际指向 v2 文件，本文档原写 `Q8_0.gguf` 已更正） |
| `SM_DS_V1` | `N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf`（与 `SM_DS_MAIN` 同值，5 分片主文件） | ✅ 存在（2026-11-24 19:29 用户指令改指：`UD-Q8_K_XL-v1.gguf` 全盘不存在，改指 5 分片首文件；v1 链路已可跑，尚未跑，见 §8） |
| `SM_OLMOE` | OlmoE 模型 | ✅ 存在，3.92 GiB |
| `SM_UPSTREAM_BIN` | `F:\Dev\computer-use\llama-b10868-bin-win-vulkan-x64\llama-server.exe` | ✅ 存在（stock 上游 llama.cpp b10868 Vulkan 发布版，供 `stock-*` engine 用；不是本仓编的） |
| `SM_OUT_ROOT` | `O:\1\exports` | ✅ 已设 |
| `SM_BENCH_OUT` | `F:\Dev\StreamMoE\benchmark\results\bench.jsonl` | ✅ 已设 |

---

## 3. Build tags（`build.bat llamalibs <tag>`）

`build.bat` 没有 `:build` label；`llamalibs <tag>` 就是编 vendored llama.cpp + 链接 route-B 的入口。主 tag：

| Tag | 含义 | 关键开关 / 用途 |
| :-- | :-- | :-- |
| `main` | CPU-only route-B 完整推理版（`GGML_VULKAN=OFF`） | 历史随手跑 + 单元测试默认 tag 之一；状态见 §1.1，**保留、不管、CI 待定** |
| `StreamMoE` | Release 旗舰（route-B + Vulkan，纯推理，无 export hook） | 对话 / bench 默认 flavor；`smoke_cli.js` 默认 `--bin` 也是它 |
| `StreamMoE_dump` | 测试 + dump 版（route-B + Vulkan + `prefill_export`） | **基线回归的 moe 侧**；`check_place_repeat.js` 默认 flavor |
| `StreamMoE_cuda` | 双后端版（route-B + Vulkan + CUDA，Phase 2 新增） | DeepSeek CUDA0 验证用；`llama-cli` / `llama-server` 在 `build\StreamMoE_cuda\llama-build\bin`；CMake 用 VS2019 cl 作 nvcc host 编译器（见 §12-9）；**2026-11-24 回归 PASS（fix 后重跑，见 §7.3）；Phase 2 已提交并 push** |
| `upstream_dump` | Stock llama.cpp 基线（`prefill_export`，CPU only，`hasRouteB: false`） | **基线回归的 upstream 侧**；`run_baseline.bat` 注释：它不编 `src/` 下的 route-B 引擎 |
| `upstream_vulkan_dump` | Stock llama.cpp Vulkan 基线（`prefill_export` + Vulkan） | flavor 之一，供 bench 对比 |
| `StreamMoE_latest` / `StreamMoE_dump_dbg` | 在研特性版（`STREAM_MOE_LATEST` 默认 ON）/ 诊断 debug 版（`STREAM_MOE_LATEST + STREAM_MOE_TEMP`） | flavor 之一，日常回归不用 |
| `dual` | 不是 tag，是组合动作：依次编 `StreamMoE` + `StreamMoE_dump` + `upstream_dump` | **改 `src/` / `patches/` 后跑它，再跑 `run_baseline`**（`CHECKPOINT.md` §3；`freshness_check.js` 会拦过期二进制） |
| `dbg` | RelWithDebInfo 版（`VULKAN=ON`） | 退出泄漏审计等诊断用（见 `SMOKE_TESTING.md` 2026-09-05 记录） |
| `memwatch` | 内存哨兵版（临时应用 memwatch 补丁后编，独立 `build\memwatch\`） | 只用于“内存被撑爆”诊断，见 `TEST_FLOW.md` §1；用后还原补丁 |

验证命令（`CHECKPOINT.md` §3）：

```bat
build.bat llamalibs dual
build.bat llamalibs main
build.bat llamalibs upstream_dump
build.bat llamalibs StreamMoE_dump
```

---

## 4. 单元测试（`build.bat test`）与 CRT 修复

### 4.1 跑法与覆盖

```bat
build.bat test StreamMoE
build.bat test main
```

- 7 个目标（`build.bat` `:test`，`test_async_dio` 已 DISABLE，见 `CMakeLists.txt`）：`test_moe_loader`、`test_profiler`、`test_scheduler`、`test_slot`、`test_mix_plan`、`test_scatter_plan`、`test_device_cuda`（Phase 2 新增，见 §7.3）。先跑 `check_tensor_data`（扫 93 个文件，0 处裸张量数据拷贝才 PASS），再 `ctest`。
- 前置：`:test` 会先检查 `build\<tag>\llama-build\src\llama.lib` 是否存在，缺失则报错并提示先跑 `build.bat llamalibs <tag>`。
- `test_moe_loader` 覆盖 loader（dense GGUF 解析、真实 MoE GGUF 解析、topo/model builder）；其余 5 个覆盖 profiler / scheduler / slot / mix-plan / scatter-plan；`test_device_cuda`（Phase 2）覆盖 CUDA DMA reader。**`device_iface`（Phase 1 新抽的层）与 Vulkan DMA 目前没有单元测试**——待补（见 §12-6、§12-11）。

### 4.2 CRT 不匹配修复（2026-11-24，已验证）

- 根因：测试目标默认 `/MD`（`MultiThreadedDLL`），链接的是 `/MT`（`MultiThreaded`）编的 `llama.lib`，lld 报 `RuntimeLibrary` 不匹配，`build.bat test StreamMoE` 链接失败。
- 修复（用户 2026-11-24 批准）：`build.bat` 的 `:test`（第 205 行）与 `:convert`（约第 269 行）两处 CMake 配置各加一行，与 `:llamalibs`（第 151 行）对齐：

```bat
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
```

- `git diff --stat`：`build.bat | 2 ++`，恰好两处插入；CRLF 保持。根 `CMakeLists.txt` 没有 `MSVC_RUNTIME` 处理，所以 `-D` 生效。
- 验证结果（2026-11-24 19:03，日志 `temp\crt-test-streammoe.log` / `temp\crt-test-main.log` / `temp\crt-convert-streammoe.log`，汇总 `temp\crt-verify-summary.log`）：
  - `build.bat test StreamMoE`：**exit 0**，`test_moe_loader.exe` 重新链出（`build\StreamMoE\bin\test_moe_loader.exe`，2026-11-24 19:03:54），ctest **6/6 通过**（`test_moe_loader` 0.33s 等，`100% tests passed`）。
  - `build.bat test main`：**exit 0**，ctest **6/6 通过**。但 `main` 的测试二进制**没有重编**（`build\main\bin\test_moe_loader.exe` 仍是 2026-10-26 的）：`main` 的 CMakeCache 自 2026-08-29 起就已带 `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`，本次 configure 后 ninja `no work to do`——CRT 不匹配从来只影响 `StreamMoE` tag，`main` 本来就是对的。另注意：Phase 1 没改任何测试目标的源文件，所以 `main` 的旧二进制对其单元仍有效；但它不覆盖 Phase 1 的新代码。
  - `build.bat convert StreamMoE`：**exit 0**，`build\StreamMoE\bin\stream_moe_convert.exe` 链出（2026-11-24 19:03:58）。
  - Phase 2 单元测试（2026-11-24 ~21:02，日志 `temp\p2wrap-test-*.log`）：**main 7/7、StreamMoE 7/7、StreamMoE_cuda 7/7**，新 `test_device_cuda` 在 CUDA tag 上 2.75s（真在 P100 上跑了），CPU-only tag 上 0.03s。注意 19:03 的 6/6 是 Phase 2 之前（`test_device_cuda` 尚未加入）。
- 未修的范围：`:asan` 段不设该 flag——它只配置 vendored 树自身（MSVC cl.exe），没有跨 configure 链接，不受此问题影响；如将来 asan 也要链根项目目标再议。
- 关联 bug 关闭：`docs\BUG_TRACKER.md` **B33 已于 2026-11-24 标为 FIXED**（`build.bat llamalibs main` 链接失败：route-B 无条件引用 `stmoe_vk_*`，`main` 的 `GGML_VULKAN=OFF` 不编 Vulkan——Phase 1 `device_iface` 按 kind 分发，`device_vulkan.cpp` 仅在 `GGML_VULKAN=ON` 时编译；链接已通过，未提交，HEAD `b308b03`）。注意 B33 这个 ID 在 `BUG_TRACKER.md` 第 46 行还另有一条已 FIXED 的内存问题（repack extra bufts），不要混淆；本次只改了第 81 行。

---

## 5. 转换器（`build.bat convert`）

```bat
build.bat convert StreamMoE
build.bat convert main
```

- 产物：`build\<tag>\bin\stream_moe_convert.exe`，用法见 `CHECKPOINT.md` §3：`stream_moe_convert.exe -m <model> -o <out> --format v3`。
- 转换矩阵：`scripts\verify_convert_matrix.bat <workdir>`（C++ 转换器；源默认 `SM_GEMMA_ORIG`）。

---

## 6. 冒烟与单 prompt（最快验证，改动后先走这里）

铁律（`TEST_FLOW.md` §0）：**先单 prompt 验证，再整轮评测；长任务交给用户手动运行的 `.bat`。**

冒烟模型（`SMOKE_TESTING.md`，2026-08-27 起）：

| 用途 | 模型 | 说明 |
| :-- | :-- | :-- |
| MoE | `N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf` | Gemma 4 26B A4B，每 token 激活 4 个专家，验证 MoE 前向/路由 |
| Dense | `F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | 2B dense，快速验证基本链路 |

```bat
rem MoE（gemma-4-26B-A4B）；--no-warmup 跳过空跑预热（MoE 下避免启动时冷盘专家装载）
build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --fit off --no-warmup --no-webui

rem Dense（Qwen3-VL-2B）
build\main\llama-build\bin\llama-server.exe -m F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --no-warmup --no-webui
```

然后 `/health`（200）就绪后 POST `/v1/chat/completions`。注意上面命令用的是 `build\main` 的二进制（历史习惯，见 §1.3；对照仍是 upstream，不是 main）。

单 prompt（DeepSeek 大模型，`TEST_FLOW.md` §2）：

```bat
build\main\llama-build\bin\llama-cli.exe -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" --expert-backend --moe-ram-pool 71680 -c 8192 -t 16 --fit off --no-warmup -p "Say hello in one short sentence." -n 32
```

- `-n` 先给 16~32；首 token 慢（模型在 N: 冷页拉取，N: 是 USB 转接 NVMe）。大 MoE 一律 `--moe-ram-pool 71680`（70GB 池），不用小池。
- 交互模式的 cli 须加 `-st`（single-turn）才会单轮结束退出；不加 `-st` 一直等输入不是 bug（2026-09-05 对照结论）。
- 跑批前后 `taskkill /F /IM llama-server.exe`，杜绝僵尸占端口。

VRAM 池驻留验证（`CHECKPOINT.md` §3，Gemma 全量专家进 VRAM；池 1024 触发 demote）：

```bat
llama-server -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf --prefill-from baseline_regression\baseline\upstream_129\tokens_id.bin --export-dir <dir> -c 2048 -t 16 --expert-backend --moe-expert-pools RAM:8192,Vulkan0:4096 --fit off --no-warmup
```

产物对 `moe_129_8192_vk` 应 IDENTICAL（vk 数值 flavor 见 §7）。

---

## 7. 基线回归（`baseline_regression\run_baseline.bat`）——CHECKPOINT C1:Vulkan0 回归入口

这是**改代码 + 编译后必跑**的数值守门：129-token prefill-from 三组导出对已知正确基线，+ per-token KL + kv_cos，直接出 PASS/FAIL。

```bat
rem CPU flavor（默认基线 moe_129_8192）
baseline_regression\run_baseline.bat

rem Vulkan flavor（StreamMoE_dump 默认；CHECKPOINT C1:Vulkan0 回归用这个）
baseline_regression\run_baseline.bat baseline_regression\baseline\moe_129_8192_vk

rem 第二个可选参数是 verify_dir，默认 baseline_regression\temp
```

### 7.1 八步（脚本头注释 + 正文逐行确认，2026-11-24）

| 步 | 做什么 | 判定 |
| :-- | :-- | :-- |
| 前置 | `call temp\sm_env.bat`；检查 `MOE_BIN`（`build\StreamMoE_dump\…\llama-server.exe`）与 `UP_BIN`（`build\upstream_dump\…\llama-server.exe`）存在；`tools\freshness_check.js` 查两二进制新鲜度（moe 侧含 `src/` + `patches/`，upstream 侧 `--no-src` 只查 `patches/` + 脏 vendored 树）——**过期直接 exit 1，先跑 `build.bat llamalibs dual`** | 缺二进制 / 过期即停 |
| [1/8] | moe prefill-from（route-B，8GB 池）：`--prefill-from baseline\upstream_129\tokens_id.bin --export-dir <OUT>\moe -c 2048 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup`，模型 Gemma v2 | 失败即停 |
| [2/8] | upstream prefill-from：同 tokens，Gemma v1，无 route-B | 失败即停 |
| [3/8] | embd/hidden/KV 对基线：`verify_prefill.js`（moe vs 基线 moe；upstream vs `baseline\upstream_129`）+ `verify_expert_history.js` | CPU flavor 要求字节 IDENTICAL；**vk flavor 字节 IDENTICAL 达不到**（host memory 形式不同），改走 cos 门 `--cos-floor 0.99 --min-ratio 0.9`；vk 下 expert-history 翻转只报告、不判 FAIL |
| [4/8] | per-token KL（基线 upstream vs 新 moe，`verify_kl.exe`，只报告） | 阈值 CPU 1.0 / vk 4.5；moe-vs-upstream KL 是后端固有噪声（路由翻转），不是 bug，见 `BACKEND_DIVERGENCE_ANALYSIS.md` |
| [5/8] | `kv_cos.js`（基线 moe vs 新 moe） | 期望全部 ~1.0（`cos_min >= ~0.999`）；NaN 格视为不可比 |
| [6/8] | `div_match.js`（moe vs 基线 moe） | 若 DIVERGED，看每个高散 token 是否都由 expert flip 解释（已知 gemma-4 gate 噪声），还是有未解释的散度 |
| [7/8] | olmoe place 回归：`verify_places.js --flavor StreamMoE --models tools\run_specs\models\olmoe.json --tasks tools\run_specs\tasks\hi.json --baseline baseline_regression\baseline\olmoe_hi_baseline.txt --temp-override 0`；[7b] `check_place_repeat.js --flavor StreamMoE_dump --engines place-c1-exp2,place-c1c2-exp1,place-c1c2-exp2,place-c1c2-exp5 --runs 3 --turns 2`（重复一致性门；exit 2 只警告） | `SM_OLMOE` 未设 / 文件不存在则整步跳过 |
| [8/8] | `node tools\smoke_cli.js`（llama-cli 混合池冒烟，默认 `build\StreamMoE\…\llama-cli.exe`） | 同上，`SM_OLMOE` 缺失则跳过 |

- 结论只看最后一行：`RESULT: PASS - all baseline and place regression checks passed` 或 `RESULT: FAIL - see above`。**注意脚本 `exit /b 0` 恒为 0**，不能只看退出码，要 grep `RESULT:`。
- 基线目录：`baseline_regression\baseline\moe_129_8192`（CPU）、`moe_129_8192_vk`（Vulkan）、`upstream_129`（upstream 固定对照）、`olmoe_hi_baseline.txt`。

### 7.2 本次试跑（CHECKPOINT C1:Vulkan0）

- 前置已做：2026-11-24 19:08 `build.bat llamalibs dual` **exit 0**（三 tag 全部重编，`[+] Triple build completed successfully`），二进制已新鲜。
- 试跑命令（2026-11-24 19:11 后台启动，日志 `temp\run-baseline-vk.log`，汇总 `temp\run-baseline-summary.log`）：

```bat
baseline_regression\run_baseline.bat baseline_regression\baseline\moe_129_8192_vk
```

- 状态：**✅ PASS（2026-11-24 19:22:38）**：`RESULT: PASS - all baseline and place regression checks passed`。分项：[3/8] moe 对基线 PASS（cos 门；expert history 3460/61920 entries differ，vk 路由噪声，仅报告），upstream 对基线 IDENTICAL；[4/8] KL PASS（阈值 4.5，max_KL 0.4028）；[5/8] kv_cos rows=258 comparable=3870 cos_min=0.990300；[6/8] 33 个高散 token 中 1 个未被 expert flip 解释（脚本只报告，不判 FAIL，值得留意）；[7/8] olmoe place 回归通过；[7b/8] 重复一致性 PASS（with 4 warnings）；[8/8] smoke 通过。完整日志 `temp\run-baseline-vk.log`，汇总 `temp\run-baseline-summary.log`。
- 边界说明：这次回归覆盖的是 **Gemma C1:Vulkan0**（`moe_129_8192_vk` 基线 + [7/8] olmoe place-c1 系列）。**DeepSeek C1:Vulkan0 跑不了**，不是缺模型文件的问题：DeepSeek dense C1 是 11.66 GB，RX 590 只有 8G 装不下；而 P100 在 TCC 模式下没有 Vulkan（2026-11-24 已定论，不再查）。DeepSeek 的 C1 已改道 CUDA0 验证通过（见 §7.3），Vulkan0 本身仍跑不了。

### 7.3 Phase 2 回归：Vulkan0 退化 FAIL（2026-11-24 22:16–22:39）→ 修复后重跑 PASS

Phase 2 内容（2026-11-24 19:42 用户开工；异步 DMA、双后端、commit+push 一条龙）：异步 DMA token 接口（read/write 返回 `pool_dma_token_t` + wait/poll）、`host_mapped` 标志（Vulkan host-mapped 直读 vs CUDA device 地址）、`settle_loaded` / `begin_htod` / `drain_htod`、move worker“先 enqueue 全部列、再 wait 一次”、非 host-mapped（CUDA）槽位强制 staging + HtoD、直接 DIO 禁用；新文件 `src/backend/device_cuda.cpp`、`tests/test_device_cuda.cpp`；双后端 tag `StreamMoE_cuda`（Vulkan+CUDA）；`stmoe_routeb_args.frag` 里 `STMOE_VK_FORCE_MAX` 只对 Vulkan 前缀 token 生效；CUDA0 走通用 `--moe-expert-pools`，不加新 flag。构建：2026-11-24 20:21 `llamalibs StreamMoE_cuda` 成功（日志 `temp\build-cuda.log`；CMake 找到 CUDAToolkit 12.6.85，host 编译器 MSVC 19.29.30159.0 = VS2019 cl）。单元测试：main 7/7、StreamMoE 7/7、StreamMoE_cuda 7/7（§4.2）。

DeepSeek CUDA0 验证（命令：`build\StreamMoE_cuda\llama-build\bin\llama-cli.exe --expert-backend --dense-placement C1:CPU --moe-expert-pools RAM:32768,CUDA0:<MB> -p hi -st -m <SM_DS_MAIN> -n 32 --temp 0`）：

- 21:38 首跑（日志 `temp\p2wrap-ds-cuda.log`）：模型加载成功（deepseek4，43 层，256 experts/层，5 分片；dense 13.6853 GB，expert 137.062 GB），CUDA0 池子 2040/2048 MB 已分配，但 agent 回合结束时后台进程被杀，无生成 token、无报错（Application 事件日志与 WER 均无 llama-cli 崩溃记录），不作数。
- 21:55 重跑 CUDA0:2048（日志 `temp\p2wrap-ds-cuda2.log`）：44s 后 exit -1073740791（0xC0000409），`CUDA error: out of memory`（`ggml-cuda.cu:107`，`ggml_cuda_kernel_can_use_pdl`）。当时 P100 已用 13,539/16,384 MiB（科研 rollout 占 13,530 MiB），池子 2040 MB 分配成功后 ggml-cuda 自身开销 OOM。
- 22:00 改 CUDA0:1024 重跑（日志 `temp\p2wrap-ds-cuda3.log`）：**exit 0，2:41**，池子 1020/1024 MB（80 slots），生成 “Okay, the user just said "hi". This is a very simple and common greeting. ...” （32 token 上限截断），0.3 t/s。**CUDA0 路径验证通过。**
- DMA 证据说明：模型运行日志里没有逐次 DMA trace（grep dma/DtoH/HtoD/cudaMemcpy/drain/transfer/move 只有池子/槽位分配行）。DMA 正确性靠 `test_device_cuda`（CUDA tag 上 2.75s，真在 P100 上跑）+ 端到端出 token 证明；不要声称有日志级 DMA trace。

测量条件（用户 2026-11-24 21:18 提醒）：**P100 限功 150W + 并发科研任务**（Qwen rollout `step2-rollout-100-tasks.py`，占 P100 约 13.5 GiB），本次验证所有性能数字（tok/s、DMA/加载耗时、lag）都不干净，只认正确性（出 token、KL、PASS/FAIL）；时间、速度、lag 数字只记作“在 150W 限功 + 并发负载下测得”。**lag 数字不得用于主动调度设计的 T_io，待干净环境重测。**

Gemma 回归 FAIL（2026-11-24 22:16–22:39，耗时 22:41，日志 `temp\run-baseline-vk-p2.log`，done `temp\p2wrap-gemma-done.txt`）：`RESULT: FAIL - see above`（脚本 exit 恒为 0，看 `RESULT:` 行）。分项：[3/8] PASS（cos 门 121/129 = 93.8%；expert history 3460/61920 differ，首个 #23，**与 19:22 PASS 完全相同**）；upstream 对基线 IDENTICAL，PASS；[4/8] KL PASS（阈值 4.5，max_KL 0.4027535，与 19:22 的 0.4028 相同）；[5/8] kv_cos rows=258 comparable=3870 cos_min=0.990300（**与 19:22 相同**，仅报告）；[6/8] 33 个高散 token 中 1 个未被 expert flip 解释（同 19:22，仅报告）；[7/8] olmoe place 回归“通过”（12 引擎都跑完，11 处 minor fp 精度差异）——**但该门不查流利度，属于假通过**，其输出实际已是退化文本（见下）；[7b/8] **FAIL（4 引擎）**，全部 `too few model lines (18)`；[8/8] **smoke FAIL**，llama-cli 输出 `"2020"×16`，4.09s。

退化详情（OlmoE，StreamMoE / StreamMoE_dump 二进制，Vulkan0 池）：

- [7b/8] 最新输出 `temp\bench_outputs\2026-11-24T22-38-54\01_StreamMoE_dump__olmoe__place-c1c2-exp5__en.txt`：Turn 1 `[Assistant]` 整行 `2018` 重复，Turn 2 `[Assistant]` 整行 `,` 重复。
- [7/8] 的 `baseline_regression\temp\olmoe_places\StreamMoE\olmoe\place-c1-exp2\hi\chat.json`：对 `Hi` 的回复是整段 `2018` / `2019` 重复。
- `18` 行的含义：每轮回复只剩 1 行重复 token（+ prompt 行），低于 `check_place_repeat.js` `validateText` 的 20 行下限（`modelLines < 20` 即 FAIL）。该门先校验、再比对一致性——4 引擎都在校验步失败，未走到一致性比对。

二分定位（2026-11-24 ~22:41，日志 `temp\smoke-repro-{A,B,C}.log`，启动器 `temp\smoke-repro.ps1`）：同一命令 `llama-cli --expert-backend --moe-expert-pools <池子> -p hi -st -m <SM_OLMOE> -n 16 --temp 0`，三组对照：

| 组 | 二进制 | 池子 | 结果 |
| :-- | :-- | :-- | :-- |
| A | `StreamMoE`（Phase 2，22:07 编） | `RAM:8192,Vulkan0:5120` | ❌ `"2020"×16`（复现回归失败） |
| B | `main`（Phase 1，18:33 编，CPU-only） | `RAM:13312` | ✅ `Hello! How can I help you today? ...` |
| C | `StreamMoE`（Phase 2，22:07 编） | `RAM:13312`（纯 RAM） | ✅ `Hello! How can I help you today? ...`（与 B 相同） |

结论：**Phase 2 代码 + Vulkan0 专家池 → 退化生成；同一二进制纯 RAM 池正常；Phase 1 二进制正常；19:22 同配置 Vulkan0 曾 PASS，期间 Vulkan/驱动/模型均未变，唯一变量是 Phase 2 代码。故障定位在 Phase 2 改动的 Vulkan 专家池路径（异步 DMA 接口 / move worker / settle 流程），不是 CUDA 路径**（DeepSeek CUDA0 正常）**，不是环境**（并发负载只影响速度，不改变 logits；且 A/C 同环境结果相反）。补充：A/C 日志均无 dma/fail/error 行——**静默失败，无报错，exit 0**；A 的池子正常分配（Vulkan0 3720/5120 MB，2048 slots，2 device regions）。prefill 导出路径（[3/8]–[6/8]）不受影响，说明初始加载/导出没坏，坏的是生成期按需加载或 decode 路径——待修时确认。

**后果：Phase 2 未 commit、未 push，提交链已停。** 修复 Vulkan 路径后必须重跑 `run_baseline.bat baseline_regression\baseline\moe_129_8192_vk` 全套并 PASS 才能提交；不要以“prefill 通过”为由放行。

**根因（B40，2026-11-24 23:14 定位，见 `docs/BUG_TRACKER.md` B40）：** Phase 2 的 `src/server/route_b_inject.cpp` 把任何无 host map 的 device region 都注册成 DMA-only（`host_mapped=false`、base 假指针 `0x1000`）。这个假设只对 CUDA 成立——CUDA 的 writer 是 `cudaMemcpyAsync`，写的是设备地址。但这台 RX 590 的 Vulkan buffer 根本没有 host map（`stmoe_vk_buffer_host_ptr` 返回 null，分配回落到纯 device-local 显存），而 Vulkan 的 `dma_write` 只是 host-map memcpy——无 map 时 0 字节写出，却返回 null token 表示"已完成"。于是所有进 Vulkan 槽的加载（`begin_htod`）和 r2v 搬迁静默空完成，槽仍标 READY，计算读空权重，生成退化成 `"2020"×16`，全程无报错、exit 0。19:22 能 PASS 是因为 Phase 1 代码直接跳过无 host map 的 region（纯 RAM 跑）；Phase 2 改动的是共用的 `device_iface` 异步 token 接口与 scheduler move/settle 流程，Vulkan 侧只适配了接口——frag（`stmoe_routeb_vk_dma.frag` / `stmoe_routeb_vk_hostmap.frag`）本身没动过。

**修复（两处，2026-11-24 23:14）：**

- `src/server/route_b_inject.cpp`：无 host map 时，只有 `kind==CUDA` 才注册 DMA-only region；其他后端（Vulkan）跳过该 region 并打印警告，恢复 Phase 1 行为（池子跑 RAM）。
- `src/backend/device_vulkan.cpp`：`dma_write` 无 host map 时 `fprintf(stderr, ...)` 大声报错（以前是 `if (base)` 静默跳过），+ `#include <cstdio>`。

修复后重编：`build.bat llamalibs StreamMoE` BUILD_EXIT=0（`build\StreamMoE\llama-build\bin\llama-cli.exe` 23:14:51）；`build.bat llamalibs dual` 三 tag 重编，回归用二进制新鲜（dump server 23:16:59）。

**注意：这次修复只是恢复/过渡，不是最终形态。** 它让 Vulkan0 池子回到 Phase 1 的"跳过 region、跑 RAM"行为，专家仍然没上卡。最终形态是真正的 Vulkan HtoD 写路径（见 §12-12）：给 DMA frag 加 `stmoe_vk_dma_write`，包 vendored 的 `ggml_vk_buffer_write`，让 Vulkan0 区成为真正的 DMA-only 区。

**验证——A/B/C 重跑（2026-11-24 ~23:14，日志 `temp\smoke-repro2-{A,B,C}.log`）：** 同 `smoke-repro.ps1` 的三组对照全部通过，均为 `Hello! How can I help you today? Is there something you would like to ...`。A 明确打印新警告（两次）"route B: device pool 'Vulkan0' has no host map and no DMA-only write path - region skipped, pool runs on RAM"，池子回到 1024 slots / 0 device regions。

**验证——Gemma 回归重跑 PASS（2026-11-24 23:17 起，耗时 8m32s，日志 `temp\run-baseline-vk-fix.log`，GEMMA_EXIT=0）：** 第 1268 行 `RESULT: PASS - all baseline and place regression checks passed`。分项：[3/8]–[6/8] 数字与 19:22 PASS 完全相同（KL max_KL 0.4027535，kv_cos cos_min=0.990300，expert history 3460/61920 differ，1/33 未解释）；[7/8] olmoe place 12 引擎全跑，7 处 minor fp 精度差异；[7b/8] **PASS，带 4 warnings**（4 个引擎 output hash 变化，但 3 次运行各自一致——属于基线漂移类警告，非 FAIL）；[8/8] smoke **PASS**，3.86s，`Hello! How can I help you today?`。

**验证——DeepSeek CUDA0 重跑 PASS（日志 `temp\p2wrap-ds-cuda4.log`，DS_EXIT=0，2m20s）：** CUDA0 池 1020/1024 MB，`hi` 生成正常回复。0.4 t/s 是污染数字（P100 限功 150W + 并发科研任务占卡），只记正确性，不作性能基线。

**结论：Phase 2 已提交并 push（见 §12-10）。** 本次回归的 FAIL 记录保留在上文作历史，BLOCKER 已关闭（§12-11）。

---

## 8. 导出（`tools\run_export.js` / `muliti_test.bat`）

`run_export.js` 与 `run_bench.js` 共用 `tools\run_specs\` 的 flavor × 模型 × 引擎 × 任务笛卡尔结构；区别是它把 prefill 导出落盘（`SM_OUT_ROOT`，默认 `O:\1\exports` 下按 `<flavor>/<model>/<engine>/<input>` 分目录），供后续离线比对。

```bat
rem DeepSeek：upstream + moe 两个 engine，prefill10000 / cn / en 三个任务
node tools\run_export.js --models tools\run_specs\models\deepseek.json --engines tools\run_specs\engines\upstream.json,tools\run_specs\engines\moe.json --tasks tools\run_specs\tasks\prefill10000.json,tools\run_specs\tasks\cn.json,tools\run_specs\tasks\en.json
```

- `muliti_test.bat`（仓根，注意拼写就是 `muliti`）就是上面这条 + 第二组 deepseek-v1（moe engine，同三个任务）。**第二组此前跑不了**：`SM_DS_V1` 文件不存在（见 §2）；2026-11-24 19:29 已改指 5 分片主文件，路径已通，但**尚未跑过**。
- `temp\run_export_win.bat` **不存在**：`CHECKPOINT.md` §3 提到用 `agy-run` 在前台窗口跑它（用户实时看），但该文件已不在仓里——需要时按 `muliti_test.bat` 的写法重建，不要当它存在。
- 引擎 `upstream.json` 的含义：`--n-gpu-layers 0`，即**同一 flavor 二进制关掉 route-B 跑**，不是换一个二进制；真换二进制对照用 `stock-*` engine（`binPath: ${SM_UPSTREAM_BIN}`）。

---

## 9. Bench 笛卡尔积（`tools\run_bench.js`）——特别慢、慎用

```bat
node tools\run_bench.js --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]> [--flavors <spec[,...]>] [--port N] [--health-timeout S] [--out FILE]
```

- 四轴：**flavors × models × engines × tasks**。默认 flavor `StreamMoE`；不指定 `--flavors` 就只跑它。先看规模再跑：

```bat
node tools\run_bench.js --models ... --engines ... --tasks ... --dry-run
```

（`--dry-run` 只打印 run 清单，不执行。）
- 规格数量（2026-11-24 清点）：flavors 6 个（见 §3 表）、models 6 个（`deepseek`、`deepseek-v1`、`gemma`、`gemma-v1`、`gemma-v2`、`olmoe`）、engines 21 个（含 `moe`、`moe-temp0`、`upstream`、`place-*` 12 个、`stock-cpu`、`stock-vulkan`、`upstream-temp0*` 4 个）、tasks 9 个（`bench`、`bench_long`、`cn`、`en`、`en_long`、`en_long2`、`hi`、`prefill10000`、`prefill3000`）。全笛卡尔积是 6×6×21×9 量级——**用户原话：特别慢，慎重。绝对不要不加限定直接跑全量。**
- 任务种类：`single`（cold → warmup 丢弃 → repeat 取中位数，`/completion` 且 `cache_prompt=false`，每次 repeat 都重 prefill）；`jsonl`（多轮串联，`/v1/chat/completions` 且 `cache_prompt=true`）。
- 端口默认 8994（`SM_BENCH_PORT`）；health 超时默认 900s；结果 JSONL 默认 `benchmark/results/bench_<ts>.jsonl`（`SM_BENCH_OUT` 可覆盖，当前 `sm_env.bat` 指到 `benchmark\results\bench.jsonl`）。
- **流利度检查是人工的，没有自动化**：每个 run 的完整输出文本存在 `temp\bench_outputs\<ts>\<seq>_<flavor>__<model>__<engine>__<task>.txt`（含 `[Assistant]` 全文），JSONL 只记时延与 OK/FAIL。所谓“逐个查输出是否是完善流利的话语”，就是**逐个打开这些 txt 人眼读**。失败的 run 记为 FAIL，批次继续。
- 用户惯例（口述，已记录，尚未逐条实测验证）：
  - **OlmoE 跑单个 task，但全部 memory placement**（place-* 全套）。
  - **Gemma 和 DeepSeek 只跑单个 placement、单个任务，先确定基本可用**，再扩大。
  - 详见 `docs\BENCHMARK.md`（runner 说明 + 2026-09-09 olmoe、2026-10-26 TMR/Vulkan 启动探针：RX 590 上每次 graph 提交约 0.17 ms + 每节点约 51 us）。

---

## 10. 长程多轮（`scripts\run_long_horizon_test.bat`，用户手动跑）

```bat
scripts\run_long_horizon_test.bat en        rem 英文 10 轮（build tag 默认 main）
scripts\run_long_horizon_test.bat zh memwatch  rem 指定 build tag
```

- 行为：杀掉残留 server → 起 tag 的 `llama-server`（端口 8992，DeepSeek 162GB 模型，`--moe-ram-pool 71680 -c 8192 -t 16 --temp 1.0 --top-p 0.95 --n-predict 384 --cache-type-k q8_0 --cache-type-v q8_0 --expert-backend --fit off --no-warmup`）→ 等 `/health` 就绪（120s）→ `node tools\bench_agent.js` 按 `benchmark\prompts\long_horizon_prompts[_zh].jsonl` 全轮串联在同一个 server 上跑 → 杀 server。
- 产物：`benchmark\results\conversation_real_<en|zh>.txt`（完整对话转写，人眼查流利度）/ `profile_real_<en|zh>.jsonl`（逐轮遥测）；server 日志 `build\<btag>\server_<tag>.log`。
- **用户职责**：盯内存；异常立即关窗 / `taskkill /F /IM llama-server.exe`。`TEST_FLOW.md` 铁律：长任务交 `.bat`，由用户自己双击运行。

---

## 11. 稀有 bug 的处理流程

- 登记：`docs\BUG_TRACKER.md`（表格式：ID / 状态 / 问题 / 位置 / 处置；状态图例 `OPEN` / `FIXED` / `WONTFIX` / `GONE` / `SUPERSEDED` / `PARTIAL`）。注意 B33 这个 ID 在文件里出现两次（第 46 行是已 FIXED 的内存问题，第 81 行是本次关闭的链接问题），编辑时按行定位，别改错。
- 数值类 bug 方法论（`BUG_TRACKER.md` §0，2026-08-26 立，8 条铁律）：**先打 log 后思考；以事实/测试为基准，不做无限猜想**；全量 log + 全量 dump（`fopen` 直写文件，不用 stderr）；二分法缩小范围；避开 PowerShell 陷阱（PS 5.1 的 `2>` 会把 stderr 转 UTF-16 ErrorRecord 并截断长行，用 cmd `.bat` 重定向或程序直写）；锁定函数后逐行打 log；能脱离主程序就脱离（独立诊断程序，如 `temp/iso.cpp`）；单点执行优先于猜想；修复后复用同一套 log/dump 逐项对照，再端到端测试，最后清理诊断代码。实战记录见 `docs\DEBUG_DELEGATION.md`。
- 内存诊断（`TEST_FLOW.md` §1）：遇“内存被撑爆”，按 `patches/README.md` 临时应用 memwatch 补丁 → `build.bat llamalibs memwatch && build.bat build memwatch`（独立 `build\memwatch\`，不碰 `build\main\`）→ 单 prompt 复现 → 看 `%TEMP%\memwatch_<pid>.log`（WS 大而 PRIV 小 = file-backed 页驻留；PRIV 也大 = 真实私有拷贝/泄漏）→ 用后还原补丁，确认 `git -C third_party/llama.cpp status` 干净。
- 最近实例：B33（`main` 链接失败）由 Phase 1 `device_iface` 消除，2026-11-24 关闭；CRT 不匹配由 §4.2 的两行 `-D` 修复，同日验证。

---

## 12. 待确认 / 待核实项

| # | 事项 | 状态 |
| :-- | :-- | :-- |
| 1 | `main` tag 去留 | ✅ **已确认：保留，不关；用户不管；纯 CPU 对照是 upstream；GitHub CI 待定**（2026-11-24 19:09）。将来若清理，§1.3 的三处（随手命令、`verify_kl.exe` 链接输入、CHECKPOINT §3 行）要同步改 |
| 2 | `SM_DS_V1` 模型文件不存在 | ✅ **已解决（2026-11-24 19:29）**：`UD-Q8_K_XL-v1.gguf` 全盘不存在，用户指令改指 5 分片主文件（与 `SM_DS_MAIN` 同值）；v1 链路已可跑，尚未跑 |
| 3 | DeepSeek C1:Vulkan0 回归 | **已改道 CUDA0 验证通过**（2026-11-24 22:00，见 §7.3）：C1:CPU + `RAM:32768,CUDA0:1024`，exit 0，出 token；2048 MB 版因并发科研任务占显存 OOM，1024 MB 重跑通过。Vulkan0 本身仍跑不了（硬件原因不变） |
| 4 | `CHECKPOINT.md` §3 第 197 行“B33 待补” | 已过时（B33 已关），文档未同步；另 `temp\run_export_win.bat` 不存在，§3 引用已失效——两处待顺手更新 |
| 5 | `:asan` 不设 `CMAKE_MSVC_RUNTIME_LIBRARY` | 已记录（§4.2），本次范围外；如 asan 也要链根项目目标再议 |
| 6 | `device_iface` 无单元测试 | 待补；Phase 1 只做了集成验证（`llamalibs main` / `StreamMoE` 链接 + 6/6 单元测试） |
| 7 | `run_bench.js` 全笛卡尔积规模 | 未实测；跑全量前先 `--dry-run` 看清单，并得用户明确同意 |
| 8 | 用户惯例“OlmoE 单 task 全 placement；Gemma/DeepSeek 单 placement 单任务” | 用户口述，已记录（§9）；尚未逐条实测验证 |
| 9 | Phase 2 前置：nvcc host 编译器 | ✅ **已解决并验证（2026-11-24 19:10–19:11）**：VS2019 vcvars（v16.11.63）+ nvcc 全路径 + `-ccbin` VS2019 cl（MSVC 14.29.30133）`-arch=sm_60` 编 `test_sm60.cu` exit 0，P100 上跑 exit 0（`KERNEL OK`）。配方在 `F:\Dev\cuda-toolkit-install\NVCC_RECIPE.md`（+ `compile-vs2019.bat`）；Phase 2 CMake 用 `-DCMAKE_CUDA_HOST_COMPILER=<VS2019 cl 路径>`，`CMAKE_CUDA_ARCHITECTURES` 含 60。铁律：VS2019 cl 只给 nvcc 当 host 编译器，主构建仍用 clang-cl，asan 仍用 VS 18；**不要再试 `-allow-unsupported-compiler` + VS 18**（cudafe++ 0xC0000005 已证实不可行） |
| 10 | Phase 1 / Phase 2 提交状态 | ✅ **Phase 1 已提交**（`201b3fc` "Phase 1: backend-agnostic device_iface (CPU/Vulkan) + B33 fix"）；✅ **Phase 2 已提交并 push**（"Phase 2: async DMA device_iface (CPU/Vulkan/CUDA) + Vulkan no-host-map fix"，见 §7.3） |
| 11 | **Phase 2 回归 FAIL：Vulkan0 专家池退化生成（BLOCKER）** | ✅ **已修复并验证，CLOSED**（2026-11-24 23:14 fix，23:26 回归 PASS）：根因见 B40（`route_b_inject.cpp` 把无 host map 的 Vulkan 区误注册成 DMA-only；`dma_write` 零字节静默完成）。修复：仅 CUDA 注册 DMA-only，无 host map 的其他后端跳过 region + 警告；`device_vulkan.cpp` 的 `dma_write` 无 map 时大声报错。A/B/C 全过，Gemma 回归 PASS，DeepSeek CUDA0 重跑 PASS。剩余工作：真正的 Vulkan HtoD（§12-12）；当前 skip 只是过渡，不是最终形态 |
| 12 | 真正的 Vulkan HtoD 写路径（最终形态） | 🔴 **OPEN，后续工作**：当前 skip-region 修复只是**恢复/过渡**（Vulkan0 池子回落到 RAM，专家没上卡），**绝不是最终方案**。最终形态：在 `patches/route-b/common/stmoe_routeb_vk_dma.frag` 里加 `stmoe_vk_dma_write`，包 vendored 的 `ggml_vk_buffer_write`（`ggml-vulkan.cpp:8512` 同步，async 版 `:8470`；非 host-visible 时走 transfer-queue staging + fence 等待，和读路径完全对称；frag include 点 `:18877` 在其后，可见性没问题）；`device_vulkan.cpp` 的 `dma_write` 改调它；`route_b_inject.cpp` 里 Vulkan 无 host map 的区不用跳过，可注册成真正的 DMA-only 区，专家真正住进 RX 590。注意：minigraph 的 H2D（`minigraph_exec.cpp` pinned staging + `dispatch_relay_batch`）早就存在，那是跨设备 relay（搬 activation/张量）的路；缺的只是专家池的 DMA 写，不要写成"写路径从来没做过" |

---

## 附：验证状态（2026-11-24）

| 时间（EST） | 事项 | 结果 |
| :-- | :-- | :-- |
| 19:03 | CRT 修复验证（§4.2） | `test StreamMoE` 6/6、`test main` 6/6、`convert StreamMoE` 链出，全部 exit 0 |
| 19:03 | B33 关闭 | `BUG_TRACKER.md` 第 81 行 → FIXED；`git diff` 仅该行；文件保持 LF |
| 19:08 | `build.bat llamalibs dual` | exit 0，三 tag 全部重编 |
| 19:11–19:22 | CHECKPOINT C1:Vulkan0 回归试跑 | **PASS**（`run_baseline.bat baseline_regression\baseline\moe_129_8192_vk`，约 11 分钟），分项见 §7.2 |
| 19:09–19:10 | 对照基线核实 + `main` 状态 | 本文档 §1；用户确认 `main` 保留、不管、对照 upstream、CI 待定 |
| 19:10–19:11 | nvcc host 编译器验证 | VS2019 cl + nvcc 编 `test_sm60.cu` 并在 P100 上跑通，配方见 `NVCC_RECIPE.md` |
| 20:21 | `build.bat llamalibs StreamMoE_cuda`（Phase 2 双后端） | exit 0；CMake 找到 CUDAToolkit 12.6.85，host 编译器 VS2019 cl（19.29.30159.0）；日志 `temp\build-cuda.log` |
| 20:52 | CUDA bin 进系统 PATH | `cublas64_12.dll` 在 CUDA bin 里但 Machine PATH 原先无 CUDA 项，用户选“一劳永逸”，已加（`temp\add-cuda-path.log` ADDED_OK）；长驻 shell 仍需每次 prepend |
| ~21:02 | Phase 2 单元测试 | **main 7/7、StreamMoE 7/7、StreamMoE_cuda 7/7**（`test_device_cuda` 新增；日志 `temp\p2wrap-test-*.log`） |
| 21:38 | `build.bat llamalibs StreamMoE_cuda` 重编 | 曾卡在 [623/637] 链接约 28 分钟（ninja 空闲无子进程），kill 后重跑成功，二进制 21:38:30；DeepSeek 验证用此二进制 |
| 21:55–22:00 | DeepSeek CUDA0 验证 | 2048 MB 版 OOM（exit -1073740791，并发科研任务占 13,530 MiB）；1024 MB 版 **exit 0**，出 token（0.3 t/s，150W + 并发负载下测得）；日志 `temp\p2wrap-ds-cuda3.log` |
| 22:07 | `build.bat llamalibs dual`（Phase 2 代码） | exit 0（33s）；StreamMoE 22:07:00、StreamMoE_dump 22:07:12；upstream_dump 无事可做（仍 10/26），后单独全量重编 exit 0（6:40，日志 `temp\p2wrap-upstream.log`），freshness 双侧 OK |
| 22:16–22:39 | Gemma C1:Vulkan0 回归（Phase 2 代码） | **FAIL**（`RESULT: FAIL - see above`）：[3/8]–[6/8] 与 19:22 PASS 完全相同；[7b/8] 4 引擎 FAIL（18 行）；[8/8] smoke FAIL（"2020"×16）。Vulkan0 退化生成，Phase 2 **未提交、未 push**；分项与二分见 §7.3 |
| 23:17 起（8m32s） | Gemma C1:Vulkan0 回归重跑（fix 后） | **PASS**（`RESULT: PASS - all baseline and place regression checks passed`，日志 `temp\run-baseline-vk-fix.log` 第 1268 行，GEMMA_EXIT=0）：[3/8]–[6/8] 与 19:22 相同；[7b/8] PASS（4 warnings）；[8/8] smoke PASS（3.86s）。Phase 2 已提交并 push；分项见 §7.3 |
| fix 后重跑 | DeepSeek CUDA0 重跑 | **exit 0**，2m20s，CUDA0 池 1020/1024 MB，出 token（0.4 t/s，150W + 并发负载下测得，只认正确性）；日志 `temp\p2wrap-ds-cuda4.log` |

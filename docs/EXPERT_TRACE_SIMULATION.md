# 任务一：专家访问历史采集 + 策略模拟器 (EXPERT_TRACE_SIMULATION.md)

> 目的：在特殊编译版本里记录一次完整 prompt→generation 的**实际专家访问路径**（token、layer、expert ID），存成独立 trace 文件；用独立策略模拟器读该历史，**不重跑模型**即可测试不同池大小 / LRU/其他淘汰策略 / 预取策略 / 专家分布下的预期命中率，得到 cache size ↔ hit rate 关系曲线。

## 采集设计

- **采集点**：`llama_context::decode` 的 ubatch 循环内（`extract_layer_inputs` 之后，单 decode 线程）。该时刻该 ubatch 的 graph 已构建并同步执行完，遍历 `res->get_gf()`（主图 `ggml_cgraph`）的所有 `GGML_OP_MUL_MAT_ID` 节点，从 `src[2]`（路由 ids）读出每个被路由的专家号。
- **记录**：`(layer, global_token, expert)` 三元组，`layer` 从权重名 `blk.N.*` 解析；`global_token = n_tokens_prev + t`（ubatch 内 t 偏移累加）。存进三个动态增长的 `std::vector<uint32_t>`（`export_expert_layer/token/id`）。
- **写入**：析构（`~llama_context`）时一次性写 `LLM_EXPORT_DIR/expert_history.bin`（magic `EXPHIST1` + `u32 n_recs` + `n_recs × (u32 layer, u32 token, u32 expert)`）。
- **多线程乱序**：采集发生在 decode 主线程（ubatch 处理完、graph 同步之后），遍历图读 ids 是**串行顺序**的，不会乱序；每个 MUL_MAT_ID 内部按 `(t,k)` 顺序读取 ids（用 `ids->nb[1]` 真实行步长，兼容 argsort 层的稀疏布局）。跨 ubatch / 跨 layer 顺序与推理一致。
- 与任务二导出共用 `LLM_EXPORT_DIR` 与析构钩子，patch 合并为同一个 `patches/prefill-export-llama.patch`。

## 用法

```bat
rem 应用 patch 并重建后（见 patches/README.md 或 PREFILL_CROSS_VALIDATION.md）
set "LLM_EXPORT_DIR=temp\export_std"
build\main\bin\stream_moe.exe -m <model> --moe-ram-pool 71680 --expert-backend --temp 0 -p "<长 prompt>" -n <N>
rem 产出 temp\export_std\expert_history.bin
node tools\simulate_cache.js temp\export_std\expert_history.bin [maxSlots]
```

## 策略模拟器 `tools/simulate_cache.js`

- 读 `expert_history.bin`，按记录顺序重放访问序列（cache key = `layer*256 + expert`，层间专家独立）。
- 策略：
  - **LRU**：`Map` 插入序即最近使用序，满时淘汰最旧。
  - **LFU**：计数最小者淘汰。
  - **EST1（简化）**：recency-weighted decaying counter（分数 = 计数 / 2^((now-last)/64)），满时淘汰分数最小。
- 扫描池大小（slots）输出命中率曲线：`node tools/simulate_cache.js ... 4096` 输出 32→4096 各档 LRU/LFU/EST1 命中率。

## 实测（2026-08-26，prompt "Say hi." -n 2，--expert-backend）

- `expert_history.bin`：4644 条记录，覆盖 43 层，1148 个唯一 (layer,expert)。
- 短对话（7 token）专家复用高：池 32 slots 命中率即 ~73.6%，512 slots 后 ~75% 趋于饱和——曲线形态符合预期；**长上下文/多轮对话才体现池容量 vs 命中率的实际关系**，后续用 `benchmark/long_horizon_prompts.jsonl` 采集长历史。

## 注意事项

- 采集仅在该特殊编译版本（patch 应用后）生效，由 `LLM_EXPORT_DIR` 触发；无该环境变量则零开销。
- 还原：`git -C third_party/llama.cpp apply -R patches\prefill-export-llama.patch && git apply -R patches\prefill-export-streammoe.patch`。

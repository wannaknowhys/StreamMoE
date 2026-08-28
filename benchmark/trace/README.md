# benchmark/trace/ - 追踪统计数据（策略/容量测试数据源）

> 非临时目录。存放 prefill 导出的专家访问历史 + prompt 快照，供
> `tools/simulate_cache.js`（LRU/LFU/EST1/OPT 命中率曲线）与容量测试使用。

## 产物

| 文件 | 来源 | 用途 |
|---|---|---|
| `expert_history_main.bin` / `expert_history_draft.bin` | llama-server/cli 运行后析构导出（需 `LLM_EXPORT_DIR=benchmark/trace`） | `node tools/simulate_cache.js <file>` 命中率曲线 |
| `prefill_export_main.bin` / `prefill_export_draft.bin` | 同上 | prefill 交叉验证（verify_prefill.js） |
| `prompt_<n>tok_<ts>.json` | `tools/prefill_from_trace.js` 生成 | 记录每次 prefill 实际喂的 prompt（可复现） |

## 使用

```bat
rem 1) 起 server（必须设 LLM_EXPORT_DIR 指向本目录，ctx 开满）：
set "LLM_EXPORT_DIR=F:\Dev\StreamMoE\benchmark\trace"
build\main\llama-build\bin\llama-server.exe -m <deepseek> --expert-backend --moe-ram-pool 71680 ^
    --cache-type-k q8_0 --cache-type-v q8_0 --fit off --no-warmup -c 1048576 -t 16 ^
    --host 127.0.0.1 --port 8993 --no-webui

rem 2) 喂 ~10000 token 真实对话 prefill：
node tools\prefill_from_trace.js --tokens 10000 --out benchmark\trace

rem 3) 退出 server（Ctrl+C 触发析构导出），然后跑策略/容量测试：
node tools\simulate_cache.js benchmark\trace\expert_history_main.bin
```

## 注意

- 本目录数据文件（*.bin / prompt_*.json）由 `.gitignore` 排除，不入仓库；`.gitignore` 本身保留。

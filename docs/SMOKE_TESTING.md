# 冒烟测试约定 (SMOKE_TESTING.md)

> **规则**：在**专家池子系统（route B 注入原版 server，M3）完成之前**，所有快速冒烟测试用下面的小模型。**不要**直接拿 DeepSeek-V4-Flash（`N:\AI_LLM\DeepSeek-V4-Flash-0731\...UD-Q8_K_XL`，162GB 在慢 USB-NVMe N: 盘）冒烟——没有专家池子系统时冷盘加载太慢（几分钟 ~ 卡死）。

## 冒烟模型

| 用途 | 模型 | 说明 |
|---|---|---|
| **MoE** | `N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf` | Gemma 4 26B **A4B**（每 token 激活 4 个专家），验证 MoE 前向路径/路由 |
| **Dense** | `F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | 2B dense，快速验证基本链路（文本）；模型在快盘 |

## 验证命令（原版 llama-server，M1 起可用）

```bat
rem MoE（gemma-4-26B-A4B）；--no-warmup 跳过空跑预热（MoE 下避免启动时冷盘专家装载）
build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --fit off --no-warmup --no-webui

rem Dense（Qwen3-VL-2B）
build\main\llama-build\bin\llama-server.exe -m F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --no-warmup --no-webui
```

然后 `/health`（200）就绪后 POST `/v1/chat/completions`（`{"messages":[{"role":"user","content":"Say hi in one short sentence."}],"stream":false}`）。

## 验证记录（2026-08-27，M1 后）

- Llama-3.3-8B-Instruct-128K Q4_K_M：✅ 3s 加载，chat 正常返回。
- Qwen3-VL-2B-Instruct Q4_K_M（dense）：✅ 6s 加载，chat 返回 "Hi!"。
- gemma-4-26B-A4B-it-UD-Q4_K_M（MoE）：✅ **已重新下载**（15.78 GB，GGUF magic 有效，之前是 0 字节占位）。用于 M3 之后 route B 专家池验证（`--expert-backend`）。

## 何时切回 DeepSeek-V4-Flash

**专家池子系统（M3 route B 注入）跑通之后**，才用 `N:\AI_LLM\DeepSeek-V4-Flash-0731\...UD-Q8_K_XL` 做短程测试（届时专家池接管 N: 冷盘专家装载，冒烟不至于几分钟）。

## 大 MoE 测试约定（2026-08-27 起）

- **DeepSeek-V4-Flash 等大 MoE 用 `--moe-ram-pool 71680`（70GB 池）**，不用小池。
  - 教训：8GB 小池 + 162GB 模型触发 llama.cpp `fit_params` 失败并卡死/死机（`--fit off` 无法完全规避，模型 mmap 虽不驻留但 fit 计算模型字节超系统内存）。大池 + `--fit off` 才稳。
  - deepseek 启动参数：`--expert-backend --moe-ram-pool 71680 --fit off`。
- 池预算乘法已修（double 计算，避免 uint64 溢出）。

## Context 与 KV 观察约定（2026-08-27 起）

- **短程测试一律 `-c 8192`**（gemma / Qwen 等小模型）。
- **长程 jsonl 用更大的 `-c`**。
- **DeepSeek KV cache 实测流程**（q8_0 + windowed SWA 下，最大应约 3GB）：
  1. `-c 8192` 启动，看 "KV Cache Memory (llama.cpp actual)" 行（server-context 注入打印，基于 `llama_get_memory_breakdown`）。
  2. `-c 100000` 观察。
  3. `-c 1048576` 开到最大观察。
  - 实测值已追加在下方记录表。
- KV 实际尺寸显示代码：`llama_get_memory_breakdown(ctx).context` 求和（自研 `llama_engine::kv_memory_bytes` 同款，llama_engine.cpp:220；原版 server 由 server-context.cpp 注入打印）。

### DeepSeek-V4-Flash KV 实测记录（q8_0 + windowed，--expert-backend --moe-ram-pool 71680）

| ctx | KV 内存 |
| :--- | :--- |
| 8192 | （待测）|
| 100000 | （待测）|
| 1048576 | （待测；预期 ~3GB）|

## Qwen3-VL 视觉测试（mmproj + 图片）

仅 Qwen3-VL（本机已备 mmproj），作为 dense 模型的完整链路测试：

- 模型：`F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf`
- mmproj：`F:\Dev\computer-use\mmproj-F16.gguf`
- 图片：`F:\Dev\computer-use\node\_game_full.png`
- Prompt（英文）：问图片里都有啥（"What is in this image?" 之类），作为验证的一部分。

```bat
build\main\llama-build\bin\llama-server.exe -m F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf ^
    --mmproj F:\Dev\computer-use\mmproj-F16.gguf --host 127.0.0.1 --port 8997 -c 4096 -t 16 --no-warmup --no-webui
rem chat body: content = [{type:text,text:"What is in this image?"},{type:image_url,image_url:{url:"data:image/png;base64,<b64>"}}]
```

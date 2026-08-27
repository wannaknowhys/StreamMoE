# 冒烟测试约定 (SMOKE_TESTING.md)

> **规则**：在**专家池子系统（route B 注入原版 server，M3）完成之前**，所有快速冒烟测试用下面的小模型。**不要**直接拿 DeepSeek-V4-Flash（`N:\AI_LLM\DeepSeek-V4-Flash-0731\...UD-Q8_K_XL`，162GB 在慢 USB-NVMe N: 盘）冒烟——没有专家池子系统时冷盘加载太慢（几分钟 ~ 卡死）。

## 冒烟模型

| 用途 | 模型 | 说明 |
|---|---|---|
| **MoE** | `N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | Gemma 4 26B **A4B**（每 token 激活 4 个专家），验证 MoE 前向路径/路由 |
| **Dense** | `F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | 2B dense，快速验证基本链路（文本）；模型在快盘 |

## 验证命令（原版 llama-server，M1 起可用）

```bat
rem MoE（gemma-4-26B-A4B）
build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 2048 -t 16 --no-webui

rem Dense（Qwen3-VL-2B）
build\main\llama-build\bin\llama-server.exe -m F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 2048 -t 16 --no-webui
```

然后 `/health`（200）就绪后 POST `/v1/chat/completions`（`{"messages":[{"role":"user","content":"Say hi in one short sentence."}],"stream":false}`）。

## 验证记录（2026-08-27，M1 后）

- Llama-3.3-8B-Instruct-128K Q4_K_M：✅ 3s 加载，chat 正常返回。
- Qwen3-VL-2B-Instruct Q4_K_M（dense）：✅ 6s 加载，chat 返回 "Hi!"。
- gemma-4-26B-A4B-it-UD-Q4_K_M（MoE）：❌ **文件 0 字节**（`N:\AI_LLM\` 下为空占位，`gguf_init` 读 magic 失败）——需重新下载后才能冒烟。

## 何时切回 DeepSeek-V4-Flash

**专家池子系统（M3 route B 注入）跑通之后**，才用 `N:\AI_LLM\DeepSeek-V4-Flash-0731\...UD-Q8_K_XL` 做短程测试（届时专家池接管 N: 冷盘专家装载，冒烟不至于几分钟）。

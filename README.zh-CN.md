[English](README.md) | [简体中文](README.zh-CN.md)

# StreamMoE

**StreamMoE** 是一款物理内存有界的 MoE 推理引擎。针对超大 MoE 模型（DeepSeek-V4 / Qwen2-MoE / Mixtral），通过自定义 ggml backend 接管 `MUL_MAT_ID`，将**专家权重从 `mmap` 中移出、装入自研紧凑槽私有池**（Direct I/O 装载），dense 层保持 llama.cpp 默认行为，从而让物理内存受控于池预算。

---

## 核心技术特性

1. **真实推理引擎**：vendored libllama（deepseek4 架构，43 层 / 256 专家）驱动完整前向——MLA、DSA indexer、hyper-connections、Sinkhorn、hash + argsort 路由、shared expert。
2. **MoE 去 mmap（route B）**：专家权重在自管理槽池中，Direct I/O（DIO）按需装载；物理内存受池预算约束（`--moe-ram-pool`，如 128GB 主机用 70GB）。与 stock 图**数值逐位一致**（已验证）。
3. **快速启动**：权重 mmap 映射而非物理拷贝，专家按需 DIO 流式装载；首 token 时延由 N: 冷盘主导，而非模型加载（模型加载约 1.5s）。
4. **按需专家装载 + 自适应驱逐（EST1）**：命中专家立即计算，未命中专家异步预取；LRU/LFU/EST1 有界池驻留策略。
5. **OpenAI 兼容 API 服务端（`stream_moe_server`）**：轻量 C++ HTTP 服务端，`/v1/chat/completions`（SSE 流式）、`/v1/models`、`/health`、`/stats`。
6. **交互式多轮对话 CLI（`stream_moe`）**：流式打字机输出 + 精确 KV Cache 内存开销显示。
7. **投机解码（draft 模型）**：规划中（见 `docs/BUG_TRACKER.md` B11）；实现后使用草稿模型文件。

---

## 工具链与路线图

架构对标 `llama.cpp` 工具链：**[docs/LLAMA_EXE_ROADMAP.md](docs/LLAMA_EXE_ROADMAP.md)**。

项目布局、构建子路径与测试/结果归档约定：**[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**。

| 可执行程序 | 职责说明 | 状态 |
| :--- | :--- | :--- |
| **`build/main/bin/stream_moe.exe`** | 交互式多轮对话 REPL & 单次提示词运行器 | **已就绪** |
| **`build/main/bin/stream_moe_server.exe`** | OpenAI 兼容流式 HTTP/SSE API 服务端 | **已就绪** |
| **`build/main/bin/stream_moe_bench.exe`** | MoE 专属多维基准评测工具 | 规划中 |
| **`build/main/bin/stream_moe_convert.exe`** | 4KB 扇区对齐零拷贝 GGUF 转换优化器 | 规划中 |

---

## 编译与测试指引

### 环境要求
- 支持 C++17 和 OpenMP 的 Clang / LLVM、MSVC 或 GCC
- Windows (PowerShell / `cmd`) 或 Linux / POSIX

### Windows 编译
```powershell
# 编译所有程序 (stream_moe.exe, stream_moe_server.exe) 到 build\main\bin\
.\build.bat build

# 执行单元测试套件
.\build.bat test

# 清理编译产物（全部 tag）
.\build.bat clean
```

### Linux 编译
```bash
make test
```

---

## 快速使用

### 1. 交互式多轮对话 CLI 模式
```powershell
# 启动交互式 REPL，使用 70GB 专家池与 16 物理核心
build\main\bin\stream_moe.exe `
    -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --moe-ram-pool 71680 `
    -c 4096 `
    -t 16 `
    -i
```

### 2. 启动 OpenAI 兼容 API 服务端
```powershell
# 在 8080 端口启动 API 服务
build\main\bin\stream_moe_server.exe `
    -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --host 127.0.0.1 `
    --port 8080 `
    --moe-ram-pool 71680 `
    -t 16
```

#### API 端点
- `POST /v1/chat/completions`（支持 `"stream": true` 流式 SSE）
- `GET /v1/models`
- `GET /health`
- `GET /stats`（实时池占用、命中计数）

---

## 许可证
MIT License.

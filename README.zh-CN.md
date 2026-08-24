[English](README.md) | [简体中文](README.zh-CN.md)

# StreamMoE (OffloadMoE) 极致内存优化 MoE 推理引擎

**StreamMoE** 是一款专为超大 MoE 架构（如 100GB+ DeepSeek-V4、Qwen2-MoE、Mixtral）设计的高性能、内存极致优化的推理引擎。突破物理显存/内存容量限制，在 32GB RAM / 8GB VRAM 等消费级设备上实现低延迟、高吞吐的流式混合推理。

---

## 核心技术特性

1. **极致内存卸载 (Extreme Memory Offload)**：通过扇区对齐异步 Direct I/O (DIO) 与非包容非排他 (NINEC) 多级缓存池，允许在 4GB-32GB RAM 预算下稳定运行 150GB+ MoE 模型。
2. **秒级即时启动 (< 0.15s)**：支持全流式零预载模式 (`--moe-preload none`)，告别传统百 GB 模型数分钟的加载等待。
3. **双线程重叠流水线 (Dual-Thread Overlapping)**：推理计算线程与 IO 调度线程完全解耦，选通 Hit 专家立即并行计算 GEMM，Miss 专家后台异步直刷，隐藏 IO 延迟。
4. **自适应频次收敛驱逐策略 (`EST1`)**：基于 LRU 与指数衰减滑动窗口，自动从全局长期频次向会话近期活跃路由热点收敛，自动持久化。
5. **零开销静态子图指针重绑定 (Pointer Rebind)**：预分配静态并发子图，运行时直接就地重绑定 `tensor->data` 指针，零动态建图开销。
6. **5 维自适应资源状态机**：自适应监控 CPU/GPU/PCIe/Disk/投机推理收益，动态调优并内置换页抖动紧急复位机制 (Thrashing Emergency Reset)。
7. **OpenAI 兼容流式 API 服务端 (`stream_moe_server`)**：原生高性能 C++ HTTP 服务端，支持 `/v1/chat/completions` (SSE 流式输出)、`/v1/models`、`/health` 与 `/stats`。
8. **交互式多轮对话 CLI (`stream_moe`)**：支持打字机流式输出、上下文长度 $N_{\text{ctx}}$ 与精确 KV Cache 内存开销动态计算。

---

## 工具链矩阵与路线图

详见 [`LLAMA_EXE_ROADMAP.md`](LLAMA_EXE_ROADMAP.md)，对标 `llama.cpp` 工具链体系（`llama-cli`, `llama-server`, `llama-bench`, `llama-quantize`）。

| 可执行程序 | 职责说明 | 状态 |
| :--- | :--- | :--- |
| **`bin/stream_moe.exe`** | 交互式多轮对话 REPL & 单次提示词运行器 | **已就绪** |
| **`bin/stream_moe_server.exe`** | OpenAI 兼容流式 HTTP/SSE API 服务端 | **已就绪** |
| **`bin/stream_moe_bench.exe`** | MoE 专属多维基准评测工具 | 规划中 |
| **`bin/stream_moe_convert.exe`** | 4KB 扇区对齐零拷贝 GGUF 转换优化器 | 规划中 |

---

## 编译与测试指引

### 环境要求
- 支持 C++17 和 OpenMP 的 Clang / LLVM、MSVC 或 GCC
- Windows (PowerShell / `cmd`) 或 Linux / POSIX

### Windows 编译
```powershell
# 编译所有程序 (stream_moe.exe, stream_moe_server.exe)
.\build.bat build

# 执行 Phase 1 ~ 5 全部单元测试套件
.\build.bat test

# 清理编译缓存
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
# 启动交互式 REPL，自动分配 75% 可用内存并使用 16 物理核心
bin\stream_moe.exe `
    -m "path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --draft-model "path/to/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" `
    --moe-ram-pool auto `
    -c 4096 `
    -t 16 `
    -i
```

### 2. 启动 OpenAI 兼容 API 服务端
```powershell
# 在 8080 端口启动 API 服务
bin\stream_moe_server.exe `
    -m "path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --host 127.0.0.1 `
    --port 8080 `
    --moe-ram-pool auto `
    -t 16
```

#### API 端点
- `POST /v1/chat/completions` (支持 `"stream": true` 流式 SSE)
- `GET /v1/models`
- `GET /health`
- `GET /stats` / `GET /metrics` (查看实时命中率、槽位分配与状态机模式)

---

## 许可证
MIT License.
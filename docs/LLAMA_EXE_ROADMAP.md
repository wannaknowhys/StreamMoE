# StreamMoE 可执行程序生态全景规划与路线图 (LLAMA_EXE_ROADMAP.md)

> **目标**：全面对标并超越 `llama.cpp` 的可执行程序矩阵，打造基于 StreamMoE 极致内存卸载内核的端到端工具链生态。

---

## 1. 对标 llama.cpp 工具链矩阵

| 目标可执行程序 | 对标 llama.cpp 工具 | 核心职责与 StreamMoE 增强特性 | 状态 / 路线 |
| :--- | :--- | :--- | :--- |
| **`stream_moe`** | `llama-cli` / `llama-run` | **交互式多轮对话 REPL & 单次推理**：<br>• 支持流式 Token 实时打字机输出<br>• 支持多轮对话历史上下文保持与 Prompt Cache 复用<br>• 有界专家池（`--moe-ram-pool`）与多线程调度<br>• 显式输出 Context 长度与精确 KV Cache 显存/内存开销 | **✅ 已迁移**（build\main\llama-build\bin\llama-cli.exe，原自研 stream_moe 已删，M4）|
| **`stream_moe_server`** | `llama-server` | **OpenAI 兼容高并发流式 API 服务端**：<br>• `/v1/chat/completions` (支持 SSE 流式传输)<br>• `/v1/models`、`/health` 与 `/stats` 实时暴露专家命中、池占用<br>• 官方 server 全模块（OpenAI API、多槽、context-shift、tools） | **✅ 已迁移**（build\main\llama-build\bin\llama-server.exe，原自研 stream_moe_server 已删，M4）|
| **`stream_moe_bench`** | `llama-bench` | **MoE 专属多维基准评测工具**：<br>• 自动化扫描缓存池容量下的 Cache Hit Rate 曲线<br>• 测量 Prefill 与 Decode 的 TPS、DIO 延迟分布<br>• 评估投机推理 (Speculative Decoding) 在不同 K 步长下的实际加速比 | **[ ] Phase 8** |
| **`stream_moe_convert`** | `llama-quantize` / `convert_hf_to_gguf` | **4KB 扇区对齐零拷贝 GGUF 优化工具**：<br>• 重排 GGUF 文件：将 Header、Dense 权重与各专家数组强制 4KB 扇区对齐<br>• **革命性收益**：彻底消除 Staging 临时缓冲区与 `memcpy` 开销，实现 Direct I/O 直接直刷进 Pinned Slot 物理内存 | **[ ] Phase 9** |
| **`stream_moe_perplexity`**| `llama-perplexity` | **困惑度与量化精度验证工具**：<br>• 在 Wikitext-2 等标准数据集上评估流式 Offload 下的 PPL 准确性<br>• 验证动态专家调度与指针重绑定的数值一致性 | **[ ] Phase 10** |

---

## 2. 核心模块详细设计与路线规划

### 2.1 交互式流式 CLI (`stream_moe`)
* **流式回调管道 (Streaming Token Callback)**：
  * 推理内核每解码出一个 Token，立即通过标准输出实时刷新（打字机流式体验）。
* **多轮对话与 Prompt Cache (KV Cache Management)**：
  * 支持 `/clear`（重置会话）、`/reset`（清空热度统计）、`/stats`（查看当前 Cache 命中率与状态机模式）。
  * 显示详细的内存画像：
    ```
    [Model Metadata]
      Architecture:    deepseek4 (43 layers, 256 experts/layer)
      Context Window:  4096 tokens (Max: 1048576)
      KV Cache Type:   FP16 (2 bytes/elem)
      KV Cache Size:   1.34 GB (Layers: 43, KV-Heads: 8, Head-Dim: 128)
      RAM Pool Size:   70.00 GB (71680 MB, bounded expert pool)
      Compute Threads: 16 Physical Cores (OpenMP OMP_PROC_BIND=spread)
    ```

### 2.2 OpenAI 兼容微型 API 服务端 (`stream_moe_server`)
* **架构设计**：
  * 基于 Win32 IOCP / POSIX epoll 原生 Socket 实现极轻量 HTTP 1.1 / SSE 解析器，严禁引入臃肿第三方库。
  * 请求路由：
    - `POST /v1/chat/completions`: 解析 messages 数组，转为模型 Prompt，支持 `"stream": true` (text/event-stream)。
    - `POST /v1/completions`: 标准续写补全接口（**未实现**，仅愿景）。
    - `GET /v1/models`: 返回已加载的 MoE 模型元数据。
    - `GET /stats`: 返回 JSON 格式的运行时指标（Hit Rate, TPS, Pool Usage, EST1 Hot Experts）。
* **并发会话与专家池共享**：
  * 多个请求并发复用全局 Pinned RAM / VRAM 专家池，共享热门专家局部性，显著提升整体吞吐。

### 2.3 4KB 扇区对齐转换器 (`stream_moe_convert`)
* **原理与创新**：
  * 现存标准 GGUF 的 Tensor 仅做 32 字节对齐，导致 Direct I/O 读取时头部和尾部必须多读扇区并经过一次 Staging 复制。
  * `stream_moe_convert` 读取任意 GGUF，将每个专家子 Tensor 紧凑重排并填充 Padding 使其物理文件偏移和长度均为 4096 整数倍。
  * **效果**：IO 引擎可直接将 `VirtualAlloc` 锁页 Slot 指针作为 IOCP 读取的目标地址，**零内存拷贝 (Zero-Copy DMA/DIO)**！

---

## 3. 分步实施排期 (Execution Roadmap)

- **Step 1 (已启动)**：升级 `stream_moe` 主程序：
  - 自动检测并默认分配 75% 可用物理内存作为 Pinned Pool。
  - 默认绑定 16 物理核心。
  - 动态计算并打印 Context 尺寸与 KV Cache 内存开销。
  - 实现交互式流式打字机 REPL 模式 (`-i` / `--interactive`)。
- **Step 2**：研发 `stream_moe_server` 服务端模块（OpenAI 兼容 + SSE 流式输出 + `/stats` 监控端点）。
- **Step 3**：研发 `stream_moe_convert` 4KB 扇区对齐工具，达成全链路 Zero-Copy 直读。
- **Step 4**：研发 `stream_moe_bench` 与 `stream_moe_perplexity` 评估套件。
[English](README.md) | [简体中文](README.zh-CN.md)

# StreamMoE (OffloadMoE) 极致内存优化 MoE 推理引擎

**StreamMoE** 是一款专为超大 MoE 架构（如 100GB+ DeepSeek-V4、Qwen2-MoE、Mixtral）设计的高性能、内存极致优化的推理引擎。突破物理显存/内存容量限制，在 32GB RAM / 8GB VRAM 等消费级设备上实现低延迟、高吞吐的流式混合推理。

---

## 核心技术特性

1. **极致内存卸载 (Extreme Memory Offload)**：通过扇区对齐异步 Direct I/O (DIO) 与非包容非排他 (NINEC) 多级缓存池，允许在 4GB-32GB RAM 预算下稳定运行 150GB+ MoE 模型。
2. **秒级即时启动 (< 0.2s)**：支持全流式零预载模式 (`--moe-preload none`)，告别传统百 GB 模型数分钟的加载等待。
3. **双线程重叠流水线 (Dual-Thread Overlapping)**：推理计算线程与 IO 调度线程完全解耦，选通 Hit 专家立即并行计算 GEMM，Miss 专家后台异步直刷，隐藏 IO 延迟。
4. **自适应频次收敛驱逐策略 (`EST1`)**：基于 LRU 与指数衰减滑动窗口，自动从全局长期频次向会话近期活跃路由热点收敛，自动持久化。
5. **零开销静态子图指针重绑定 (Pointer Rebind)**：预分配静态并发子图，运行时直接就地重绑定 `tensor->data` 指针，零动态建图开销。
6. **5 维自适应资源状态机**：自适应监控 CPU/GPU/PCIe/Disk/投机推理收益，动态调优并内置换页抖动紧急复位机制 (Thrashing Emergency Reset)。

---

## 编译与测试指引

### 环境要求
- 支持 C++17 和 OpenMP 的 Clang / LLVM、MSVC 或 GCC
- Windows (PowerShell / `cmd`) 或 Linux / POSIX

### Windows 编译
```powershell
# 编译主程序 (bin\stream_moe.exe)
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

```bash
# 在 4GB Pinned RAM 预算下驱动 150GB DeepSeek-V4 MoE 5 分片模型进行投机推理
bin\stream_moe.exe ^
    -m "path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" ^
    --draft-model "path/to/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" ^
    --moe-ram-pool 4096 ^
    --moe-preload none ^
    -p "请详细阐述 MoE 稀疏门控路由的核心物理机制。" ^
    -n 64
```

### 命令行参数说明
| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `-m, --model <path>` | GGUF 模型主路径（支持多分片 `-00001-of-00005.gguf` 自动发现） | *必填* |
| `--draft-model <path>` | Dense 草稿模型路径（启用投机推理） | 无 |
| `-ngl, --gpu-layers <N>` | 卸载到 GPU 的 Dense 骨干层数 | `0` |
| `--moe-vram-pool <MB>` | Pinned VRAM 显存专家缓存池容量 (MB) | `4096` |
| `--moe-ram-pool <MB>` | Pinned Host RAM 内存专家缓存池容量 (MB) | `8192` |
| `--moe-preload <policy>` | 预加载策略 (`none`, `ram`, `vram`, `all`) | `none` |
| `--stats-file <path>` | 专家热度统计文件路径 (`EST1` 格式) | `<model_name>.bin` |
| `-n, --n-predict <N>` | 生成 Token 最大数量 | `32` |
| `-t, --threads <N>` | CPU GEMM 运算线程数 | `16` |

---

## 系统架构总览

```
[ GGUF 多分片存储介质 ]
        |
        v (Direct I/O 4KB 扇区对齐直读)
[ 异步 DIO IOCP / io_uring 引擎 ]
        |
        v (Staging 缓冲区切片 memcpy)
[ Pinned RAM 锁页内存池 (VirtualAlloc/VirtualLock) ] <---> [ Pinned VRAM 显存池 (Vulkan/CUDA) ]
        |                                                                 |
        +-------------------------------+---------------------------------+
                                        |
                         [ 门控路由 Hit / Miss 自动分离 ]
                         /                               \
         [ Hit 专家：立即并行 GEMM ]            [ Miss 专家：后台异步调度搬运 ]
                         \                               /
                          +---------> [ 各层激活累加归约输出 ]
```

---

## 许可证
MIT License.
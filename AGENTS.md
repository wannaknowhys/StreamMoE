# StreamMoE AGENTS.md - 用户思维与协作偏好（先读，做选择时参考）

> 本文是用户与 AI 协作的**偏好指南**。开始任何工作前先读本文件；做技术选型、方案取舍、
> 或不确定怎么做时，参考这里的偏好；**拿不定主意就问用户**（用 question 工具）。

## 用户是谁

用户是这套 StreamMoE 引擎的架构师与工程主导者：用缓存理论想问题、用手搓代码落地、用粗判据把关、用完整记录兜底。下列偏好是从长期协作中提炼的，逐条对应实际例子。

## 一、思维方式

1. **抽象与务实并存，切换极快**
   - 高层用抽象心智（如"专家池 = 只读多级缓存 NINEC"）统摄 CPU/GPU/磁盘。
   - 落地用粗判据快速把关（如"cos 只要一直 0.9 就没大错"、"真 short 不会只少 1M"）。
   - 做判断时两者都要给：既有结构视角，也有数量级/阈值视角。

2. **追问本质，但接受多因并存**
   - 会追问到根因（如"是 ulp 差异还是少一个 repack"），但接受"直接原因 + 放大机制"的复合解释，不强迫单一归因。

3. **偏好"改函数/改代码路径"，而非引入抽象机制**
   - 倾向直接改 llama.cpp 的函数（build_moe_ffn、路由、内核），而不是伪造数据结构 / buffer 机制（extra_buffer_type 等）。
   - 报告方案时优先给"直接改代码路径"的走法；只有确实需要才提机制层方案。

4. **通用性 > 单例修复**
   - "我不是单独给 gemma 的补丁，我是为了所有 MoE 通用"——修复/功能要落到模型无关的通用机制，不迁就眼前模型。

## 二、工程偏好

5. **先聊透设计，再动手**
   - 重大改动前：把方案、挂点、数据结构、取舍先摆出来讨论，确认后才实现。用户会说"你先给我讲讲""咱们先商量一下""聊完设计你就改"。

6. **用户自己掌控执行**
   - 给清晰的命令行（bat/node），用户自己跑、自己看输出。不要替用户跑长任务（除非明确要求）。

7. **一切可回放、可追溯**
   - 数据、模型输出、prompt、命令历史都要留档（bin / chat.json / prompt 快照）。模型吐出的**全部文本**（thinking + content，含截断/ctx 耗尽）都要记录。
   - 排查问题时追数据来源、追历史命令。

8. **性能底线不可妥协（异步/零拷贝）**
   - 异步加载（IOCP / io_uring / io_submit）、并发 in-flight、零拷贝直读是前提不是可选项。
   - 报性能方案时默认按"异步 + 并发 + 免 staging"设计。

9. **规范与防污染**
   - vendored `third_party/llama.cpp` **永不 commit**，改动只留工作区、靠 `patches/*.patch` 记录；主仓库文件正常 git add/commit。
   - 多 commit、每步成功就 commit；临时调试代码用完清理；临时文件进 temp/（gitignored）。
   - **`muliti_test.bat`（跑 deepseek 导出测试的入口）随 commit 一并提交**——它是测试入口，保持跟踪，不要忽略/删除。

10. **文档双语文档 + 落地**
    - 设计文档写英文 + 简体中文两版（`[English](x.md) | [简体中文](x.zh-CN.md)`）；讨论定的方案要落地成 md 并提交。

## 三、协作规则

- 开始任务前：先读本文件 + `docs/CHECKPOINT.md`（当前状态）+ `docs/PROJECT_STRUCTURE.md`（结构）。
- 做技术选型/方案取舍时：按上述偏好给推荐（改函数优先、通用性优先、性能底线、可回放），并说明取舍。
- **拿不定主意 / 有重大分叉（改动面大、影响架构、删除保留等）→ 用 question 工具问用户**，不要自作主张。
- 用户会纠正——纠正意见记进本文件（追加"修订"段，保持本文件与用户当前想法同步）。

## 四、docs 文档地图（按需读取）

### 重要必读（会话开始 / 大改动前）
| 文档 | 内容 |
| :--- | :--- |
| `docs/CHECKPOINT.md` | 当前状态、下一步、验证命令（会话恢复先读）|
| `docs/PROJECT_STRUCTURE.md` | 目录/产物/规范、vendored patch 纪律 |
| `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` | route B 核心设计（第三路径：官方内核 + 均匀 stride 槽池）|
| `docs/Backend.md` | 自定义 backend / expert pool 调度设计 |
| `docs/VENDORED_MODIFICATIONS.md` | vendored 改动汇总 + patch 记录 |

### 按场景读取
| 场景 | 文档 |
| :--- | :--- |
| 调度/池（dir 二维、异步装载、全局线程、驱逐打分）| `docs/EXPERT_SCHEDULER_DESIGN.md` |
| GPU/多设备（vulkan、HOST_VISIBLE、EMA 放置）| `docs/ROUTE_B_GPU_PHASE.md` |
| 多模型池 / 异构子池 | `docs/MULTI_MODEL_POOL.md`、`docs/MULTI_SUBPOOL.md` |
| GGUF 格式 v1/v2 / RAID0 分片 | `docs/STREAMMOE_GGUF_FORMAT.md` |
| prefill 交叉验证 / 专家历史模拟 / repack 排查 | `docs/PREFILL_CROSS_VALIDATION.md`、`docs/EXPERT_TRACE_SIMULATION.md`、`docs/REPACK_DIVERGENCE_DEBUG.md` |
| delegate 排查方法论 / bug 清单 | `docs/DEBUG_DELEGATION.md`、`docs/BUG_TRACKER.md` |
| patch 拆分/更新踩坑 | `docs/PATCH_SPLITTING_PITFALLS.md` |
| 迁移上游工具 / 可执行程序路线 | `docs/UPSTREAM_TOOLS_MIGRATION.md`、`docs/LLAMA_EXE_ROADMAP.md` |
| 冒烟/测试/采样 | `docs/SMOKE_TESTING.md`、`docs/TEST_FLOW.md`、`docs/SAMPLING.md` |
| ASan 构建 | `docs/ASAN_BUILD.md` |
| v2 架构修正 | `docs/V2_ARCHITECTURE_REVISION.md` |

### 可以不读（参考/历史）
| 文档 | 内容 |
| :--- | :--- |
| `docs/REVIEW_2026_08_28.md` | 早期审查对照（结论已并入代码）|
| `docs/LLAMA_MMAP_CALLS.md` | mmap 调用点调试地图（低优先级）|

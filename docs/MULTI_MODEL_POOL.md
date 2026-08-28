# 多模型专家池设计（主模型 + 草稿模型）(MULTI_MODEL_POOL.md)

> 状态：**设计定案（2026-08-27）**。目标：主模型（deepseek）与草稿模型（dspark/dflash）同时走 route B 专家池，
> 各自独立预算、独立池实例，draft 默认全常驻。
> 相关：`docs/MULTI_SUBPOOL.md`（异构分组）、`docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7（第三路径：官方 `ggml_mul_mat_id` + 均匀 stride 槽）。

---

## 1. 背景与问题

- 主模型（如 deepseek-4，43 层 / 256 专家）与 draft（dspark，**3 层 / 256 专家**，dflash 架构，全量 10.3GB，每专家槽 ~13.4MB Q8_0）同时驻留。
- 两者专家张量**同名**（`blk.N.ffn_{gate,up,down}_exps`）、**层号都从 0 开始**、拓扑/量化独立——不能共享同一个池控制面。
- 现状 `route_b_setup` 只对主模型生成 overrides；若 draft 加载复用了主模型 overrides，draft 专家会被挂进主池 → 槽布局错乱（同名 pattern 冲突）。
- draft 专家量小（10.3GB），适合**全常驻**；主模型专家 162GB 只能按热度驻留一部分。

## 2. 参数设计（与主池多设备参数对称）

| 参数 | 归属 | 语义 |
| :--- | :--- | :--- |
| `--moe-ram-pool <MB>` | 主模型专家池（RAM） | 不变；缺省 0 = 75% 空闲 RAM |
| `--moe-vram-pool <MB>` | 主模型专家池（VRAM） | 不变；Phase B 前占位 |
| `--moe-draft-ram-pool <MB>` | **草稿专家池（RAM）** | 新增；**缺省 = 草稿专家完整字节（全常驻，无驱逐）** |
| `--moe-draft-vram-pool <MB>` | **草稿专家池（VRAM）** | 新增；缺省 0 = 草稿不进 VRAM（Phase B 前仅 RAM 生效） |

- **位置** = 给了哪个设备参数就是哪个设备（与主池完全对称）。
- **草稿池预算独立**，不占 `--moe-ram-pool`。总物理占用 = 主池 + 草稿池。

## 3. 行为语义

1. **草稿池无大小参数 = 完整大小**：`route_b_setup(draft)` 解析拓扑得到专家总字节（dspark = 10.3GB），草稿池按全量分配，全部槽常驻、不驱逐。
2. **草稿池有大小参数 = 指定大小**：如 `--moe-draft-ram-pool 4096` → 草稿池 4GB，草稿池也走热度驱逐（EST1，同主池机制）。
3. **指定大小 > 专家全量**：clamp 到全量 + warning（无意义）。
4. **draft 无专家（dense draft）**：草稿池参数 no-op（拓扑 `n_expert == 0` 不建池）。
5. **锁定顺序 = 草稿优先驻留**：草稿池固定分配（全常驻）优先于主池；主池的驱逐/动态行为永不碰草稿池槽，草稿池（缺省）也永不驱逐。实现上草稿池实例先 commit 物理内存。

## 4. 架构：每模型一个池实例 + buft 身份

- **模型池注册表**：`route_b_setup` 从"全局单池"改为"每模型一个池实例"，按模型路径去重（幂等）。
  - 每个池实例 = 拓扑（`moe_loader`）+ 预算（RAM/VRAM）+ 独立 weight buft + 独立 overrides + 独立 scheduler（slot_meta/expert_directory/EST1/DIO 句柄）。
- **"哪个文件" = buft 身份**：主模型加载挂主模型 overrides（主模型 buft），draft 加载挂 draft overrides（draft buft）。
  同名张量靠 **per-model overrides 只在各自加载调用生效** 隔离，运行时零处理。
- **图独立**：主模型 `ctx_tgt` 与 draft `ctx_dft` 是两个独立 llama_context，各建各的 cgraph、各跑各的 sched。
  llama.cpp 图构建零改动；共享的只有 backend 注册 + 池内存。
- **运行时反查**：`graph_compute` 执行 MUL_MAT_ID 时从 `src0.buffer->buft` 反查池实例 → 拿该模型的
  `pool_base / slot_size / branch_layout`。每个池实例一个 buft，天然区分。

## 5. 实现落点

| 文件 | 改动 |
| :--- | :--- |
| `common/arg.cpp` | 新增 `--moe-draft-ram-pool` / `--moe-draft-vram-pool` |
| `common/common.h` | `common_params` 加 `moe_draft_ram_pool_mb` / `moe_draft_vram_pool_mb` |
| `common/common.cpp` | 主模型注入点不变（只取主池字段）；draft 注入点见下 |
| `common/speculative.cpp` | **新注入点**：加载 draft 前调 `route_b_setup(draft_path, draft_pool_mb, ...)`，draft overrides 挂 draft mparams |
| `src/server/route_b_inject.*` | `route_b_setup` 泛化：池实例注册表（每模型独立 buft/overrides/scheduler/拓扑）+ 预算切分逻辑 |
| `src/backend/moe_backend.*` | 支持多 weight buft；backend 持 `buft -> 池实例` 映射 |
| `src/backend/minigraph_exec.*` | `w3d` 按 `src0.buft` 定位池实例，再取 `branch_layout/slot_size/pool_base` |
| `src/backend/scheduler.*` | 每池实例独立实例化（复用现有多子池逻辑，模型 = 顶层组） |
| `scripts/run_deepseek_draft.bat` | 可显式加 `--moe-draft-ram-pool`（不写则 draft 全常驻） |

## 6. Bookkeeping 清单

- **新核心 key**：池实例 id（= weight buft 指针或自增 id）。
- **每池实例独立记账**：拓扑、池分配（组×槽）、`slot_meta`/`expert_directory`/generation、EST1 热度、DIO 文件句柄/read plan、`branch_layout`。
- **生命周期**：注册顺序 = 加载顺序（主模型先，draft 后）；销毁逆序（server 销毁 spec 时先释放 draft 池，主池最后）。
- **预算**：主池缺省 75% 空闲 RAM + draft 全常驻（10.3GB）——显式给主池大小时确认物理内存足够（128GB 主机一般 OK，页缓存可回收）。

## 7. 现存隐患（本设计强制解决）

- 若 draft 加载复用了主模型 `tensor_buft_overrides`（同名 `_exps` pattern），draft 专家会挂进主池 → 槽布局错乱。
- **铁律：overrides 必须 per-model 生成、per-model 挂载，绝不复用**。

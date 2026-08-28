# 该 review 已过时（2026-08-28 标注）

> 本文是 ChatGPT 对**早期 route-b-inject.patch** 的代码审查快照。此后架构已大幅演进：
> - 多模型专家池（main/draft 独立 pool）已实现
> - route B 已跑通并数值等价验证（gemma/v2/draft）
> - instrumentation 已拆分为 route-b-inject（持续）+ prefill-export（临时）两 patch
> - 有效未决问题见 `docs/REVIEW_2026_08_28.md`（当前状态总结）
>
> 阅读本文时请以当前代码为准，本文仅作历史审查参考。

可以。我认真看了一遍这个 route-b-inject.patch，先给你一个代码 review 第一轮结论：这个 patch 的思路是成立的，而且我觉得它已经不是“实验性地塞几行 hook”了，而是在把 StreamMoE 的 Route B 系统性地接进 llama.cpp 的模型加载、speculative、context、server 生命周期。

但同时，我看到了几个我会标成 P0/P1 级别 的风险，其中有一个尤其值得你马上检查。

总体评价

架构方向：8/10
代码侵入性：6/10（偏高）
正确性风险：目前中等偏高
可维护性：5~6/10
实验工程价值：很高

这个 patch 实际修改了 llama.cpp 的几个非常核心的边界：

common
 ├─ 参数
 ├─ model init
 └─ speculative draft

llama
 ├─ context
 ├─ KV cache
 ├─ model loader
 └─ model loading

server
 └─ lifecycle / loading

而真正的 StreamMoE 核心逻辑则留在外面的 src/：

server/route_b_inject.cpp
backend/moe_backend.cpp
backend/minigraph_exec.cpp
backend/scheduler.cpp
io/async_dio_win.cpp
io/staging_reader.cpp
loader/moe_loader.cpp
pool/expert_stats.cpp

也就是说，这个 patch 本质上是一个 adapter/injection layer，而不是把整个 StreamMoE 塞进 llama.cpp。这一点我认为是对的。


---

1. 我最喜欢的设计：expert pool 通过 tensor buft override 接进去

这一段是整个 Route B 最关键的架构：

params.tensor_buft_overrides = ovr;

然后 loader 看到 pooled tensor 就不正常读取：

if (!skip_pooled) {
    file->seek(weight->offs, SEEK_SET);
    file->read_raw(cur->data, n_size);
}

也就是：

GGUF
 │
 ├── dense tensor ──→ 正常 llama.cpp loader
 │
 └── expert tensor ──→ dummy/pool buffer
                         │
                         ↓
                    StreamMoE scheduler
                         │
                     SSD → RAM/VRAM

这个边界非常漂亮。

它意味着 llama.cpp 仍然认为自己在加载 tensor，但实际上 expert tensor 的 backing storage 被 StreamMoE 接管。

尤其这里：

> expert tensors on the pool buft carry a dummy data pointer



然后 loader 主动跳过 read_raw()，避免把数据写进 dummy address。

这比直接在 llama_model_loader 里面写一堆：

if (is_moe)
    ...

要干净得多。


---

2. 但这里有一个我认为必须检查的风险

if (tensor_buft_overrides) {
    for (const auto * o = tensor_buft_overrides;
         o->pattern != nullptr; ++o) {

        if (o->buft &&
            cur->buffer &&
            ggml_backend_buffer_get_type(cur->buffer) == o->buft) {

            skip_pooled = true;
            break;
        }
    }
}

这里判断的是：

> 当前 tensor 的 buffer type 是否等于 override 的 buft



而不是：

> 当前 tensor 是否真的匹配了 expert pattern。



这两者不是完全等价的。

如果未来出现：

同一个 buft
 ├── expert tensor
 └── 非 expert tensor

那么这里可能把非 expert tensor 的正常读取也跳掉。

目前如果你的 route_b_setup() 保证：

expert override → 独立 buft

那么没问题。

但我会强烈建议这里增加一个明确 invariant：

one pooled buft == only tensors intentionally owned by StreamMoE

否则这是非常隐蔽的 loader corruption bug。


---

3. mmap 强制关闭：方向对，但 scope 有点粗

这里：

if (use_mmap) {
    ...
    if (n_expert > 0) {
        ...
        use_mmap = false;
    }
}

注释甚至明确写了：

> MoE models never use mmap



也就是说这是：

只要发现 MoE
    ↓
整个 model 禁用 mmap



这和你最近一直在讨论的那个问题其实直接相关。

它意味着：

不是只有 expert 不 mmap，而是 dense weights 也跟着离开 mmap。

这可能是故意的，因为 StreamMoE 想要：

dense → deterministic buffer
expert → managed pool

但代价是很明显：

150 GB model
     │
     ├── 70 GB expert
     └── 80 GB dense

即使只有 70 GB expert 需要特殊管理，你也会放弃 dense 的 mmap。

所以如果你的目标是：

> 低 RAM 跑超大 MoE



我会把这个列成后续优化点：

不要把“expert 不 mmap”和“整个 model 不 mmap”绑定。

理想状态应该是：

expert tensor
    ↓
StreamMoE pool

dense tensor
    ↓
正常 mmap / read

这也正好对应你之前对 llama.cpp mmap 的吐槽。


---

4. route_b_setup() 给 draft model 单独建 pool：这是非常正确的

这一段我认为是 patch 里面非常重要的正确性修复：

auto * ovr =
    stream_moe::route_b_setup(
        model_path.c_str(),
        params.moe_draft_ram_pool_mb,
        params.cpuparams.n_threads,
        true);

if (ovr) {
    mparams.tensor_buft_overrides = ovr;
}

并且注释明确强调：

> draft model gets its own expert pool



否则如果 main 和 draft 共用：

main experts
draft experts
     ↓
同一个 pool

就会产生非常恶心的 ownership / eviction / lifetime 问题。

现在是：

MAIN
  ↓
main expert pool

DRAFT
  ↓
draft expert pool

这是正确的。




---

5. 但是 draft pool 的 lifetime 我会重点检查

这里注释说：

// draft pool's own (terminated) array

而 ovr 是：

auto * ovr = route_b_setup(...)
mparams.tensor_buft_overrides = ovr;

这里我会特别问：

> ovr 的生命周期到底是谁管理？



因为：

mparams.tensor_buft_overrides

显然不是一个普通 std::vector，而是一个 C-style terminated array。

如果：

route_b_setup()
    ↓
malloc/new array
    ↓
mparams 保存指针
    ↓
llama_model_load()

那么没问题。

但要确认：

model load 完成后
      ↓
ovr 是否还需要？
      ↓
model 是否保存这个 pointer？

如果 llama model 内部保存了 override 指针而不是深拷贝，那么这个地方非常容易出现 UAF。

我会让 agent 专门 audit 这一条 ownership chain。


---

6. Expert history 那部分：设计很好，但有一个巨大的潜在性能问题

你这里：

for (int t = 0; t < ids->ne[1]; ++t)
    for (int k = 0; k < ids->ne[0]; ++k)

然后：

export_expert_layer.push_back(...)
export_expert_token.push_back(...)
export_expert_id.push_back(...)

这其实是：

每 token
 × 每 layer
 × 每 routed expert

全部存起来。

对于你的 DeepSeek V4 Flash 这种规模，数据量其实会非常可观。

但你这里做得好的地方是：

> 你没有重新计算 routing，而是从已经执行的 graph 的 MUL_MAT_ID 节点里面取真实 routing ids。



这个非常重要。

尤其你已经处理了：

ids->nb[1]
ids->nb[0]

而不是：

ids[t * K + k]

这正好避开了你前几天踩过的那个 stride/padding 坑。



这个地方我给 正确性 +++。


---

7. 但 atoi(nm + 4) 有点脆

if (std::strncmp(nm, "blk.", 4) != 0) continue;
const int layer = std::atoi(nm + 4);

这依赖：

blk.0....
blk.1....

而且没有验证 suffix。

比如：

blk.foo

会得到：

0

虽然现实中 llama.cpp tensor name 不会这么离谱，但作为 instrumentation，我还是建议：

char * end;
long layer = std::strtol(nm + 4, &end, 10);
if (end == nm + 4 || *end != '.') continue;

不是大 bug，是 robustness。


---

8. LLM_EXPORT_DIR 这一套 instrumentation 最大的问题：会污染正常性能

例如：

if (std::getenv("LLM_EXPORT_DIR")) {
    ggml_backend_tensor_get_async(...)
    ggml_backend_synchronize(...)
}

你这里明显会：

GPU
 ↓
synchronize
 ↓
CPU
 ↓
copy

尤其：

ggml_backend_synchronize(b);

这会直接打断异步 pipeline。



所以它绝对不能拿来做正常 benchmark。

但作为：

> oracle trace / prefill cross-validation / cache policy simulation



这是完全合理的。

实际上我很喜欢这个设计，因为你现在获得了：

真实执行
   ↓
expert_history.bin
   ↓
offline simulator
   ↓
测试：
  70GB pool
  80GB pool
  90GB pool
  LRU
  LFU
  oracle
  aggressive prefetch

这对于你现在这个项目的研究价值非常高。


---

9. KV export 会非常吃内存

这里：

std::vector<uint8_t> buf((size_t) nb);

然后：

ggml_backend_tensor_get_async(...)
ggml_backend_synchronize(...)
fwrite(...)

也就是说最终 KV cache 会：

VRAM
 ↓
CPU temporary buffer
 ↓
file

如果是你现在这种超长 context：

这可能是 GB 甚至几十 GB 的一次性 allocation。

所以我会把它标成：

P1：实验模式下可以，生产模式不能。

最好以后变成 chunked streaming：

tensor
 ↓
8/64/256 MB chunk
 ↓
file
 ↓
next chunk


---

10. 另一个值得警惕的是 destructor 做大量工作

你现在：

~llama_context()
{
    ...
    export_prefill_final();
    export_expert_history_final();
    ...
}

这意味着：

用户退出 session
        ↓
析构 context
        ↓
写 expert history
        ↓
写 prefill
        ↓
读 KV
        ↓
可能 GPU synchronize
        ↓
大量磁盘写
        ↓
真正退出

所以如果用户：

Ctrl-C

或者 server：

destroy context

可能突然卡很久。

而且更重要的是：

异常退出时这些数据可能完全丢失。

我会更推荐：

正常运行
 ↓
incremental trace
 ↓
periodic flush

而不是全压到 destructor。


---

11. 我比较担心的一个架构问题：patch 混入了太多“实验工具”

现在这个 patch 同时做：

Route B
+
expert pool
+
draft pool
+
KV placement
+
expert history
+
prefill export
+
timers
+
KV reporting
+
speculative statistics

例如：

sm_tmr::timer _t("common_init_from_params");

KV Cache Memory ...

LLM_EXPORT_DIR

common_speculative_print_stats()

这些其实不是 Route B 本身。

所以从代码审查角度，我会建议未来拆成：

route-b-core.patch
    ↓
真正运行所必须

instrumentation.patch
    ↓
expert history
prefill export
timers

experiment.patch
    ↓
KV placement
benchmark flags
debug logging

这样会舒服很多。


---

12. 但有个地方我反而认为做得非常好：没有大面积 fork llama.cpp

整个 patch 对 upstream 的侵入主要集中在：

common 参数
common model init
speculative model loading
llama context
KV getter
model loader
server lifecycle

而不是：

把 llama.cpp fork 一遍
然后到处改 MoE

这很重要。

所以你的总体结构实际上是：

upstream llama.cpp
                         │
             ┌───────────┴───────────┐
             │                       │
        normal tensors          expert tensors
             │                       │
       normal llama path       tensor buft override
                                     │
                              StreamMoE Route B
                                     │
                         ┌───────────┼──────────┐
                         ↓           ↓          ↓
                       RAM         VRAM        SSD

这是我认为这个项目目前最值得保留的架构。


---

我的 review 结论

如果让我现在给这个 patch 提交 review：

🔴 必须检查

1. tensor_buft_overrides 的 ownership/lifetime


2. pool buft 是否可能承载非 expert tensor


3. draft pool 和 main pool 的生命周期


4. 关闭 mmap 是否误伤 dense tensor


5. destructor 中 export 是否可能阻塞/失败丢数据



🟡 建议修改

1. atoi() → 更严格的 layer name parsing


2. export 改成 incremental/chunked


3. instrumentation 和 Route-B core 分离


4. KV export 不要一次性申请整个 KV


5. expert history 最好提供开关，不要让 instrumentation 默认存在



🟢 我认为设计正确

1. tensor-buft override 作为 expert ownership boundary


2. main/draft 独立 expert pool


3. loader 对 pooled tensor 跳过正常 read_raw


4. 从实际执行 graph 的 MUL_MAT_ID 获取 routing history


5. 正确处理 ids->nb[] stride


6. Route B 作为 injection layer，而不是把 StreamMoE 核心塞进 llama.cpp



尤其第 5 点，我知道你前几天刚被那个二维 ids stride 坑了一次——这里这个 patch 反而已经是正确处理方式了。

所以我的总体判断是：

> 这不是“agent 拼出来能跑就算”的垃圾 patch。它已经有相当明确的系统架构意识。真正的问题不是核心 Route B 思路错，而是一个 patch 里塞进了太多 instrumentation，以及几个 C++ ownership / lifecycle 边界需要严审。



如果你愿意，我下一步可以直接：，按 **真正的 code review 方式**来做。这会比现在这种架构层 review 更容易直接拿给 agent 改。 把这 806 行逐个 hunk 过一遍给每个修改标 **P0/P1/P2/P3 + 具体 bug/潜在 bug + 修改建议**
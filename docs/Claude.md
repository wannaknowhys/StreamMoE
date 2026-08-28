# 该 review 已过时（2026-08-28 标注）

> 本文是 Claude 对**早期 moe_backend/scheduler/moe_loader 骨架**的审查快照。
> 关键过时点：文中"graph_compute 故意失败、--expert-backend 跑不起来"——**现已实现并跑通**
> （gemma/v2/draft 数值等价验证）。有效未决问题（单线程调度、驱逐忙等、打分魔数）见
> `docs/REVIEW_2026_08_28.md`。阅读时以当前代码为准。

看了这三个文件,信息量很大,先说最重要的一点:

## 最关键的发现:后端目前还跑不起来

`moe_backend.h` 的注释写得很直白:
> "Skeleton status: device/buft registration + support queries are real; **graph_compute currently rejects execution (GGML_STATUS_FAILED)** until the per-expert mini-graph delegation lands (next milestone)."

也就是说——buft 注册、内存记账(accounting-only,`alloc_buffer` 不做真实物理分配)、scheduler、loader 这些"骨架"都写好了,但真正执行 MoE 算子的 `graph_compute` 目前是**故意返回失败**的。换句话说,`--expert-backend` 现在跑起来会在真正需要计算专家权重时直接失败,不是"能跑但慢",是"根本还没接通"。之前 patch 里那些侵入 `common.cpp`/`llama-model-loader.cpp` 的路由逻辑,其实是在给一个还没有执行能力的后端搭桥——这个顺序本身没问题(先打通配置/加载路径,再补计算),但评估现状时要明确:**这是一个进行中的原型,不是能用的 feature**。

## scheduler.cpp:核心设计问题

**1. 单线程 scheduler_loop 是最大的架构隐患**
所有 expert miss 都塞进一个队列,由**唯一一个 worker 线程**顺序处理,且 `load_slot` 里的 `read_expert_sync` 是同步阻塞读。对于 top-k 路由的 MoE(一个 batch/ubatch 里经常同时 miss 好几个不同专家,尤其是 prefill 阶段),这意味着所有跨专家的加载请求会被这一个线程严格串行化——不管你的 IO 有多快,专家换入的吞吐上限就是"单线程顺序 blocking read"。这是我认为最值得优先解决的性能瓶颈,通常这类系统会用多个 worker 线程,或者用真正的异步 IO(多个 in-flight 请求并发提交)来打开并行度。

**2. 驱逐时的忙等没有超时保护**
```
while (slot_word_refcount(slots_[victim].load()) != 0) {
    std::this_thread::yield();
}
```
如果某个 pin 者因为 bug 没有正常 `unpin()`(比如上层代码路径有异常提前返回、忘记调用),这里会**无限自旋**,而且这是发生在唯一的 scheduler 线程里,一旦卡死,整个 pool 的后续请求全部停摆。建议加超时 + 诊断日志(哪个 layer/expert 卡住、refcount 是多少),而不是无界 yield。

**3. 驱逐打分公式里藏了个魔数**
```
score = 0.5 * freq + 0.5 * (1.0 - generation / 1e9)
```
用固定常数 `1e9` 归一化 generation 而不是相对当前最大 generation,长时间运行或高频场景下这个 recency 项会逐渐失真(所有 slot 的 generation 都远小于 1e9 时区分度不够,或者反过来早晚会溢出这个假设)。另外 `best_seq` 算出来后只在最后 `(void)best_seq;` 消除警告,是死代码,建议删掉或者真正用上。

**4. pin_expert 重试逻辑有轻微冗余**
等待加载完成的分支里,每次 `wait_version()` 返回后如果状态还是 UNASSIGNED 会**再次 push 一个新请求**,可能造成重复请求入队(不会导致重复加载,因为串行的 scheduler_loop 在 pop 时会用 `dir_->find` 去重,但会有些许队列膨胀和无谓的锁开销)。

## moe_loader.cpp:总体质量不错,但有几个值得注意的点

**做得好的地方**:
- 分片(shard)发现逻辑对"分片数量不匹配""分片缺失"都**直接抛异常而不是静默继续**,这个"fail hard"的态度是对的,避免加载出一个悄悄损坏的模型。
- v1 路径对 `_exps` 张量做了三层校验才敢按 `expert_size = total_size / n_expert` 切片:整除性检查、`ne[2] == n_expert` 检查、量化 block size 对齐检查——这是很扎实的防御性编程,能在源头拦住"切片公式假设不成立"导致的静默数值错误。

**需要注意的地方**:
- 这套简单的按偏移切片（`e * slice_bytes`）隐含一个假设:GGUF 里 `_exps` 张量在文件中的物理布局是以专家维度(`ne[2]`)作为最外层、连续存放的。对 llama.cpp 自己写出来的标准 MoE 张量这通常成立,但这是一个"没有显式验证、只是隐式假设"的前提,建议在代码里加一句注释明确写出这个物理布局假设,而不是只验证维度数值对不对。
- "v2 expert-blocks" 是一套完全自定义的 GGUF 扩展布局(靠 `stream_moe.expert_sections` 等自定义 KV 驱动),这意味着模型要先过一遍专门的转换工具才能用 v2 路径。这会让你同时维护两套逻辑(标准 GGUF 切片 + 自定义预打包格式),测试和维护成本翻倍,而且用户想用 v2 必须先跑你的转换器,这对易用性是个隐性负担,值得在文档里说清楚"什么情况该用 v1、什么情况该转 v2"。
- 这个文件里所有异常都是直接 `throw`,没有在本文件内 catch。如果调用方(`route_b_setup` / 上层 `common_model_params_to_llama`)没有妥善捕获,模型加载失败会变成整个进程崩溃而不是优雅的错误返回给上层调用者(CLI 报错退出还行,但如果 server 常驻进程,加载失败最好能返回错误而不是直接挂掉)。值得确认一下调用链上有没有 try/catch。
- header 注释提到支持 `ffn_*_shexp`(shared expert),但这个文件里 `sub_tensor_names` 只列了 `_exps` 四种,没有 `_shexp` 的处理——和之前 patch 里"共享专家暂时留在默认后端"的说明是一致的,不是 bug,只是提醒一下:文档描述的"model-agnostic 支持"目前实际上还没覆盖 shared expert。

## 小结
`moe_loader.cpp` 的解析/校验逻辑是这三个文件里质量最高的部分,防御性做得到位。`scheduler.cpp` 的状态机思路(slot word 打包 state+refcount+generation)是对的,但单线程串行调度会是实际吞吐的硬伤,驱逐时的无界自旋是稳定性隐患。最重要的是心里要有数:**graph_compute 现在是故意失败的**,所以现阶段这套东西的价值主要在"加载/调度骨架已经打通",距离"能跑通推理"还差最后一块——per-expert mini-graph 的实际执行代理,这才是接下来真正要啃的硬骨头,建议优先看你打算怎么实现这部分(以及打算怎么解决上面提到的单线程瓶颈,因为一旦 graph_compute 打通,吞吐问题会立刻暴露出来)。
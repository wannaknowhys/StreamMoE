[English](ROUTE_B_GPU_PHASE.md) | [简体中文](ROUTE_B_GPU_PHASE.zh-CN.md)

# Route B Multi-Device Expert Pool - Design (Phase B)

> Status: design, **revised 2026-09** (supersedes the 2026-08-29 draft). Builds
> on route B third path (official `ggml_mul_mat_id` kernel + uniform-stride slot
> pool, `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7).
> Verified facts carried over: Vulkan and CUDA natively support `MUL_MAT_ID`
> with the pool-slot 3D layout; RX 590 exposes a real 8 GiB
> `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` heap; GGML_VULKAN=ON builds clean.
> Related: `docs/MULTI_MODEL_POOL.md`, `docs/MULTI_SUBPOOL.md`, `docs/Backend.md`.

## 1. Goal and mainline

Selected experts live across multiple device pools (CPU RAM + GPU VRAM), each
bounded by its pool budget; GPU is a first-class compute node (not storage).
MoE execution is **not** per-node on the llama graph: the stream_moe backend
takes a **whole MoE layer** in `graph_compute` and **self-schedules** it - read
ids, resolve each expert's pool, dispatch per-device subsets, run each subset's
expert chain inside that device, merge results, write the layer output back.

Confirmed mainline (2026-09): **compute takes the whole layer and self-schedules
to each device** - required, not optional.

## 2. Current mechanisms (what stays, what extends)

- **Why a backend**: registering a ggml backend/device gives us the scheduler's
  formal `graph_compute` callback. `supports_op` claims only the MoE expert
  `MUL_MAT_ID` domain (+ views whose source is our host compute buft); the
  scheduler hands us exactly those nodes, once each, correctly ordered.
- **mini-graph today**: each original `MUL_MAT_ID` is re-executed in a private
  leaf-only mini-graph with the **official kernel**: `w3d` leaf fakes the pool's
  compact slots as `[ne0, ne1, group_slots]` (data = pool base + branch offset,
  nb[2] = slot stride), `ids_slot` translates expert ids -> slot index, `b_leaf`
  wraps the main-graph activation, result writes the main dst. CPU-only, zero
  copy (all one RAM).
- **Notify protocol**: compute -> scheduler = MPSC push + sleep on a per-(L,E)
  version word; scheduler -> compute = version bump + `WakeByAddressAll`/futex
  wake. This protocol is **device-agnostic and stays**.

## 3. Whole-layer self-scheduling

The llama main graph still builds every MoE layer node (`can_reuse`/plan
stable). `supports_op` extends to the layer's whole MoE chain domain. At
execution `graph_compute` receives one layer's MoE chain and runs:

1. read ids (`ffn_moe_topk-N`; gating stays on the graph / dense side - ids arrive computed)
2. resolve every (L,E) to its pool slot (directory scan, pin/wait as today)
3. choose the **lead device** (see §5)
4. dispatch per non-lead device a subset (ids slice + weights slice)
5. per device: run its expert chain in the device's fixed arena (official
   `mul_mat_id` + official silu/mul/add on that backend); big intermediates
   never leave the arena
6. merge expert contributions (embedding-dim, small) back to the lead device,
   write main `ffn_moe_out-N` dst

Dynamic (which experts, which device, lead choice) lives entirely in the
execution layer; the graph/reuse shape is static.

**Execution boundary**: the whole per-layer MoE chain is executed by us into our
own fixed arenas; the main graph only carries the two ends - layer input (`cur`)
and layer output (`ffn_moe_out-N`). Cross-device "copying" is not a physical law
- it only appears when intermediates live in llama-managed buffers where the
scheduler inserts cpy by backend ownership. Intermediates in our own arena are
never copied; the price is the chain execution is ours to orchestrate and the
main graph cannot keep individual MoE-chain nodes between the two ends.

### 3.1 Privatization admission check (fail-fast, no fallback)

Hiding intermediates (real results in our arenas, main-graph node dst left
uncomputed) is only safe if no node OUTSIDE the layer's MoE chain consumes them.
We verify, loud, and refuse to run:

- **Hook**: after each of the three `model.build_graph()` call sites
  (`llama-context.cpp`: process_ubatch / graph_reserve / llama_encode), call
  `stream_moe_verify_graph(gf)` - our function; the chain/exemption logic lives
  on our side. verify only runs when the graph is actually rebuilt (`can_reuse`
  skips build entirely).
- **Check 1 - no external consumer**: for every privatized MoE-chain
  intermediate, its consumers in the whole graph must lie inside the same
  layer's MoE chain (eventually converging on `ffn_moe_out-N`). Exemptions:
  `ffn_moe_out-N` itself (output end, consumed by residual add); the gating
  segment (dense-domain, not privatized - not checked).
- **Check 2 - chain integrity**: our `graph_compute` verifies the split it
  received actually contains the full expected chain for the layer (all
  expert-buft `MUL_MAT_ID` + the chain views/silu/mul/add). A missing/misplaced
  node means the scheduler's heuristic backend expansion put a chain node on the
  wrong side - the failure mode is loud, not a silent wrong number.
- **Failure mode**: any violation -> log full context (intermediate, external
  consumer or missing node, layer, arch) and exit. **No silent fallback, no
  escape hatch.**
- **Narrow privatized set**: when unsure whether a node is chain or gating,
  keep it out of the privatized set rather than risk a false positive.

### 3.2 Why the whole layer lands in one split (scheduler mechanics, verified)

ggml-backend-sched (`ggml/src/ggml-backend.cpp`) splits by contiguous backend:

- **pass1-4 (backend assignment)**: nodes with weights follow the weight
  buffer's backend; weightless nodes are expanded to adjacent backends (GPU
  priority, CPU lowest), gated by `supports_op`. View/reshape/transpose have a
  **deterministic rule** (pass4: views always share their `view_src` backend).
- **pass5 (split)**: a run of nodes on the same backend forms one split; a new
  split starts on a backend change or incompatible weight input.
- **Cross-split inputs** (lines ~1400-1420): an src on a different incompatible
  backend gets an automatic copy in the split's backend and `node->src` is
  re-pointed at it.

Consequences: if `supports_op` claims the whole layer chain, the chain is
contiguous -> it lands as **one split** in our `graph_compute`. The gating
segment stays in the dense split; ids / weights / `cur` arrive as our split's
inputs - **the scheduler copies them, no manual broadcast at the llama layer**.

### 3.3 The two ends: cur / moe_out

- **cur**: the scheduler handles the cross-split input copy. Optimisation: the
  copy happens only if `ggml_backend_sched_buffer_supported` fails - if our
  `supports_buft` accepts CPU's host buft (both host memory), cur is shared
  zero-copy by pointer. Broadcasting cur to the per-device pools **inside** our
  self-scheduler is our own staging (outside the llama layer).
- **moe_out**: chain output-end node -> its backend is ours -> gallocr allocates
  its dst in our buffer -> we write the real merged result there (the graph
  output end must be real - the residual reads it). The residual add (dense
  backend) consumes it; the scheduler handles that cross-split hop (host share
  or small copy).
- **Hidden nodes** (chain intermediates): we do not truly compute their dst, but
  gallocr still allocates space for them in our buffer (dead space - cheap host
  memory; can be optimised later). Check 1 guarantees no external reader, so a
  hole is safe.

### 3.4 Async discipline: graph_compute exit synchronisation

Scheduler-level cross-split ordering is explicit and **independent of copy
nodes** - at each split boundary the scheduler synchronises the previous
backend's async work (event or `synchronize`, because the allocator reuses
buffer regions across splits). So a zero-copy host pointer share does **not**
lose ordering between our split and the dense split.

The real ordering risk is **inside our self-scheduler**: `graph_compute`'
contract is "return = this split finished, dst ready". If we asynchronously
submit per-device sub-chains (e.g. Vulkan queues) and return without waiting,
the next split trusts a dst that is not ready. Discipline (required, not a
test): **before graph_compute returns, wait on all internally-submitted
per-device work** (backend event / synchronize per device), then write moe_out.
CPU-only today is naturally synchronous; the rule bites with GPU async - keep a
"slow-down and check" regression at M3.

### 3.5 Executor implementation decisions (M1/G3, fixed)

- **Single chain-node predicate** `moe_chain_node_is_privatizable(node)`, shared
  by `supports_op` (collection backstop), verify Check1/Check2, and the executor
  traversal (one definition, one module, `route_b_chain.*`). Judgement: src-chain
  reaches an expert-buft `MUL_MAT_ID` / hidden `ffn_moe_*` intermediate name /
  the moe_out end.
- **Collection is EXPLICIT, not heuristic** (2026-09 correction): declaring the
  ops in `supports_op` is NOT enough - the scheduler also runs these generic ops
  (geglu/mul/add) on the CPU backend and won on best-supported ties, so the
  chain never lands as one split. Instead, right after `build_graph` (the verify
  hook), assign every privatizable chain **compute** node to the stream_moe
  backend via `ggml_backend_sched_set_tensor_backend` - the scheduler's pass1
  "do not overwrite user assignments" respects it, so the chain forms one split.
  View/layout nodes are left to pass4 (they follow `view_src` automatically).
  `supports_op` still declares privatizable as a backstop.
- **Privatization lands in one step (B2)**: hide intermediates immediately.
  verify Check1 ships together with G3 as the safety net.
- **pin/unpin lifecycle**: phase 1 collects ids ONCE (gate_up and down share the
  same ids) -> pin (missing experts push a request to the scheduler and wait on
  the version word) -> run the whole chain -> **unpin after the merge node
  (chain-tail add / moe_out)** - experts are used to the very end of the chain
  (moves later than today's release-after-down).
- **verify Check1 verdict is cached**: the external-consumer relation is
  topology-only (arch/layers), shape-independent - an architectural conclusion.
  Cache it by **model-topology signature** (arch + n_layer + per-layer config;
  main and draft are separate models). First full-graph scan records
  `{signature: PASS}`; later rebuilds of the same signature skip scanning.
  Check2 (cheap execution-side node count) still runs every execution and is not
  cached. The verdict cache lives in the route_b_chain module; the privatized
  set is the predicate's output - two distinct things.

## 4. device_pool[] + fixed per-device execution arena

Current single-pool code already keeps a pool dimension in the directory
(entries is (L,E) x pool, `scan` iterates pools) - keep that. What changes is
physical slot addressing (today one base carved into layer groups):

```
device_pool[] = {
  ggml_backend handle,
  physical base (VRAM or RAM), slot stride, capacity,
  per-pool layer-group carve (subpools_), per-pool slot meta,
  fixed execution arena,
}
```

**Fixed arena per device**: one preallocated block sized for the worst-case
layer the device will process:

```
arena = [ staging(cur copy) | exec region (chain intermediates overwrite in
          place) | result (expert contributions) ]
```

Value is not saving allocation - it makes per-device expert chains run with
intermediates never leaving the block, so cross-device traffic is only the small
items (§5). Watch: in-place overwrite ordering per chain step (layers are serial
so safe; explicit read/write windows needed).

### 4.1 Scheduling: slot-level mechanics (per-pool local)

Mechanism level operates on **slots**, not on an expert-placement policy:

- the directory records (L,E) -> (pool, slot); one expert may hold several
  positions at once (non-inclusive, non-exclusive read-only replicas); `scan`
  finds the first usable.
- a pool is a bounded set of slots with its own budget and its **own local
  eviction**: eviction happens because THAT pool is full - look inside that pool
  for what leaves; never a cross-pool judgement about the expert.
- primitives: load content into a slot (DIO into RAM or VRAM), move slot content
  between pools (RAM<->VRAM; no VRAM<->VRAM - a GPU-to-GPU move is not worth it,
  re-read/DIO instead), release a slot (evict that copy).
- replicas make eviction per-copy: each pool evicts its own slots.

## 5. Gate vs expert chain: the boundary rule

**Boundary judgement**: the gate segment (`cur x gate_proj` = dense mm,
softmax, topk) only touches the dense `gate_proj`; topk produces ids but never
fetches an expert weight by id. The chain segment (`mul_mat_id(w, cur, ids)` for
gate_up/up/down) **accesses expert weights by id** - that is what needs the
pool. So the rule is **"does the node access expert weights by id"** (not "does
it have an id tensor").

**Implementation**: don't guess by name - check `src[0]`'s buffer type: an
expert-buft `src[0]` (MUL_MAT_ID) is chain; everything else (dense gate_inp,
etc.) is gating. Weightless chain nodes (view/silu/mul/add) are attributed by
backend expansion + Check 2 (chain integrity) makes that safe instead of
hoped-for.

Placement rules inside a layer:

- **Gate** is a dense computation -> runs on the layer's **dense device**
  (static, decided by layer placement; dense weights are never moved for
  gating). ids + per-expert weights fan out to the executing pools (small:
  n_used x tokens).
- **Lead device** (merge point) = pool holding the **most of this run's
  selected experts**, chosen AFTER every expert is pinned/resolved (the actual
  per-pool distribution of this run is known then - argmax, no heuristic).
- **cur broadcast**: every executing pool needs cur - one copy per pool
  (unavoidable minimum).
- **Merge**: lead sums its own + returning contributions (embedding-dim, small)
  -> `ffn_moe_out-N` -> back to the dense device for residual/add/norm.

Data flow: big intermediates never cross devices; only ids/weights/cur
broadcast/expert contributions/moe_out move - all small.

## 6. dense / KV per-layer assignment (independent follow-up)

Decouple dense weights per layer and assign a device per layer (static, no
runtime migration); KV follows the layer's dense device (e.g. last N layers'
dense + KV on the RX 590). Value: dense device = gate device = ideal lead for
that layer -> zero cross-device.

- llama's native layer offload is **contiguous** (`-ngl N`) - an arbitrary
  per-layer table ("layers 27-29 on GPU") is a loader/`dev_layer` change
  (independent work item).
- Only the expert pool is dynamic/migratable; dense + KV are fixed at
  load/init. Size the VRAM budget per layer before doing this.

Not blocking the mainline: M1/M2 first, exercise native contiguous offload,
then decide whether a custom layer table earns its keep.

## 7. Numerical / correctness baseline

- Per-expert results use official kernels + same weights -> identical;
  expert-add order may differ -> ulp. Gate on v2/deepseek alignment (route
  IDENTICAL + cos ~ ulp).
- Pool size must never change numerics (8192 vs 71680 already verified
  IDENTICAL) - keep as a regression guard.
- Boundary misclassification is caught structurally by Check 2 before any
  number is produced (not by end-to-end value checks). Add one targeted
  regression at M3 (a weightless node adjacent to both dense and expert
  backends) as a numeric backstop, but it is not the primary guard.
- Any two backends show ~5% routing flips + hidden amplification - inherent
  noise (docs/BACKEND_DIVERGENCE_ANALYSIS.md), not a bug.

## 8. Milestones (risk-decreasing)

- **M1** - CPU-only private-chain skeleton: per-layer private mini-graph
  (topology replayer for the gemma arch; official kernels, intermediates in our
  arena, only cur/moe_out on the main graph) + two fake device_pools (both CPU,
  independent pools + arenas) to pin down ids-grouping / per-pool subset chain /
  merge logic and numerics == official (v2 alignment). Defines the
  self-scheduler interface and the chain-topology replayer.
- **M2** - device_pool[] lands (backend handle / slot addressing / fixed arena);
  one real Vulkan device + CPU mixed, exercise gate-follows-dense + lead-merge
  live + the graph_compute exit sync discipline.
- **M3** - same-layer cross-device dispersion (2 GPU0 / 2 GPU1 / 3 CPU style) +
  arena read/write-window discipline at full speed + slow-down correctness
  regression.
- **M4** - device migration (directory set/clear + move primitives).

## 9. Kept verified facts (build / hardware)

- Vulkan + CUDA native `MUL_MAT_ID` with the slot-3D layout - the w3d trick is
  backend-portable.
- RX 590 8 GiB BAR1 is a real `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` heap -
  zero-copy DIO into mapped VRAM remains available once a device pool exists.
- `GGML_VULKAN=ON` builds clean; `libomp.dll` must sit next to the exe.
  Realistic target stays: dense on Vulkan + hot experts on GPU + most experts
  in RAM.

## 10. Structural clarifications (2026-09, post-review)

Calibration notes so the design is not misread later.

1. **Execution form = per-node mini-graph with private data, NOT a chain
   replayer and NOT a second tree.** "Execute the original, own the data":
   `graph_compute` receives the layer's original chain nodes as-is (we never
   rebuild topology). For every compute node (mul_mat_id / activation / mul /
   add - anything non-view) we build a one-node mini-graph: input leaf points at
   (main-graph inputs cur/ids/weights, or the previous result already in our
   arena), the official kernel runs, dst data lands in our private arena
   (moe_out is the exception - it writes the main-graph dst). View nodes are not
   built: the consumer's input leaf is just upstream data + view offset (C-style
   pointer math). No shadow chain is maintained - per-node mini-graphs are
   independent, chained by arena slots; the order is our traversal of the split.
   Main graph only touches cur (in) and moe_out (out).
2. **Cross-split synchronisation is accepted.** Per layer = one dense split +
   one of-our split alternation is the original sched structure anyway; sync
   cost vs tokens/s is not a bottleneck - do not optimise for it.
3. **Verify**: Check 1 (build-time: no external consumer of a privatised
   intermediate) is the admission gate. Check 2 (chain integrity inside
   `graph_compute`) stays as a cheap execution-side self-check - not a
   cross-layer interface.
4. **Scheduling is slot-level and per-pool local** (§4.1): eviction happens only
   because that pool is full; the directory is a location record, not a policy.
5. **No per-arch chain replayer is needed**: because we execute the original
   nodes one mini-graph at a time (item 1), topology comes from the main graph -
   gemma's fused gate_up+view and deepseek's separate gate/up/down both work by
   construction. The per-op leaf-wrapper coverage list (which ops we wrap) is
   small and arch-independent. M1 enumerates the gemma ops actually hit.
6. **Multi-copy pipeline (n_copies > 1)**: coding note - the split must read
   the `cur_copy`'s inputs; not a design-direction issue.

# Per-Device Arena Plan (carry split, per-device compact, C1/C2 placement)

[English](PER_DEVICE_ARENA.md) | [简体中文](PER_DEVICE_ARENA.zh-CN.md)

> Status: **design, 2026-09-14**. Follows the promotion of whole-layer ownership
> to production (`LAYER_EXECUTOR_DESIGN.md`). The single host arena is too big
> and ignores `--dense-placement`; this doc plans the per-device arena.
> Related: `ROUTE_B_LAYER_OWNERSHIP.md` (target shape, §3.4/§3.5/§3.8),
> `M2_DEVICE_EXECUTOR.md` (device executor), `DENSE_PLACEMENT.md` (C1/C2),
> `GRAPH_PARTITION.md` (regions), `LAYER_EXECUTOR_DESIGN.md` (§4.3 arena).

## 1. Problem

After whole-layer ownership became production, `layout_arena` pre-allocates
**every** compute node into one host arena laid out as `[carry][compact]
[closure]`. Measured temp space at ubatch 512 (host bufts):

| model | carry | compact | closure | need |
| :-- | --: | --: | --: | --: |
| olmoe | 62 MB | 218 MB | 142 MB | 0.4-0.5 GB |
| gemma | 160 MB | 2498 MB | 198 MB | 2.8 GB |
| deepseek | 2630 MB | 1300 MB | 248 MB | 4.1 GB |

Three concrete defects:

1. **carry accumulates.** Every cross-layer tensor gets a unique, never-reused
   offset. Measured: olmoe 15 `l_out-N`, gemma 29, deepseek 84 (43 `l_last-N` +
   41 hyper-connection `node_NNN`) all coexist -> deepseek 2.6 GB at ub 512.
2. **compact is unpacked.** It is the byte SUM of the worst layer's nodes, with
   no within-layer liveness reuse. gemma's 2.5 GB is dominated by the dense MLP
   and the C2 lm_head logits.
3. **C1/C2 placement is overridden.** `--dense-placement C1:<dev>,C2:<dev>`
   sets the buft, but `layout_arena` rewrites the node's `buffer`/`data` to the
   host arena, so the placement is ignored and the logits always land on host.

## 2. Carry split: cross-1 vs cross-N

**Analysis (verify-time, dynamic).** For every captured cross-layer tensor,
compute `distance = max_consumer_layer - producer_layer` (using the layer
attribution built in `collect_layer_nodes`).

- `distance == 1` -> **cross-1**: a pipeline tensor (layer L's output consumed
  by L+1). Only a bounded number are live at any time -> reusable.
- `distance > 1` -> **cross-N**: live across many layers -> must be retained.

**Measured (ub=1):** all three models are **100% cross-1, cross-N = 0**:

| model | cross-1 | cross-N |
| :-- | :-- | :-- |
| olmoe | 15 nodes / 122880 B (max 8192) | 0 |
| gemma | 29 nodes / 326656 B (max 11264) | 0 |
| deepseek | 84 nodes / 5505024 B (max 65536) | 0 |

**Layout.** `carry1` is **double-buffered** on the layer-boundary index:
the set of cross-1 tensors produced by layer L and consumed by L+1 is packed
into buffer `L % 2` (two buffers so `O_L` and `O_{L+1}` coexist at L+1's tail,
where the residual is read and the next output is written). `carry1` size =
`2 * max over boundaries of pack(boundary_set)`. `carryN` is packed once and
retained (empty for the current three models, kept for generality).

Projected at ub 512: olmoe 62 MB -> ~0.3 MB, gemma 160 MB -> ~11 MB,
deepseek 2630 MB -> ~67 MB.

## 3. Per-device plan (finalized 2026-09-14)

`layout_arena` becomes a per-device planner. One `region_plan_t` per device.
Regions are grouped by **liveness**, not by stage:

```
struct region_plan_t {
    std::string dev;                     // "" = host
    ggml_backend_buffer_t buf;           // grow-only, one per device
    // carry: the cross-layer pipeline (logically one; physically per-device)
    size_t carry1_bytes;                 // 2 x max boundary pack (layer parity)
    size_t carryN_bytes;                 // retained cross-N
    // scratch: every within-layer temporary of this device, MERGED:
    //   compact (dense head/tail) + closure (MoE)
    size_t scratch_bytes;                // max over layers
    size_t off_carryN, off_scratch;
    std::unordered_map<const ggml_tensor*, size_t> carry1_off, carryN_off, scratch_off;
    std::set<const ggml_tensor*> cross_device;   // carry a consumer reads remotely
};
```

**Two regions, by liveness.**

- **carry** (cross-layer): the residual stream / layer outputs. **Logically one
  pipeline**; physically each device reserves its own grow-only slots. When
  layer L's carry is consumed by layer L+1 on another device, the executor moves
  it **explicitly** (D2D / transfer-queue; host boundaries via staging) before
  L+1 first reads it - not implicit, not duplicated. CPU-only = one device = no
  move.
- **scratch** (within-layer): `compact` (dense head/tail) **merged with
  `closure`** (the MoE computation). Both are per-layer temporaries reused
  across layers, so on a device they are one pool. The executor's closure
  offsets (`moe_chain_fullalloc_buffer`) index into this merged pool instead of
  a separate closure block; `closure` is no longer its own region.

**Closure is per-device by construction.** The MoE computation runs where its
experts live, and every device has its own expert pool, so the closure is
naturally per-device - there is no "which device for the closure" choice.

**Node -> device:**

- **C1 dense** (attention / norms / dense MLP / gating) -> its
  `--dense-placement` device. C1 may be split across devices and may migrate
  (`DENSE_PLACEMENT.md`).
- **C2 output head** (`result_norm` / `result_output`, the logits) ->
  `C2:<dev>`, whole (never split).
- **MoE closure** -> the device of that layer's expert pool.
- **carry** -> the producing layer's device (moved at a boundary, above).
- Default (no placement): one host plan.

**Device resolution (landed).** A captured node's device is the device that
owns its weight operand (dense layers -> their placement device, via the weight
buffer's device; experts -> their pool device); weightless nodes inherit from a
producer. Route B's own host backend and the CPU device collapse to the host
plan. The expert pools are recorded at `route_b_setup`.

**Cross-device** tensors (a node whose consumer is on another device) are marked
in the plan so the executor moves them (D2D / the transfer queue,
`ROUTE_B_LAYER_OWNERSHIP.md` §3.5) - never host staging except at host
boundaries.

## 4. Reuse the existing packing

Extract the best-fit-decreasing interval packing currently in
`moe_chain_verify_graph` (the closure result layout, `route_b_chain.cpp`) into
`pack(nodes, last_use) -> { offsets, size }`. Use it for the merged `scratch`
(`compact` ∪ `closure`), `carryN`, and each `carry1` boundary set. No second
allocator.

## 5. C1/C2 placement semantics

`--dense-placement C1:<dev>,C2:<dev>` (already parsed) is the single source of
truth for C1/C2 placement. The arena for a device uses that device's buffer
type; owned C1/C2 nodes get their `buffer`/`data` from the per-device arena
instead of a fixed host arena. **The logits follow C2.**

Two layers of work:

- **Plan/allocation layer** (this doc, phase 1-2): can land now; with CPU-only
  placement every device collapses to one, so numerics are unchanged.
- **Execution layer**: actually running C1/C2 on the placement device needs the
  device executor (`run_dense_subgraph` currently always calls the CPU backend).
  That is `M2_DEVICE_EXECUTOR` (phase 3).

## 6. Implementation order

1. **carry double-buffer + `pack` extraction.** No device dependency; immediate
   memory win; verify olmoe/gemma/deepseek numerics + `run_baseline`.
2. **per-device plan structure.** Node -> (device, region, offset), per-device
   sizes, cross-device marking. CPU-only collapses to one device (numerics
   unchanged).
3. **Device executor.** Execute C1/C2/closure on the placement device (D2D /
   transfer-queue for cross-device). Separate large workstream.

## 7. Status (2026-09-14)

**Phase 1 - carry split: LANDED.** `layout_arena` splits the carry region:
cross-1 is double-buffered on layer parity (`2 x max boundary`), cross-N is
retained. The closure best-fit interval packing is extracted into
`pack_interval(nodes, start, end)` and reused for `carryN`; `layout_arena` now
runs AFTER the closure result layout so the packed closure size is used.

Measured at ubatch 512 (production build): deepseek carry 2630 MB -> ~67 MB,
olmoe 62 MB -> ~0.25 MB, gemma 160 MB -> ~11 MB. The closure block also drops
(olmoe ub1 278528 -> 131072 B). Verified: production `run_baseline` PASS;
olmoe / gemma / deepseek whole-layer OK.

**Compact pack: NOT a math bug - it breaks the `--prefill-from` export capture.**
Production `StreamMoE_dump` keeps it opt-in (`STREAM_MOE_TMP_COMPACT_PACK`,
default off = byte sum); the latest build `StreamMoE_latest` / `StreamMoE_dump_dbg`
defaults it ON (`AGENTS.md` 15), `=0` opts out.

Measured (gemma v2, 129-token prefill-from, 8 GB pool, vs `moe_129_8192_vk`):

| compact layout | exported embd cos (tok 0) | top-4 logits |
| :-- | --: | :-- |
| byte sum (default) | 0.98631 | identical |
| interval pack (`=1`) | 0.01568 | identical |

The interval-pack "FAIL" is an **export artifact, not an inference bug**:

- **Generation is unaffected.** `llama-cli` with pack ON and pack OFF produces
  the same output token-for-token (temp 0: identical "Paris." completion).
- **The model output is identical.** The exported top-4 logits + logsumexp are
  **bit-identical** pack vs sum (0/129 id mismatches, maxLogitDiff = 0,
  maxLseDiff = 0). The KV caches are identical too (base + swa, 0 diffs).
- **Only the exported `embd` / `hidden` differ.** `embd` is `t_embd` - the input
  token embeddings, deterministic from the token ids, so it *cannot* legitimately
  differ. Under whole-layer ownership it is a layer-0 node in the arena, and
  compact packing reuses its slot for a later L0 node: `embd` and `norm-0` share
  `off=0` / the same `data=` pointer (`[cpack]` / `[stage]`), while byte sum
  gives every node a unique slot. The export reads the slot **after** route B has
  already reused it, so it captures the overwriter's value.

So compact packing is correct for execution; the **prefill-export capture is
incompatible with arena slot reuse**. The earlier "packing is broken" reading is
superseded. (An even earlier "RelWithDebInfo sum == pack, 0 diffs" result was an
artifact - every dbg run had `STREAM_MOE_TMP_COMPACT_PACK` set, so its "sum" runs
were really pack runs.)

**Root cause of the stale read (found 2026-09-14).** With `cb_eval` installed the
scheduler stops computing a split as a whole and walks it **node by node**
(`ggml-backend.cpp:1747`), because the export callback returned `true` for every
node. Under whole-layer ownership each 1-node view makes route B run the whole
layer containing it (`exec_state` is per graph_compute call), so export mode was
~8x slower (measured: prompt 4.2 -> 0.5 t/s) and the capture fired after the layer
had already reused the slot.

**Fix - LANDED (2026-09-14).**
1. The export callback now returns `true` only for the tensors it actually reads
   (`embd` / `hidden` / `logits` / routed `MUL_MAT_ID` ids), so the scheduler only
   chunks at those points and route B runs a layer about once. Export overhead is
   back to ~1.4x (prompt 4.3 -> 3.3 t/s; the residual is the per-layer MoE cut).
2. Those tensors are declared to route B (`route_b_set_export_retained`) before
   the layer capture, and `layout_arena` keeps them out of the reuse pool (pulled
   into the retained `carryN` region). A post-split read then sees the computed
   value. The expert-history ids need this too (their last-layer ids were reused
   by the tail).

Gated on `STREAM_MOE_PREFILL_EXPORT` (the callback) and on
`STREAM_MOE_ROUTE_B && STREAM_MOE_PREFILL_EXPORT` (the retain API), so no code is
compiled where it cannot be used.

Verified (gemma v2, 129-token prefill-from, `StreamMoE_latest`, pack default ON):
exported `embd` / `hidden` / KV and the expert history are **IDENTICAL** pack vs
sum; top-4 logits stay bit-identical; pack matches the `moe_129_8192_vk` gate;
production `run_baseline` (`StreamMoE_dump`, pack off) still PASS.

**Repro.** Build `build.bat llamalibs StreamMoE_latest`, then the 129-token
prefill-from with no env takes the packed path and `=0` takes byte sum; both now
produce identical exports. For the `[cpack]` offset plan / `[stage]` node values
use `StreamMoE_dump_dbg` (+ `STREAM_MOE_TMP_STAGE_DUMP=1
STREAM_MOE_TMP_COMPACT_DEBUG=1`).

**Per-device plan: LANDED (2026-09-14, commit `fdd772e`).** `layout_arena` now
builds one `region_plan_t` per device, each with its own grow-only buffer:

- Two regions by liveness: `carry` (cross-layer pipeline) + `scratch`
  (within-layer: dense head/tail merged with the MoE closure).
- `carry1` per device (parity double-buffered), `carryN` per device (retained).
  `cross_device` marks a carry tensor whose consumer is on another device; the
  executor relays it (`ggml_backend_tensor_copy`, `M2_DEVICE_EXECUTOR.md` §7.8)
  before the consumer's first read. carry1 and carryN move together at every
  boundary (relay: the copy current at layer L is always on dev(L), so a
  consumer on any device reads the right one).
- The closure's `ex.out_off` / `result_bytes` are rewritten to index the merged
  scratch; `moe_chain_fullalloc_buffer` is now per-device (the expert pool's
  device) and returns the same pool the dense head/tail use. `route_b_in_arena`
  checks every device buffer.
- The MoE closure device is the expert pool device (`route_b_closure_device`,
  recorded at `route_b_setup`) - the closure runs where its experts live. The
  expert weights are not in a normal buffer, so `dev_of` could not resolve them
  from a weight operand.

Measured (gemma v2, 129-token prefill-from, ub 129): decode build carry1 88 KB /
scratch 8 MB; prefill build carry1 1.4 MB / scratch 128 MB. (The §1 ub-512
byte-sum table had 2.5 GB compact; the merged packed scratch is far smaller.)

Verified (CPU-only collapses to one host plan, so numerics are unchanged):
pack vs sum `IDENTICAL` (embd / hidden / KV + expert history); vk gate 121/129
(93.8%); production `run_baseline` (`StreamMoE_dump`) PASS.

**Cross-device carry relay: LANDED (2026-09-14, commit `0ef71df`).** A carry
tensor is the producer node's output, so it stays on the producer's device. When
its (max) consumer is on another device, `layout_arena` allocates a local copy -
a shell tensor reserved in the consumer layer's carry1/carryN boundary set (so
it shares the carry's packing and lifetime) - and rewires the consumers' `src`
to it. `exec_layer_burst` runs `ggml_backend_tensor_copy(src, dst)`
(synchronous; the `M2_DEVICE_EXECUTOR.md` §7.8 transport) at the front of the
consumer layer, before the head reads it. carry1 and carryN relay together at
every boundary.

Single-device / CPU-only: no cross-device carry, so no relays - verified
`IDENTICAL` (pack vs sum + expert history).

Still open (device executor, phase 3): running C1/C2 on the placement device
(`run_dense_subgraph` still uses the CPU backend). Caveat: the rewire mutates
the captured graph's `src` after the scheduler's tensor-backend pass - safe
while route B owns the whole layer, but must be re-checked if that changes.

## 8. Open questions

1. Is 2 buffers enough for `carry1`, or do some layers need more (e.g. a
   boundary set read at both head and tail plus a same-layer second boundary)?
2. `carry1` boundary set: pack per boundary, or one shared set sized to the max
   across boundaries (simpler, slightly larger)?
3. C2 on a device other than C1: the last layer's output -> C2 needs a
   cross-device move; keep it a plan-marked transfer, or require C2 == C1's
   device for zero-copy (design §4.10 C2 takeover)?
4. Do we keep a single grow-only buffer per device, or one buffer per region?
5. When C1 is on a device, `run_dense_subgraph` must call that device's backend;
   `moe_exec_mul_mat_id` already receives `cpu_backend` - how is the per-layer
   device backend threaded through?

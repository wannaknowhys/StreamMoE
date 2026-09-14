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

## 3. Per-device plan

`layout_arena` becomes a per-device planner. One `region_plan_t` per device:

```
struct region_plan_t {
    size_t carry1_bytes;   // 2 * max boundary pack
    size_t carryN_bytes;   // retained
    size_t compact_bytes;  // max over layers of pack(layer dense nodes)
    size_t closure_bytes;  // max over layers of pack(layer MoE closure)
    std::unordered_map<const ggml_tensor*, size_t> off;   // node -> offset
    ggml_backend_buffer_t buf;                            // device arena
};
```

**Node -> device:**

- **C1 dense** (attention / norms / dense MLP / gating) -> `C1:<dev>`
  (`--dense-placement C1`); this device also owns the layer's `carry`.
- **C2 output head** (`result_norm` / `result_output`, i.e. the logits) ->
  `C2:<dev>` (`--dense-placement C2`).
- **MoE closure** -> the device of that layer's expert pool
  (`--moe-expert-pools <dev>:<MB>`).
- Default (no placement): everything on CPU/RAM.

**Per-device size** is computed from what is present on the device: has C1
(-> carry + compact), has an expert pool (-> closure), has C2 (-> the C2
activation). The resulting plan guides both the C1/C2 arena and the expert
closure layout.

**Cross-device tensors** (a node whose consumers live on another device: `cur`
into a remote pool, `acc` back to the owner, cross-device `carry`) are marked
in the plan so the executor can move them with D2D / the transfer queue
(`ROUTE_B_LAYER_OWNERSHIP.md` §3.5) - never host staging.

## 4. Reuse the existing packing

Extract the best-fit-decreasing interval packing currently in
`moe_chain_verify_graph` (the closure result layout, `route_b_chain.cpp`) into
`pack(nodes, last_use) -> { offsets, size }`. Use it for `compact`, `closure`,
`carryN`, and each `carry1` boundary set. No second allocator.

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

Next: snapshot `embd` / `hidden` at their real compute time (or exclude exported
tensors from arena reuse) so the prefill regression is valid with pack on.

**Repro.** Build `build.bat llamalibs StreamMoE_latest`, then the 129-token
prefill-from with no env takes the packed path and `=0` takes byte sum. The
exported `embd` differs, but the top-4 logits (the real output) do not - check
both with `baseline_regression/tools/verify_prefill.js` and a top-4 comparator.
For the `[cpack]` offset plan / `[stage]` node values use `StreamMoE_dump_dbg`
(+ `STREAM_MOE_TMP_STAGE_DUMP=1 STREAM_MOE_TMP_COMPACT_DEBUG=1`).

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

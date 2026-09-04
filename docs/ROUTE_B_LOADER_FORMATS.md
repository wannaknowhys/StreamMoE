# Route-B Loader: GGUF Input Formats, Gaps, and Async Loading Design

[English](ROUTE_B_LOADER_FORMATS.md) | [简体中文](ROUTE_B_LOADER_FORMATS.zh-CN.md)

> Source of truth for this doc: `tools/stream_moe_layout.js` (converter layout
> logic, write direction) + `src/loader/moe_loader.cpp` / `src/io/staging_reader.h`
> (loader, read direction). The converter evolved after the loader; this doc
> records the divergence and the target async loading design.

> **2026-09 revision (supersedes v1-as-superset and whole-block-DIO below)**:
> ggml-vulkan hardcodes the per-expert stride to the single-tensor compact size
> (`ne0*ne1`), so route B is moving to **struct-of-array pools (one column per
> tensor)** and v2 blocks with **each branch tensor slice 4K-aligned inside the
> block** (see `STREAMMOE_GGUF_FORMAT.md` §2.6). Consequences for the loader:
> v1 sections-v1 is dead (GGUF tensor offsets must be compact/monotonic - the
> writeV1 per-expert reflow produced GGUF that llama refuses to load). For v2,
> loading becomes **one DIO per (expert, tensor-slice)** instead of one DIO per
> whole block: a slice whose perExpert is a 4K multiple loads straight into its
> tensor column (aligned source + aligned slot); otherwise DIO reads a 4K window
> into staging then moves into the column. The column layout is decided in
> `src/loader` + `src/backend/scheduler` (SoA), independent of the file format.

## 1. Input formats

| format | `stream_moe.layout` | `incomplete` | expert layout | alignment | read plan |
|---|---|---|---|---|---|
| **original GGUF** (single or `-00001-of-N.gguf`) | absent / `"original"` | - | per-tensor contiguous: expert slice = `tensor.offset + e*perExpert`; 3 sub-tensors (gate/up/down or gate_up/down) per expert | GGUF default (32B / quant block) | 3 sector-aligned reads into staging buffer + memcpy to slot (needs staging) |
| **v1 sections-v1** | `"sections-v1"` | - | same per-tensor slices, but tensors 4K-aligned by the converter | 4096 | **intended**: 3 async DIO straight into slot (no staging). **Not implemented** - falls through to the original staging path |
| **v2 expert-blocks-v2** | `"expert-blocks-v2"` | - | per-(layer,expert) block; branches (gate_up/gate/up/down) concatenated at `branchOff` inside the block; block size = alignUp(sum(branch perExpert), 4096) | 4096 blocks | 1 async DIO whole-block straight into slot (block layout == slot layout) |
| **v2 chunk** | `"expert-blocks-v2"` | `1` | block strips scattered across N strip files (`chunk_slices` per file); one expert block spans up to N file segments | 4096 strips | **Not implemented** - loader hardcodes single file (`shard_idx = 0`) |

## 2. Layout KV semantics (`stream_moe.*`)

Written by `writeV2` / `writeV2chunk` in `stream_moe_layout.js`, read by
`build_v2_experts` in `moe_loader.cpp`:

| KV | meaning |
|---|---|
| `stream_moe.layout` | `"original"` / `"sections-v1"` / `"expert-blocks-v2"` |
| `stream_moe.incomplete` | `1` = v2 chunk (strip files); `0`/absent = single file |
| `stream_moe.dense_section` | `[0, denseEnd]` - dense tensor area (before blocks) |
| `stream_moe.expert_sections` | `[off, size, nsub]` per block (nLayer*nExpert blocks) |
| `stream_moe.expert_branch_names` | flattened per-layer full branch tensor names |
| `stream_moe.expert_branch_sizes` | per-branch `perExpert` bytes (flattened, same order as names) |
| `stream_moe.expert_branch_counts` | per-layer branch count (non-uniform MoE layers) |
| `stream_moe.chunk_no` / `chunk_total` | strip index / total for v2 chunk |
| `stream_moe.chunk_slices` | `[denseBlocks, blockSlices...]` per file - 4K-aligned strips this file holds |

## 3. Current gaps (loader vs converter)

1. **v1 4K alignment unused.** `moe_loader::parse_gguf_topology` only sets
   `topo.layout = V1_SECTIONS` for `sections-v1` and then falls through to the
   ORIGINAL per-tensor slice path (staging + copy). The 4K-aligned v1 layout
   should allow 3 direct async DIO reads into the slot (no staging, no copy).
   Needs confirmation: is every per-expert slice actually 4K-aligned in v1
   (perExpert may not divide to 4K even if the tensor start is 4K-aligned)?

2. **v2 chunk unsupported.** `build_v2_experts` hardcodes `shard_idx = 0`,
   reads one whole block from the main file, never reads `chunk_slices` /
   `incomplete`. For v2 chunk each expert block is a strip scattered across N
   files (like `rangeToSegs` in the converter) - the loader must map one block
   to N file segments and issue per-file DIO.

3. **Heterogeneous experts.** Per-expert size groups exist (`topo.groups`,
   `MULTI_SUBPOOL.md`); the read plans must be built per-group (independent
   staging size / slot size / DIO count per group), which holds for v2 but must
   be preserved in the v1/original rework too.

## 4. Target async loading design (concept - mirrors convertd)

Unified planner + uniform async DIO:

```
input path(s)
  -> format detect (layout KV + incomplete flag, both in file header)
  -> per-format planner -> uniform plan:
       expert e -> [ { file, off, len, slot_off } ... ]   (1..N segments)
  -> async DIO engine (IOCP / io_uring / io_submit fallback), N in-flight
  -> completion -> slot placement
```

Per-format DIO profile:
- **original**: 3 reads/expert, each into an 8K-padded staging buffer (front+back
  padding, size+2*4096), then memcpy to slot. Aligned to 4K for DIO.
- **v1**: 3 async DIO straight into slot (no staging) if slices are 4K-aligned.
- **v2**: 1 async DIO whole-block straight into slot.
- **v2 chunk**: per-file strip reads (N segments/expert), straight into slot.

All reads async + continuous (largest contiguous aligned intervals per file),
concurrent in-flight. **Time field in the expert async header**: raw TSC
(`uint64_t req_tsc` / `dio_tsc` / `done_tsc`) captured at submit/DIO-complete/
ready, converted to ns at profile time (TSC frequency calibrated once at startup
via chrono). No per-expert printf - the field is inert data for future dynamic
profile / adaptive prefetch.

## 5. Open questions

- **Q1 original "中转" semantics**: the staging path is 3 sector-aligned reads +
  copy; confirm original slices are not 4K-aligned and cannot be read straight
  into the slot even after padding (i.e. staging is mandatory, not optional).
- **Q2 v1 "3 reads" breakdown**: gate_up / down / scale as three separate 4K
  aligned regions? (scale is currently treated as dense in the converter - see
  `buildModel` `isScale`.) Confirm the exact v1 section layout.
- **Q3 layout single source**: converter layout lives in JS
  (`stream_moe_layout.js`); a C++ loader cannot reuse it directly. Options:
  (a) sink layout computation into a C++ shared module both convertd and the
  loader link; (b) full C++ convertd (drop JS); (c) keep JS as authority and
  re-implement layout in the loader (divergence risk). Q3 determines the size
  of this work.

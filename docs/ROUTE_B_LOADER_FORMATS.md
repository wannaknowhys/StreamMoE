# Route-B Loader: GGUF Input Formats, Gaps, and Async Loading Design

[English](ROUTE_B_LOADER_FORMATS.md) | [简体中文](ROUTE_B_LOADER_FORMATS.zh-CN.md)

> Source of truth for this doc: `src/loader/model_builder.cpp` / `src/loader/model.h`
> (shared read model) + `src/convert/writer.cpp` (write direction). The converter
> is now pure C++ and shares `model_t` with the loader.

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
| :--- | :--- | :--- | :--- | :--- | :--- |
| **original GGUF** (single or `-00001-of-N.gguf`) | absent / `"original"` | - | per-tensor contiguous: expert slice = `tensor.offset + e*perExpert`; 3 sub-tensors (gate/up/down or gate_up/down) per expert | GGUF default (32B / quant block) | 3 sector-aligned reads into staging buffer + memcpy to slot (needs staging) |
| **v2 expert-blocks-v2** | `"expert-blocks-v2"` | - | per-(layer,expert) block; branches (gate_up/gate/up/down) concatenated at `branchOff` inside the block; block size = alignUp(sum(branch perExpert), 4096) | 4096 blocks | 1 async DIO whole-block straight into slot (block layout == slot layout) |
| **v2 chunk** | `"expert-blocks-v2"` | `1` | block strips scattered across N strip files (`chunk_slices` per file); one expert block spans up to N file segments | 4096 strips | **Not implemented** - loader hardcodes single file (`shard_idx = 0`) |
| **v3 category-sections** | `"v3"` | - | four sections: C2 global-dense / C1 layer-dense / C4 expert-meta / C3 expert blocks (block layout == v2). Split by whether the tensor is consumed by the MoE closure (see `STREAMMOE_GGUF_FORMAT.md` §3) | 4096 | **Not implemented** - each section maps to a distinct residency policy: C2 pinned resident, C1 streamed per layer, C4 replicated per device, C3 pooled / cross-device |

## 2. Layout KV semantics (`stream_moe.*`)

Written by `src/convert/writer.cpp`, read by `parse_model` (`src/loader/model_builder.cpp`):

| KV | meaning |
| :--- | :--- |
| `stream_moe.layout` | `"original"` / `"expert-blocks-v2"` / `"v3"` |
| `stream_moe.incomplete` | `1` = v2 chunk (strip files); `0`/absent = single file |
| `stream_moe.dense_section` | `[0, denseEnd]` - dense tensor area (before blocks) |
| `stream_moe.expert_sections` | `[off, size, nsub]` per block (nLayer*nExpert blocks) |
| `stream_moe.expert_branch_names` | flattened per-layer full branch tensor names |
| `stream_moe.expert_branch_sizes` | per-branch `perExpert` bytes (flattened, same order as names) |
| `stream_moe.expert_branch_counts` | per-layer branch count (non-uniform MoE layers) |
| `stream_moe.chunk_no` / `chunk_total` | strip index / total for v2 chunk |
| `stream_moe.chunk_slices` | `[denseBlocks, blockSlices...]` per file - 4K-aligned strips this file holds |
| `stream_moe.dense_global_section` (v3) | `[off, size]` - C2 global-dense area |
| `stream_moe.dense_layer_sections` (v3) | `[layer, off, size, ...]` - C1 per-layer dense areas |
| `stream_moe.expert_meta_sections` (v3) | `[layer, off, size, ...]` - C4 per-layer expert-meta tables (may be empty) |

## 3. Current gaps (loader vs converter)

1. **v2/v3 chunk read.** `parse_model` maps chunk strips to multi-segment `src`
   lists (v3 unit = [global] + [C1 layer] + [C4 layer] + [block]); the
   scheduler/DIO path must consume the multi-segment plan (`moe_loader.cpp`
   historically hardcoded `shard_idx = 0`).

2. **Heterogeneous experts.** Per-expert size groups exist (`topo.groups`,
   `MULTI_SUBPOOL.md`); read plans are built per-group.

3. **v3 residency policy not wired.** `parse_model` understands v3; the engine
   still loads all dense tensors the same way. The four categories imply
   distinct policies (C2 pinned / C1 layer-streamed / C4 replicated / C3 pool).

## 4. Target async loading design (concept - mirrors the C++ writer)

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
- **v2**: 1 async DIO whole-block straight into slot.
- **v2/v3 chunk**: per-file strip reads (N segments/expert), straight into slot.

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
- **Q2 layout single source (resolved)**: the converter is now pure C++ and
  shares `src/loader/model.h::model_t` + `layout_math.h` with the loader; no JS
  double-implementation remains.

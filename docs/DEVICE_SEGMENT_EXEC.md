[English](DEVICE_SEGMENT_EXEC.md) | [简体中文](DEVICE_SEGMENT_EXEC.zh-CN.md)

# Device-Segmented Execution (Path B)

## Problem

The xfer-stage mechanism (`run_layer_xfers`) assumes same-layer producers are
ready before `LAYER_FRONT` xfers run. When C2 ≠ C1, non-layer nodes (embd,
output_norm, output) are assigned to adjacent layers by fallback propagation and
placed on the C2 device. Xfers copy their output **before** they are computed,
reading uninitialised device memory → garbled inference.

## Design

Replace `run_layer_xfers` + `run_dense_by_device` with a single
**topo-segmented executor** that interleaves device segments and inter-device
copies in topological order. The three-phase structure is preserved:

```
exec_layer_burst(layer):
  1. topo_segment_execute(dense_head)
  2. IDs readback (hard sync) + expert pin + bucket executor (unchanged)
  3. topo_segment_execute(dense_tail)
```

### topo_segment_execute(nodes)

Input: a list of tensors in topological order, each with a known physical device
(`route_b_node_device`).

Algorithm:

1. Walk the list sequentially. Track the "current device".
2. Accumulate contiguous same-device nodes into the current **segment**.
3. When the device changes (or the list ends):
   a. **Flush**: build a mini-graph from the current segment, call
   `ggml_backend_graph_compute(real_backend, mini_graph)`.
   b. **Copy**: for each node in the _next_ segment, if any `src` tensor
   was produced on a _previous_ device and has a relay shell on the current
   device, call `ggml_backend_tensor_copy(src, shell)`.
   c. Start a new segment on the new device.

This guarantees copies happen **after** the producer segment and **before** the
consumer segment — ordering is correct by construction.

### Changes to layout_arena

The relay shells (`g_relays`) are still created: they are arena-allocated tensors
on the consumer device that receive the cross-device copy. What changes:

- **Remove stage tagging** (`ROUTE_B_XFER_LAYER_FRONT`, `ROUTE_B_XFER_CLOSURE`,
  `ROUTE_B_XFER_TAIL`). Relay execution is no longer stage-driven; it is
  segment-boundary-driven.
- The shell allocation and src→shell rewiring in `layout_arena` stay the same.

### Non-layer nodes

`embd`, `norm` (output_norm), `result_norm`, `result_output` are assigned to
layers by fallback propagation in `collect_layer_nodes`. They join the layer's
`head` or `tail` list via `build_layer_plans`. The topo-segmented executor
handles them naturally: if `embd` is on Vulkan0 (C2) and the rest of the head is
on CPU (C1), the executor creates a Vulkan0 segment for `embd`, copies the
result to CPU, then executes the CPU segment.

### IDs readback

The routing IDs readback is a hard synchronisation point between head and
closure. The host must inspect the IDs to decide which experts to pin. The
closure (bucket executor) runs its own specialised path — it stages leaves, runs
the per-bucket chain, and writes `moe_out`. This is unchanged.

### What stays the same

- Arena layout: carry1/carryN/scratch BFD packing, double-buffered pipeline
- Per-device plan buffers (host plan + optional device plans)
- Relay shell creation and src rewiring in `layout_arena`
- Bucket executor for the MoE closure
- `run_dense_subgraph`: still the leaf function that executes a mini-graph on a
  real backend

### What is deleted

- `run_layer_xfers` function
- Stage constants (`ROUTE_B_XFER_LAYER_FRONT`, `ROUTE_B_XFER_CLOSURE`,
  `ROUTE_B_XFER_TAIL`)
- `run_dense_by_device` function (replaced by `topo_segment_execute`)
- All three `run_layer_xfers` call sites in `exec_layer_burst`

### What is new

- `topo_segment_execute(ctx, cpu_backend, nodes, layer, stage_name)`: the
  segment-boundary executor described above.

## Invariants

1. **Topological order preserved**: the node list from `build_layer_plans` is
   already in graph topological order. The segment executor respects this.
2. **Copy before consume**: a cross-device copy fires only between flushing the
   producer segment and starting the consumer segment.
3. **No regression for single-device**: when C1 == C2 (or all-RAM), there are no
   device transitions → one segment, zero copies. Identical to current path.
4. **GPU end-state compatible**: each device gets one contiguous graph segment,
   copies are explicit data-movement operations — this maps directly to per-GPU
   command queues with DMA transfers on the real GPU path.

# L2 Whole-Layer Execution Experiment Archive (2026-09)

Archive for the Route B Whole-Layer L2 dense delegation and execution experiment
(docs/ROUTE_B_LAYER_OWNERSHIP.md L2). The experiment explored taking over whole-layer
compute nodes (dense head + MoE burst + dense tail) under our backend to eliminate
per-layer backend boundary synchronization and redundant seam copies.

Status:
L2 still exhibits numerical divergence on the output-token reduction path
(hence gated behind STREAM_MOE_TEMP in commit 6cda201).

Restore (if ever needed):

    git apply debug_patch/l2-whole-layer/l2_whole_layer.patch
    # or
    git am debug_patch/l2-whole-layer/commits/0002-route_b-gate-whole-layer-L2-behind-STREAM_MOE_TEMP-f.patch

Files:

- `l2_whole_layer.patch`: Net diff of commit 6cda201 (L2 implementation:
  run_dense_nodes, dense_head/tail split, moe_exec_mul_mat_id multi-layer traversal,
  moe_chain_assign_backend whole-layer assignment, supports_buft, extra scratch allowance,
  and CONT alias fix).
- `0001-route_b-gate-whole-layer-L2-behind-STREAM_MOE_TEMP-f.patch`: format-patch of commit 6cda201.
- `commits/`: Individual patches including:
  - `0001-route_b-whole-layer-node-capture-L1-layer-ownership-.patch` (c6d0822, L1 capture + docs)
  - `0002-route_b-gate-whole-layer-L2-behind-STREAM_MOE_TEMP-f.patch` (6cda201, L2 gate + execution)

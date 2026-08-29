[English](REPACK_DIVERGENCE_DEBUG.md) | [简体中文](REPACK_DIVERGENCE_DEBUG.zh-CN.md)

# Route B vs Upstream: Bit-Level Debug of the gate_up Divergence (cos 0.508) and the repack Root Cause

> Status: investigated, root cause confirmed, verified bit-identical, cleaned up (2026-08-29).
> Related code: `third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp` (repack kernels / extra buffer type), `src/backend/minigraph_exec.cpp` (route B delegate mini-graph).

## Background

- Goal: verify that route B (expert pool, bounded physical memory) is numerically consistent with upstream llama.cpp for gemma-4-26B-A4B (Q4_K_M).
- Symptom: with 30 layers of MoE, the LM head cosine drops to **0.508** while attention stays bit-identical.

## Investigation Chain

1. **First bit divergence**: `ffn_moe_geglu-0` (cur) at rb step #24.
2. **cur (gate/up slice) differs by 1-13 ulp** from upstream. Every layer re-norms (4-5 rms_norm/layer), so a 1e-7 difference on a large value (up branch) gets amplified layer by layer -> LM head cos 0.508.
3. **Not caused by stride**: changing the expert stride (3717120 / 1486848 / 7929856) does not change the output.
4. **Not caused by delegation args**: delegate -> kernel params match upstream 52/52 (nb, data heads, ids).
5. **down output matches upstream**; the L1 down difference is inherited from the previous layer.

## Key Lesson: Never use fnv for numeric comparison

- fnv is a byte-level hash: a single 1-ulp difference flips almost every bit -> "completely different" even though the values are nearly identical.
- Use **cos / maxDiff / ulp-bit distribution** (per-value popcount of the XOR of two float bit patterns).
- fnv is only valid when you specifically want "are these byte sequences bit-identical".

## Root Cause: repack vs plain kernel path

- Upstream repacks gemma's `ffn_gate_up_exps.weight` (Q4_K_M, 3D, ne[0]=1408 / ne[1] % 16 == 0) through `ggml::cpu::repack`'s extra buffer type: `MUL_MAT_ID` is routed to the repack `forward_mul_mat_id` with `block_q4_Kx8` layout.
- `down_exps` does NOT match the repack support type, so it runs the plain `ggml_compute_forward_mul_mat_id`.
- Route B's mini-graph (`w3d` = pool slot, plain CPU buffer) never matches the repack `supports_op`, so gate_up/down both run the plain kernel.
- => gate_up executes a different kernel path on each side -> 1-16 ulp differences, amplified to cos 0.508.
- deepseek (Q8_K_XL) does not match any repack type -> both sides run the same kernel -> cos 0.99999999.

## Bit-Level Verification

L0 gate_up output, token 0, expert column (1408 floats):

- **rb plain vs up repack**: bit-identical 20.8%, differ 79.2%. Bit-diff distribution: 1bit=7110, 2-4bit=9357, 5-16bit=1378, 17+bit=1, max=20bit. Values display identically (-0.05954 etc.) -> 1e-6-level float jitter, NOT garbage.
- **rb repack path vs up**: **bit-identical 1408/1408 (100.0%), 0 differences**.

Conclusion: identical input (bit) + identical weight bytes + identical kernel path => bit-identical output. The entire divergence is the repack-vs-plain path difference.

## How the Verification Was Implemented

### Idea 1: simulate the dispatch
Build a one-expert `MUL_MAT_ID` whose src0 is a repacked copy of the slot weight and give it a repack `buffer->buft` + `extra`, so `ggml_backend_graph_compute` routes it through the repack extra buffer type.

### Idea 2 (used): drive the kernel directly
Expose a helper in `repack.cpp` that:
1. allocates a repack buffer (`ggml_backend_buft_alloc_buffer`), sets `t->buffer/data/extra`, repacks via `traits->repack(...)` (must match the kernel, see below);
2. builds `ggml_tensor` structs on the stack (src0/src1/ids/op with explicit ne/nb/data), a `ggml_compute_params` (wdata/wsize, threadpool, ith=0, nth=1), then calls `((tensor_traits*)src0->extra)->compute_forward(&params, &op)`.

### Pitfalls hit (all resolved)
- x86 AVX2 Q4_K repack kernel is **`q4_K_8x8_q8_K` (8x8 layout)**; `q4_K_16x1_q8_K` is RISC-V only. Using the wrong repack layout produces huge/NaN outputs.
- Use the kernel's own `traits->repack()` instead of hard-coding a repack function.
- `ggml_compute_params` has no `type` field.
- `params->threadpool` must be non-null (`ggml_barrier` dereferences it); create one with `ggml_threadpool_params_default(1)` + `ggml_threadpool_new`.
- `wsize = GGML_PAD(nbw3, 8) + n_as*(ne12+1)*8`; add headroom to be safe.
- With `no_alloc` ggml contexts every tensor `data` is NULL - assign manually (cur slice, ids, dst).
- `block_q4_Kx16` is 2304 bytes (d[16]+dmin[16]+scales[192]+qs[2048]); the 16-row interleave writes exactly `ggml_nbytes` of the original Q4_K tensor.

## Debug Methodology (engineering notes)

- Reading the `MUL_MAT_ID` output at the accumulation site is useless (buffers are reused); log at the execution site (delegate / kernel entry / repack).
- Run `llama-cli` directly with cmd redirection: `cmd /c "... > file 2>&1"`. PS `*>` stream redirection buffers and hangs; `Start-Process` + `Sleep` + `Stop-Process` is racy.
- Check `$LASTEXITCODE`: `-1073741819` (0xC0000005) = access violation, `-1073740791` (0xC0000409) = assertion/stack-buffer-overrun.
- Stderr to a file is not line-buffered; use `fflush(stderr)` after diagnostic prints when the process may crash.

## repack internals (for future reference)

- `supports_op` for `MUL_MAT_ID`: `src[0]->buffer && buft == ggml_backend_cpu_repack_buffer_type() && ggml_n_dims(src[0]) == 3 && ggml_repack_get_optimal_repack_type(src[0]) && src[1]->type == GGML_TYPE_F32`.
- `get_tensor_traits` returns `src[0]->extra`; the buffer's `init_tensor` sets `tensor->extra = ggml_repack_get_optimal_repack_type(tensor)`.
- The repack data buffer holds the re-interleaved weights; tensor `ne`/`nb` stay at the original values (assert `nb00 == ggml_type_size(src0->type)` holds for Q4_K = 144).

## Cleanup

- No formal changes were pending: submodule HEAD `5ab785cf8` (export) and main repo `36e2b99`/`113c60f` already contain all production code.
- `git checkout` reset the debug-only edits in 6 files (minigraph_exec.cpp + submodule ggml-cpu.c / repack.cpp / ggml.c / llama-context.cpp / llama-graph.cpp).
- Deleted 102 temp files + `temp/export_ds`, `temp/sequences`, `temp/gemma_export`. Kept `temp/sm_env.example.bat`, `temp/patches_split/`, `temp/patch_backup_*/`, `temp/llama_verify/`.

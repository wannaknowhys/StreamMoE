#pragma once

// StreamMoE bucket scatter planning (docs/SCATTER_PLAN.md).
// Pure / deterministic / model-agnostic - no llama.cpp or ggml dependency.
//
// After the compact chain folds per-token partial sums, one column per active
// token (a order) must be accumulated back into the full-width acc_d at the
// original token id t[a]. ggml_acc can only express "len CONTIGUOUS src
// columns -> dst columns in one arithmetic progression (dst, dst+delta, ...)",
// so scatter_plan decides a tight order (order[]) that permutes the per-token
// columns, then decomposes the token-to-dst map into few arithmetic runs
// (segs[]). Each seg costs exactly one ggml_acc_inplace.
//
// Storage is caller-owned (grow-only, reused across layers); the plan points
// into it.

#include <cstdint>
#include <vector>

namespace stream_moe {

// One arithmetic run of acc-dst columns fed by a contiguous src run.
//   acc_d[.., dst + i*delta] += per_token[.., src + i]   for i in [0, len)
// dst/delta are dst-column indices into acc_d[d_out, n_t]; the executor turns
// them into ggml_acc params: nb1 = delta*d_out*esize, offset = dst*d_out*esize.
struct scatter_seg_t {
    uint32_t src   = 0;    // tight-order src start column (== run index offset)
    uint32_t dst   = 0;    // acc-dst column of src[0]
    uint32_t len   = 0;    // columns in this run (>= 1)
    uint32_t delta = 0;    // dst column step between consecutive src columns
};

struct scatter_plan_t {
    // Tight order: new per-token column i (0..n_active-1) was originally the
    // active-token entry `order[i]` (index into the input `t` sequence). The
    // cur-copy layer gathers cur columns in this order.
    const uint32_t *      order = nullptr;
    const scatter_seg_t * segs  = nullptr;
    uint32_t              n_order = 0;
    uint32_t              n_segs  = 0;
    uint32_t              n_active = 0;
    uint32_t              n_t      = 0;
};

// Build the plan for one bucket. `t[0..n_active)` = original token id of each
// active column (a order, i.e. t[a] = scatter[a*w_b].t). n_t = layer token
// count (bounds: dst+delta*(len-1) < n_t). out_order / out_segs are cleared and
// filled; they are the caller's grow-only buffers.
//
// Deterministic greedy (docs/SCATTER_PLAN.md §4): repeatedly extract the
// longest remaining arithmetic run of token ids. Tie-break: largest run len,
// then smallest delta, then smallest start value. Runs are emitted in
// extraction order and each run's values ascending.
//
// Invalid input (null t with n_active>0, any t[i] >= n_t, duplicate token id)
// returns a plan with n_order==0 and n_active left at the input value - callers
// distinguish "no work" (n_active==0) from "rejected" (n_active>0, n_order==0).
// The rectangle peel never produces duplicates (each round owns one disjoint
// slot slice per active token), so a duplicate is an input error.
scatter_plan_t build_scatter_plan(const uint32_t* t, uint32_t n_active, uint32_t n_t,
                                  std::vector<uint32_t>& out_order,
                                  std::vector<scatter_seg_t>& out_segs);

} // namespace stream_moe

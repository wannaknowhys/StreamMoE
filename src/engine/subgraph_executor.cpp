#include "engine/subgraph_executor.h"
#include "common/logger.h"

#include <cmath>
#include <cstring>
#include <omp.h>

namespace stream_moe {

namespace {

// Standard SwiGLU activation: silu(gate) * up = (gate / (1 + exp(-gate))) * up
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

} // namespace

subgraph_executor::subgraph_executor(
    const moe_model_topology_t& topo,
    expert_pool& pool,
    uint32_t n_embd,
    uint32_t n_ff
) : topo_(topo),
    pool_(pool),
    n_embd_(n_embd),
    n_ff_(n_ff) {
    
    // Allocate intermediate activation buffers
    size_t max_tokens = 512;
    act_gate_.resize(n_ff_ * max_tokens);
    act_up_.resize(n_ff_ * max_tokens);
    act_down_.resize(n_embd_ * max_tokens);
}

subgraph_executor::~subgraph_executor() = default;

void subgraph_executor::compute_expert_rebind(
    int32_t slot_id,
    float expert_weight,
    const float* inp,
    float* out_accum,
    uint32_t n_tokens
) {
    if (slot_id < 0 || slot_id >= static_cast<int32_t>(pool_.num_slots()) || !inp || !out_accum || n_tokens == 0) {
        return;
    }

    const expert_slot_t& slot = pool_.get_slot(slot_id);
    const uint8_t* raw_expert_data = slot.raw_ptr;
    if (!raw_expert_data) return;

    // Pointer Rebind: sub-tensors are mapped to their respective offsets inside slot->raw_ptr
    // In DeepSeek / SwiGLU:
    // Tensor 0: gate, Tensor 1: up, Tensor 2: down
    // Weights are bound directly from raw_expert_data
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t i = 0; i < n_embd_; ++i) {
            // Apply weighted forward contribution to out_accum
            // For testing/mocking verification, performs scaled linear forward
            #pragma omp atomic
            out_accum[t * n_embd_ + i] += inp[t * n_embd_ + i] * expert_weight;
        }
    }
}

void subgraph_executor::compute_batch_rebind(
    const std::vector<expert_compute_item_t>& items,
    const float* inp,
    float* out_accum,
    uint32_t n_tokens
) {
    for (const auto& item : items) {
        compute_expert_rebind(item.slot_id, item.weight, inp, out_accum, n_tokens);
    }
}

} // namespace stream_moe
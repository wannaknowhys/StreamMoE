#pragma once

#include "common/types.h"
#include "loader/moe_loader.h"
#include "pool/expert_pool.h"

#include <vector>
#include <memory>

namespace stream_moe {

struct expert_compute_item_t {
    int32_t slot_id;
    float   weight;
};

class subgraph_executor {
public:
    subgraph_executor(
        const moe_model_topology_t& topo,
        expert_pool& pool,
        uint32_t n_embd,
        uint32_t n_ff
    );
    ~subgraph_executor();

    // Disable copy/move
    subgraph_executor(const subgraph_executor&) = delete;
    subgraph_executor& operator=(const subgraph_executor&) = delete;

    // Execute an expert GEMM on CPU/Vulkan via Pointer Rebind and accumulate into out_accum
    // inp: [n_embd, n_tokens] (float/fp16)
    // out_accum: [n_embd, n_tokens] (accumulated output)
    void compute_expert_rebind(
        int32_t slot_id,
        float expert_weight,
        const float* inp,
        float* out_accum,
        uint32_t n_tokens
    );

    // Batch compute a list of ready expert slots
    void compute_batch_rebind(
        const std::vector<expert_compute_item_t>& items,
        const float* inp,
        float* out_accum,
        uint32_t n_tokens
    );

private:
    [[maybe_unused]] const moe_model_topology_t& topo_;
    expert_pool&                pool_;
    uint32_t                    n_embd_ = 0;
    uint32_t                    n_ff_ = 0;

    // Intermediate activations buffers
    std::vector<float>          act_gate_;
    std::vector<float>          act_up_;
    std::vector<float>          act_down_;
};

} // namespace stream_moe
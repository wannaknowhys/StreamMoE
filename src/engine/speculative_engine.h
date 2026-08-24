#pragma once

#include "common/types.h"
#include "engine/state_machine.h"
#include "scheduler/moe_scheduler.h"
#include "loader/moe_loader.h"

#include <vector>
#include <string>
#include <functional>

namespace stream_moe {

struct spec_draft_output_t {
    std::vector<int32_t> draft_tokens;
    std::vector<float>   draft_logits;
};

struct spec_verify_result_t {
    uint32_t             n_accepted = 0;
    int32_t              next_token = 0;
    double               acceptance_rate = 0.0;
    std::vector<int32_t> accepted_tokens;
};

class speculative_engine {
public:
    speculative_engine(
        const moe_model_topology_t& target_topo,
        state_machine& sm,
        moe_scheduler& scheduler
    );
    ~speculative_engine() = default;

    // Set draft model GGUF path
    bool load_draft_model(const std::string& draft_model_path);

    // Multi-token expert router pre-exposure:
    // Given K draft candidate tokens, computes the union of experts required at layer L
    std::vector<uint32_t> aggregate_draft_experts(
        uint32_t layer_idx,
        const std::vector<int32_t>& draft_tokens,
        uint32_t top_k_per_token
    );

    // Execute verification step:
    // Verifies K draft tokens against target model probabilities, accepts M prefix tokens, updates state machine
    spec_verify_result_t verify_and_accept(
        const std::vector<int32_t>& draft_tokens,
        const std::vector<int32_t>& target_top_tokens
    );

    // Get current active draft step K
    uint32_t get_active_k() const;

private:
    const moe_model_topology_t& target_topo_;
    state_machine&              sm_;
    moe_scheduler&              scheduler_;
    std::string                 draft_model_path_;
    bool                        draft_model_loaded_ = false;

    // Moving average of acceptance rates
    double                      avg_acc_rate_ = 0.75;
};

} // namespace stream_moe
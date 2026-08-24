#include "engine/speculative_engine.h"
#include "common/logger.h"

#include <filesystem>
#include <unordered_set>
#include <algorithm>

namespace stream_moe {

speculative_engine::speculative_engine(
    const moe_model_topology_t& target_topo,
    state_machine& sm,
    moe_scheduler& scheduler
) : target_topo_(target_topo),
    sm_(sm),
    scheduler_(scheduler) {}

bool speculative_engine::load_draft_model(const std::string& draft_model_path) {
    draft_model_path_ = draft_model_path;
    if (!draft_model_path_.empty() && std::filesystem::exists(draft_model_path_)) {
        draft_model_loaded_ = true;
        LOG_INFO("Draft Model registered for speculative decoding: " << draft_model_path_);
        return true;
    }
    LOG_WARN("Draft model file not found at " << draft_model_path_ << " (speculative decoding disabled)");
    draft_model_loaded_ = false;
    return false;
}

std::vector<uint32_t> speculative_engine::aggregate_draft_experts(
    uint32_t layer_idx,
    const std::vector<int32_t>& draft_tokens,
    uint32_t top_k_per_token
) {
    std::unordered_set<uint32_t> expert_set;
    if (target_topo_.n_expert == 0 || draft_tokens.empty()) {
        return {};
    }

    // For each draft candidate token, compute pseudo-routing / simulated routing
    for (size_t t = 0; t < draft_tokens.size(); ++t) {
        for (uint32_t k = 0; k < top_k_per_token; ++k) {
            uint32_t exp_id = (static_cast<uint32_t>(draft_tokens[t]) + layer_idx * 7 + k * 13) % target_topo_.n_expert;
            expert_set.insert(exp_id);
        }
    }

    std::vector<uint32_t> result(expert_set.begin(), expert_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

spec_verify_result_t speculative_engine::verify_and_accept(
    const std::vector<int32_t>& draft_tokens,
    const std::vector<int32_t>& target_top_tokens
) {
    spec_verify_result_t res;
    size_t k = std::min(draft_tokens.size(), target_top_tokens.size());
    if (k == 0) return res;

    // Greedy matching verification: accept longest matching prefix
    uint32_t accepted = 0;
    for (size_t i = 0; i < k; ++i) {
        if (draft_tokens[i] == target_top_tokens[i]) {
            res.accepted_tokens.push_back(draft_tokens[i]);
            accepted++;
        } else {
            // First mismatch: target model's prediction takes over
            res.next_token = target_top_tokens[i];
            break;
        }
    }

    res.n_accepted = accepted;
    res.acceptance_rate = static_cast<double>(accepted) / static_cast<double>(k);

    // Update moving average acceptance rate
    avg_acc_rate_ = 0.8 * avg_acc_rate_ + 0.2 * res.acceptance_rate;

    // Feed telemetry into state machine
    runtime_telemetry_t telemetry;
    telemetry.draft_acc_rate = avg_acc_rate_;
    telemetry.cpu_load       = 0.70;
    telemetry.gpu_load       = 0.40;
    telemetry.cache_hit_rate = 0.85;

    sm_.update_telemetry(telemetry);

    return res;
}

uint32_t speculative_engine::get_active_k() const {
    auto policy = sm_.current_policy();
    if (!policy.draft_enabled || !draft_model_loaded_) {
        return 1; // Fall back to standard 1-step autoregressive
    }
    return policy.draft_k_step;
}

} // namespace stream_moe
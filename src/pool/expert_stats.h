#pragma once

#include "common/types.h"
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>

namespace stream_moe {

constexpr uint32_t EST1_MAGIC = 0x31545345; // "EST1" in Little-Endian
constexpr uint32_t DEFAULT_SYNC_TOKEN_THRESHOLD = 8192;

class expert_stats_tracker {
public:
    expert_stats_tracker() = default;
    ~expert_stats_tracker();

    // Prevent copying to maintain thread-safe file sync lifecycle
    expert_stats_tracker(const expert_stats_tracker&) = delete;
    expert_stats_tracker& operator=(const expert_stats_tracker&) = delete;

    // Load existing EST1 file, or initialize with all zeros and create file if not found
    bool init(const std::string& file_path, uint32_t n_layer, uint32_t n_expert, uint32_t sync_threshold = DEFAULT_SYNC_TOKEN_THRESHOLD);

    // Record an access to a specific expert
    void record_access(uint32_t layer_idx, uint32_t expert_idx);

    // Apply multiplicative decay to all scores (recency-weighted decaying counter:
    // score = sum over hits of decay^(age_in_tokens); NOT a normalized EMA)
    void apply_decay(double factor = 0.999);

    // Notify tokens generated; triggers auto-save if cumulative tokens > sync_threshold
    void notify_tokens_generated(uint32_t num_tokens);

    // Force flush to disk
    bool flush();

    // Query metrics
    // Returns the decaying recency-weighted access score, normalized to [0, 1] against
    // the current maximum score. Cold-boot values derive from persisted global counts.
    double   get_adaptive_frequency(uint32_t layer_idx, uint32_t expert_idx) const;
    uint64_t get_global_count(uint32_t layer_idx, uint32_t expert_idx) const;

    uint32_t n_layer() const { return n_layer_; }
    uint32_t n_expert() const { return n_expert_; }
    uint64_t total_historical_tokens() const { return total_tokens_; }

private:
    size_t index(uint32_t l, uint32_t e) const {
        return static_cast<size_t>(l) * n_expert_ + e;
    }

    std::string           file_path_;
    uint32_t              n_layer_ = 0;
    uint32_t              n_expert_ = 0;
    uint32_t              sync_threshold_ = DEFAULT_SYNC_TOKEN_THRESHOLD;
    uint32_t              tokens_since_last_sync_ = 0;
    uint64_t              total_tokens_ = 0;
    bool                  dirty_ = false;

    mutable std::mutex    mutex_;
    std::vector<uint64_t> global_counts_;      // Persisted long-term counts
    std::vector<double>   adaptive_scores_;    // In-memory decaying scores (converging to recent)
};

} // namespace stream_moe
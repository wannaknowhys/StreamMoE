#pragma once

#include "common/types.h"
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <fstream>
#include <mutex>
#include <atomic>

#if defined(_MSC_VER) || defined(__clang__)
#include <intrin.h>
#endif

namespace stream_moe {

inline uint64_t read_timestamp_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

inline uint64_t read_rdtscp(uint32_t* aux = nullptr) {
#if defined(_MSC_VER) || defined(__clang__)
    uint32_t local_aux;
    return __rdtscp(aux ? aux : &local_aux);
#else
    return read_timestamp_ns();
#endif
}

struct turn_profile_t {
    uint32_t turn_id = 0;
    uint64_t timestamp_ns = 0;
    uint32_t prompt_tokens = 0;
    uint32_t generated_tokens = 0;
    double   prefill_tps = 0.0;
    double   decode_tps = 0.0;
    double   total_duration_ms = 0.0;

    // Device Expert Hits
    uint32_t total_lookups = 0;
    uint32_t gpu_hits = 0;
    uint32_t ram_hits = 0;
    uint32_t disk_misses = 0;

    // Speculative Acceptance Histogram: [count_0, count_1, count_2, ..., count_8]
    std::array<uint32_t, 9> spec_accept_hist = {0};

    // Nanosecond Timing Breakdown
    uint64_t t_prefill_ns = 0;
    uint64_t t_prefix_match_ns = 0;
    uint64_t t_attn_layer_ns = 0;
    uint64_t t_expert_total_ns = 0;
    uint64_t t_expert_wait_io_ns = 0;
    uint64_t t_expert_cpu_ns = 0;
    uint64_t t_expert_gpu_ns = 0;
    uint64_t t_sync_pcie_ns = 0;
    uint64_t t_merge_reduce_ns = 0;

    double gpu_hit_rate() const {
        return total_lookups > 0 ? (100.0 * gpu_hits / total_lookups) : 0.0;
    }
    double ram_hit_rate() const {
        return total_lookups > 0 ? (100.0 * ram_hits / total_lookups) : 0.0;
    }
    double total_hit_rate() const {
        return total_lookups > 0 ? (100.0 * (gpu_hits + ram_hits) / total_lookups) : 0.0;
    }
};

class profile_logger {
public:
    static profile_logger& instance() {
        static profile_logger logger;
        return logger;
    }

    void init(const std::string& filepath);
    void close();
    bool is_enabled() const { return enabled_; }

    void log_request_ingest(uint32_t turn_id, size_t prompt_chars, uint32_t prompt_tokens);
    void log_response_finish(const turn_profile_t& profile);

private:
    profile_logger() = default;
    ~profile_logger() { close(); }

    std::atomic<bool> enabled_{false};
    std::string       filepath_;
    std::ofstream     file_;
    std::mutex        mutex_;
};

} // namespace stream_moe
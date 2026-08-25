#include "profile/profiler.h"
#include "common/logger.h"

#include <cstdio>
#include <cstring>
#include <charconv>

namespace stream_moe {

void profile_logger::init(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    filepath_ = filepath;
    if (filepath_.empty()) {
        enabled_ = false;
        return;
    }

    file_.open(filepath_, std::ios::out | std::ios::app);
    if (!file_.is_open()) {
        LOG_ERROR("Failed to open profile log file: " << filepath_);
        enabled_ = false;
        return;
    }

    enabled_ = true;
    LOG_INFO("High-Resolution Profiler initialized. Logging to: " << filepath_);
}

void profile_logger::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
    enabled_ = false;
}

void profile_logger::log_request_ingest(uint32_t turn_id, size_t prompt_chars, uint32_t prompt_tokens) {
    if (!enabled_) return;

    // Fixed-size pre-allocated buffer with fast snprintf (zero dynamic heap allocation)
    char buf[512];
    uint64_t ts = read_timestamp_ns();
    int len = std::snprintf(
        buf, sizeof(buf),
        "{\"event\":\"request_ingest\",\"turn\":%u,\"timestamp_ns\":%llu,\"prompt_chars\":%zu,\"prompt_tokens\":%u}\n",
        turn_id,
        static_cast<unsigned long long>(ts),
        prompt_chars,
        prompt_tokens
    );

    if (len > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.write(buf, len);
            file_.flush();
        }
    }
}

void profile_logger::log_response_finish(const turn_profile_t& p) {
    if (!enabled_) return;

    // Fixed-size pre-allocated buffer formatting
    char buf[2048];
    int len = std::snprintf(
        buf, sizeof(buf),
        "{\"event\":\"response_finish\",\"turn\":%u,\"timestamp_ns\":%llu,\"prompt_tokens\":%u,\"generated_tokens\":%u,"
        "\"prefill_tps\":%.2f,\"decode_tps\":%.2f,\"duration_ms\":%.2f,"
        "\"hits\":{\"total\":%u,\"gpu\":%u,\"ram\":%u,\"disk_miss\":%u,\"gpu_hit_pct\":%.2f,\"ram_hit_pct\":%.2f,\"total_hit_pct\":%.2f},"
        "\"speculative_hist\":[%u,%u,%u,%u,%u,%u,%u,%u,%u],"
        "\"timings_ns\":{\"prefill\":%llu,\"prefix_match\":%llu,\"attn_layer\":%llu,\"expert_total\":%llu,\"expert_wait_io\":%llu,\"expert_cpu\":%llu,\"expert_gpu\":%llu,\"sync_pcie\":%llu,\"merge_reduce\":%llu}}\n",
        p.turn_id,
        static_cast<unsigned long long>(p.timestamp_ns),
        p.prompt_tokens,
        p.generated_tokens,
        p.prefill_tps,
        p.decode_tps,
        p.total_duration_ms,
        p.total_lookups,
        p.gpu_hits,
        p.ram_hits,
        p.disk_misses,
        p.gpu_hit_rate(),
        p.ram_hit_rate(),
        p.total_hit_rate(),
        p.spec_accept_hist[0], p.spec_accept_hist[1], p.spec_accept_hist[2], p.spec_accept_hist[3],
        p.spec_accept_hist[4], p.spec_accept_hist[5], p.spec_accept_hist[6], p.spec_accept_hist[7], p.spec_accept_hist[8],
        static_cast<unsigned long long>(p.t_prefill_ns),
        static_cast<unsigned long long>(p.t_prefix_match_ns),
        static_cast<unsigned long long>(p.t_attn_layer_ns),
        static_cast<unsigned long long>(p.t_expert_total_ns),
        static_cast<unsigned long long>(p.t_expert_wait_io_ns),
        static_cast<unsigned long long>(p.t_expert_cpu_ns),
        static_cast<unsigned long long>(p.t_expert_gpu_ns),
        static_cast<unsigned long long>(p.t_sync_pcie_ns),
        static_cast<unsigned long long>(p.t_merge_reduce_ns)
    );

    if (len > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.write(buf, len);
            file_.flush();
        }
    }
}

} // namespace stream_moe
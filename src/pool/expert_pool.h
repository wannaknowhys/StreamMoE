#pragma once

#include "common/types.h"
#include "pool/expert_stats.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <string>

namespace stream_moe {

enum slot_status_flags : uint32_t {
    SLOT_EMPTY       = 0,
    SLOT_PIN_LOCKED  = 1 << 0, // In use by compute thread (cannot be evicted)
    SLOT_IO_INFLIGHT = 1 << 1, // Async DIO / DMA transfer in progress
    SLOT_READY       = 1 << 2, // Valid expert payload in memory
};

enum class eviction_policy_t {
    HYBRID_EST1 = 0, // Default: EST1 historical stats + exponential decay towards recent
    PURE_LRU    = 1, // Pure Least Recently Used (zero prior knowledge / stats ignored)
    PURE_LFU    = 2  // Pure Least Frequently Used in current session
};

inline eviction_policy_t parse_eviction_policy(const std::string& name) {
    if (name == "lru" || name == "pure_lru") return eviction_policy_t::PURE_LRU;
    if (name == "lfu" || name == "pure_lfu") return eviction_policy_t::PURE_LFU;
    return eviction_policy_t::HYBRID_EST1;
}

inline const char* eviction_policy_name(eviction_policy_t policy) {
    switch (policy) {
        case eviction_policy_t::PURE_LRU: return "PURE_LRU";
        case eviction_policy_t::PURE_LFU: return "PURE_LFU";
        case eviction_policy_t::HYBRID_EST1:
        default: return "HYBRID_EST1";
    }
}

struct expert_slot_t {
    int32_t               layer_idx       = -1;
    int32_t               expert_idx      = -1;
    std::atomic<uint32_t> flags           {SLOT_EMPTY};
    uint64_t              last_access_seq = 0;
    uint32_t              session_hits    = 0;
    uint8_t*              raw_ptr         = nullptr;

    expert_slot_t() = default;
    expert_slot_t(const expert_slot_t& other)
        : layer_idx(other.layer_idx),
          expert_idx(other.expert_idx),
          flags(other.flags.load(std::memory_order_relaxed)),
          last_access_seq(other.last_access_seq),
          session_hits(other.session_hits),
          raw_ptr(other.raw_ptr) {}
    expert_slot_t& operator=(const expert_slot_t& other) {
        if (this != &other) {
            layer_idx = other.layer_idx;
            expert_idx = other.expert_idx;
            flags.store(other.flags.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_access_seq = other.last_access_seq;
            session_hits = other.session_hits;
            raw_ptr = other.raw_ptr;
        }
        return *this;
    }
};

class expert_pool {
public:
    expert_pool(size_t slot_size, uint32_t num_slots, eviction_policy_t policy = eviction_policy_t::HYBRID_EST1);
    ~expert_pool();

    // Disable copy/move for static memory safety
    expert_pool(const expert_pool&) = delete;
    expert_pool& operator=(const expert_pool&) = delete;

    // Lookup slot by layer and expert index (-1 if miss)
    int32_t find_slot(int32_t layer_idx, int32_t expert_idx) const;

    // Pin a slot to protect it from eviction during GEMM execution
    bool pin_slot(int32_t slot_id);

    // Unpin a slot after layer computation completes
    void unpin_slot(int32_t slot_id);

    // Mark slot as ready for computation
    void mark_ready(int32_t slot_id);

    // Allocate an empty slot or evict a victim slot
    int32_t allocate_or_evict_slot(
        int32_t layer_idx,
        int32_t expert_idx,
        const expert_stats_tracker& stats,
        uint64_t current_seq,
        double w_lru = 0.5,
        double w_freq = 0.5
    );

    // Getters / Setters
    expert_slot_t& get_slot(uint32_t slot_id) { return slots_[slot_id]; }
    const expert_slot_t& get_slot(uint32_t slot_id) const { return slots_[slot_id]; }
    
    size_t             slot_size() const { return slot_size_; }
    uint32_t           num_slots() const { return num_slots_; }
    size_t             total_bytes() const { return total_allocated_bytes_; }
    uint8_t*           base_ptr() { return base_ptr_; }
    eviction_policy_t  policy() const { return policy_; }
    void               set_policy(eviction_policy_t p) { policy_ = p; }
    // True only if the OS confirmed the whole pool is locked into physical RAM
    bool               is_pinned() const { return is_pinned_; }

private:
    uint64_t make_key(int32_t l, int32_t e) const {
        return (static_cast<uint64_t>(static_cast<uint32_t>(l)) << 32) | static_cast<uint32_t>(e);
    }

    size_t                                slot_size_ = 0;
    uint32_t                              num_slots_ = 0;
    size_t                                total_allocated_bytes_ = 0;
    uint8_t*                              base_ptr_ = nullptr;
    bool                                  is_pinned_ = false;
    eviction_policy_t                     policy_ = eviction_policy_t::HYBRID_EST1;
    std::unique_ptr<expert_slot_t[]>      slots_;

    mutable std::mutex                    mutex_;
    std::unordered_map<uint64_t, int32_t> lookup_map_; // (layer, expert) -> slot_id
};

} // namespace stream_moe
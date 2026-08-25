#include "pool/expert_pool.h"
#include "common/logger.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace stream_moe {

expert_pool::expert_pool(size_t slot_size, uint32_t num_slots, eviction_policy_t policy)
    : slot_size_(slot_size),
      num_slots_(num_slots),
      policy_(policy),
      slots_(std::make_unique<expert_slot_t[]>(num_slots)) {
    
    total_allocated_bytes_ = slot_size_ * num_slots_;

#if defined(_WIN32)
    base_ptr_ = static_cast<uint8_t*>(VirtualAlloc(
        NULL,
        total_allocated_bytes_,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE
    ));

    if (!base_ptr_) {
        LOG_ERROR("VirtualAlloc failed for " << (total_allocated_bytes_ / (1024 * 1024)) << " MB");
        throw std::bad_alloc();
    }

    // Try to lock pages in physical RAM (ignore error if process working set quota is reached)
    VirtualLock(base_ptr_, total_allocated_bytes_);
#else
    if (posix_memalign(reinterpret_cast<void**>(&base_ptr_), 4096, total_allocated_bytes_) != 0) {
        LOG_ERROR("posix_memalign failed for " << (total_allocated_bytes_ / (1024 * 1024)) << " MB");
        throw std::bad_alloc();
    }
    mlock(base_ptr_, total_allocated_bytes_);
#endif

    // Initialize slot descriptors
    for (uint32_t i = 0; i < num_slots_; ++i) {
        slots_[i].layer_idx       = -1;
        slots_[i].expert_idx      = -1;
        slots_[i].flags.store(SLOT_EMPTY, std::memory_order_relaxed);
        slots_[i].last_access_seq = 0;
        slots_[i].session_hits    = 0;
        slots_[i].raw_ptr         = base_ptr_ + i * slot_size_;
    }

    LOG_INFO("Initialized Pinned Expert Pool: " << num_slots_ << " slots of " 
             << (slot_size_ / 1024) << " KB each (Total: " 
             << (total_allocated_bytes_ / (1024 * 1024)) << " MB, Policy: " 
             << eviction_policy_name(policy_) << ")");
}

expert_pool::~expert_pool() {
    if (base_ptr_) {
#if defined(_WIN32)
        VirtualUnlock(base_ptr_, total_allocated_bytes_);
        VirtualFree(base_ptr_, 0, MEM_RELEASE);
#else
        munlock(base_ptr_, total_allocated_bytes_);
        free(base_ptr_);
#endif
        base_ptr_ = nullptr;
    }
}

int32_t expert_pool::find_slot(int32_t layer_idx, int32_t expert_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t key = make_key(layer_idx, expert_idx);
    auto it = lookup_map_.find(key);
    if (it != lookup_map_.end()) {
        int32_t slot_id = it->second;
        uint32_t f = slots_[slot_id].flags.load(std::memory_order_relaxed);
        if (f & SLOT_READY) {
            return slot_id;
        }
    }
    return -1;
}

bool expert_pool::pin_slot(int32_t slot_id) {
    if (slot_id < 0 || slot_id >= static_cast<int32_t>(num_slots_)) return false;
    uint32_t old_flags = slots_[slot_id].flags.fetch_or(SLOT_PIN_LOCKED, std::memory_order_acq_rel);
    return (old_flags & SLOT_READY) != 0;
}

void expert_pool::unpin_slot(int32_t slot_id) {
    if (slot_id < 0 || slot_id >= static_cast<int32_t>(num_slots_)) return;
    slots_[slot_id].flags.fetch_and(~SLOT_PIN_LOCKED, std::memory_order_release);
}

void expert_pool::mark_ready(int32_t slot_id) {
    if (slot_id < 0 || slot_id >= static_cast<int32_t>(num_slots_)) return;
    slots_[slot_id].flags.fetch_or(SLOT_READY, std::memory_order_release);
    slots_[slot_id].flags.fetch_and(~SLOT_IO_INFLIGHT, std::memory_order_release);
}

int32_t expert_pool::allocate_or_evict_slot(
    int32_t layer_idx,
    int32_t expert_idx,
    const expert_stats_tracker& stats,
    uint64_t current_seq,
    double w_lru,
    double w_freq
) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Check if already present
    uint64_t key = make_key(layer_idx, expert_idx);
    auto it = lookup_map_.find(key);
    if (it != lookup_map_.end()) {
        int32_t slot_id = it->second;
        slots_[slot_id].last_access_seq = current_seq;
        slots_[slot_id].session_hits++;
        return slot_id;
    }

    // 2. Check for empty slot
    for (uint32_t i = 0; i < num_slots_; ++i) {
        uint32_t f = slots_[i].flags.load(std::memory_order_relaxed);
        if (f == SLOT_EMPTY) {
            slots_[i].layer_idx       = layer_idx;
            slots_[i].expert_idx      = expert_idx;
            slots_[i].last_access_seq = current_seq;
            slots_[i].session_hits    = 1;
            slots_[i].flags.store(SLOT_IO_INFLIGHT, std::memory_order_release);
            lookup_map_[key] = static_cast<int32_t>(i);
            return static_cast<int32_t>(i);
        }
    }

    // 3. Find victim slot according to policy
    int32_t victim_slot = -1;

    if (policy_ == eviction_policy_t::PURE_LRU) {
        // Pure LRU: strictly find the unlocked slot with oldest last_access_seq (ignore stats)
        uint64_t oldest_seq = std::numeric_limits<uint64_t>::max();
        for (uint32_t i = 0; i < num_slots_; ++i) {
            uint32_t f = slots_[i].flags.load(std::memory_order_relaxed);
            if (f & (SLOT_PIN_LOCKED | SLOT_IO_INFLIGHT)) continue;

            if (slots_[i].last_access_seq < oldest_seq) {
                oldest_seq = slots_[i].last_access_seq;
                victim_slot = static_cast<int32_t>(i);
            }
        }
    } else if (policy_ == eviction_policy_t::PURE_LFU) {
        // Pure LFU: find slot with lowest session hits
        uint32_t min_hits = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < num_slots_; ++i) {
            uint32_t f = slots_[i].flags.load(std::memory_order_relaxed);
            if (f & (SLOT_PIN_LOCKED | SLOT_IO_INFLIGHT)) continue;

            if (slots_[i].session_hits < min_hits) {
                min_hits = slots_[i].session_hits;
                victim_slot = static_cast<int32_t>(i);
            }
        }
    } else {
        // Hybrid EST1: combine normalized LRU age + adaptive frequency
        double min_score = std::numeric_limits<double>::infinity();
        uint64_t min_seq = current_seq;
        uint64_t max_seq = current_seq;
        for (uint32_t i = 0; i < num_slots_; ++i) {
            uint32_t f = slots_[i].flags.load(std::memory_order_relaxed);
            if (!(f & (SLOT_PIN_LOCKED | SLOT_IO_INFLIGHT))) {
                if (slots_[i].last_access_seq < min_seq) min_seq = slots_[i].last_access_seq;
                if (slots_[i].last_access_seq > max_seq) max_seq = slots_[i].last_access_seq;
            }
        }
        double seq_range = (max_seq > min_seq) ? static_cast<double>(max_seq - min_seq) : 1.0;

        for (uint32_t i = 0; i < num_slots_; ++i) {
            uint32_t f = slots_[i].flags.load(std::memory_order_relaxed);
            if (f & (SLOT_PIN_LOCKED | SLOT_IO_INFLIGHT)) continue;

            double lru_norm = static_cast<double>(slots_[i].last_access_seq - min_seq) / seq_range;
            double freq_val = stats.get_adaptive_frequency(slots_[i].layer_idx, slots_[i].expert_idx);
            double score = w_lru * lru_norm + w_freq * freq_val;

            if (score < min_score) {
                min_score = score;
                victim_slot = static_cast<int32_t>(i);
            }
        }
    }

    if (victim_slot == -1) {
        LOG_WARN("allocate_or_evict_slot: all " << num_slots_ << " slots are locked/in-flight!");
        return -1;
    }

    // 4. Evict victim slot
    uint64_t old_key = make_key(slots_[victim_slot].layer_idx, slots_[victim_slot].expert_idx);
    lookup_map_.erase(old_key);

    slots_[victim_slot].layer_idx       = layer_idx;
    slots_[victim_slot].expert_idx      = expert_idx;
    slots_[victim_slot].last_access_seq = current_seq;
    slots_[victim_slot].session_hits    = 1;
    slots_[victim_slot].flags.store(SLOT_IO_INFLIGHT, std::memory_order_release);

    lookup_map_[key] = victim_slot;
    return victim_slot;
}

} // namespace stream_moe
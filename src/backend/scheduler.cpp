#include "backend/scheduler.h"
#include "common/logger.h"

#include <cstring>
#include <chrono>

namespace stream_moe {

expert_scheduler::~expert_scheduler() {
    stop();
    if (pool_base_) {
        async_dio_engine::free_aligned(pool_base_);
        pool_base_ = nullptr;
    }
}

bool expert_scheduler::init(const moe_model_topology_t& topo, async_dio_engine& dio,
                            const std::vector<dio_file_t*>& files, size_t pool_bytes) {
    topo_ = &topo;
    dio_  = &dio;
    files_ = files;

    slot_size_ = topo.expert_slot_size > 0 ? topo.expert_slot_size : (4096 * 4);
    // Cap slot count to the model's total expert count.
    uint32_t n_experts_total = static_cast<uint32_t>(topo.n_layer) * topo.n_expert;
    size_t max_slots = pool_bytes / slot_size_;
    num_slots_ = static_cast<uint32_t>(std::min<size_t>(max_slots, n_experts_total));
    if (num_slots_ == 0) num_slots_ = 1;
    pool_bytes_ = static_cast<size_t>(num_slots_) * slot_size_;

    pool_base_ = static_cast<uint8_t*>(async_dio_engine::alloc_aligned(pool_bytes_));
    if (!pool_base_) {
        LOG_ERROR("expert_scheduler: pool alloc failed for " << (pool_bytes_ / (1024 * 1024)) << " MB");
        return false;
    }

    slots_ = std::make_unique<slot_meta[]>(num_slots_);
    owner_.resize(num_slots_, {0, 0});
    dir_owned_ = std::make_unique<expert_directory>(topo.n_layer, topo.n_expert, num_slots_);
    dir_ = dir_owned_.get();

    staging_ = make_aligned_buffer(topo.expert_dio_staging_size > 0 ? topo.expert_dio_staging_size : (1024 * 1024));
    stats_.init("", topo.n_layer, topo.n_expert, 8192);

    LOG_INFO("Expert pool: " << num_slots_ << " slots x " << (slot_size_ / 1024)
             << " KB = " << (pool_bytes_ / (1024 * 1024)) << " MB committed (hard cap on expert residency)");
    return true;
}

void expert_scheduler::start() {
    if (running_.exchange(true)) return;
    worker_ = std::thread(&expert_scheduler::scheduler_loop, this);
}

void expert_scheduler::stop() {
    if (!running_.exchange(false)) return;
    if (worker_.joinable()) worker_.join();
}

void expert_scheduler::clear_directory(uint32_t layer, uint32_t expert) {
    if (dir_->find(layer, expert) != SLOT_UNASSIGNED) {
        dir_->set(layer, expert, SLOT_UNASSIGNED);
    }
}

int32_t expert_scheduler::alloc_or_evict(uint32_t layer, uint32_t expert) {
    // 1. free slot
    for (uint32_t i = 0; i < num_slots_; ++i) {
        if (slot_word_state(slots_[i].load()) == SLOT_EMPTY) {
            slots_[i].begin_reload();
            owner_[i] = {layer, expert};
            return static_cast<int32_t>(i);
        }
    }

    // 2. evict victim: READY && refcount==0, lowest hybrid score (EST1) then oldest
    int32_t victim = -1;
    double  best_score = std::numeric_limits<double>::infinity();
    uint64_t best_seq = 0;
    for (uint32_t i = 0; i < num_slots_; ++i) {
        uint64_t w = slots_[i].load();
        if (slot_word_state(w) != SLOT_READY || slot_word_refcount(w) != 0) continue;
        double freq = stats_.get_adaptive_frequency(owner_[i].first, owner_[i].second);
        double score = 0.5 * freq + 0.5 * (1.0 - (static_cast<double>(slot_word_generation(w)) / 1e9));
        if (score < best_score) {
            best_score = score;
            victim = static_cast<int32_t>(i);
            best_seq = slot_word_generation(w);
        }
    }
    if (victim < 0) return -1; // all slots pinned/in-flight

    // 3. evict: READY -> EVICTING, clear directory, drain refcount, reuse
    if (!slots_[victim].begin_evict()) return -1;
    clear_directory(owner_[victim].first, owner_[victim].second);
    while (slot_word_refcount(slots_[victim].load()) != 0) {
        std::this_thread::yield();
    }
    slots_[victim].begin_reload();
    owner_[victim] = {layer, expert};
    (void)best_seq;
    return victim;
}

void expert_scheduler::load_slot(int32_t slot, uint32_t layer, uint32_t expert) {
    const expert_info_t& info = topo_->get_expert(layer, expert);
    bool ok = read_expert_sync(dio_, files_, info.read_plan, staging_.get(),
                               pool_base_ + static_cast<size_t>(slot) * slot_size_);
    if (ok) {
        slots_[slot].mark_ready();
        dir_->set(layer, expert, static_cast<uint32_t>(slot));
        n_misses_.fetch_add(1, std::memory_order_relaxed);
    } else {
        slots_[slot].mark_failed();
        LOG_ERROR("expert_scheduler: DIO load failed for L" << layer << " E" << expert);
    }
}

void expert_scheduler::scheduler_loop() {
    slot_request_t req;
    while (running_) {
        if (requests_.pop(req)) {
            // Skip if already resident (duplicate request raced us).
            if (dir_->find(req.layer, req.expert) != SLOT_UNASSIGNED) continue;
            int32_t slot = alloc_or_evict(req.layer, req.expert);
            if (slot < 0) {
                // pool fully pinned right now; retry shortly
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                requests_.push(req);
                continue;
            }
            load_slot(slot, req.layer, req.expert);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

expert_handle_t expert_scheduler::pin_expert(uint32_t layer, uint32_t expert) {
    n_lookups_.fetch_add(1, std::memory_order_relaxed);
    bool waited_for_load = false;
    for (;;) {
        uint32_t s = dir_->find(layer, expert);
        if (s == SLOT_UNASSIGNED) {
            waited_for_load = true;
            uint32_t v = dir_->version(layer, expert);
            requests_.push({layer, expert, static_cast<uint32_t>(seq_.fetch_add(1, std::memory_order_relaxed))});
            dir_->wait_version(layer, expert, v);
            continue;
        }
        int64_t gen = slots_[s].try_pin();
        if (gen >= 0) {
            if (!waited_for_load) {
                n_hits_.fetch_add(1, std::memory_order_relaxed);
            }
            return {static_cast<int32_t>(s), static_cast<uint32_t>(gen), layer, expert, true};
        }
        std::this_thread::yield(); // transient (evicting); retry
    }
}

void expert_scheduler::wait_ready(uint32_t layer, uint32_t expert) {
    for (;;) {
        uint32_t s = dir_->find(layer, expert);
        if (s == SLOT_UNASSIGNED) {
            uint32_t v = dir_->version(layer, expert);
            requests_.push({layer, expert, static_cast<uint32_t>(seq_.fetch_add(1, std::memory_order_relaxed))});
            dir_->wait_version(layer, expert, v);
            continue;
        }
        if (slot_word_state(slots_[s].load()) == SLOT_READY) return;
        std::this_thread::yield();
    }
}

void expert_scheduler::unpin(const expert_handle_t& h) {
    if (h.slot >= 0 && h.slot < static_cast<int32_t>(num_slots_)) {
        slots_[h.slot].unpin();
    }
}

} // namespace stream_moe

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

    // Carve per-expert-group sub-pools by byte fraction (docs/MULTI_SUBPOOL.md).
    // Slot count = group budget / group expert size, so both the slot ratio and
    // the byte ratio match the source model's expert composition.
    uint64_t total_bytes = 0;
    for (const auto& g : topo.groups) total_bytes += g.total_bytes;
    if (total_bytes == 0) {
        LOG_ERROR("expert_scheduler: no expert groups in topology");
        return false;
    }

    num_slots_ = 0;
    for (const auto& g : topo.groups) {
        // byte-fraction budget; use double to avoid 64-bit overflow for big pools
        size_t budget = static_cast<size_t>(
            static_cast<double>(pool_bytes) * (static_cast<double>(g.total_bytes) / static_cast<double>(total_bytes)));
        uint32_t ns = static_cast<uint32_t>(std::max<size_t>(1, budget / g.expert_size));
        uint32_t g_experts_total = static_cast<uint32_t>(g.layers.size()) * topo.n_expert;
        if (ns > g_experts_total) ns = g_experts_total;
        subpools_.push_back({ num_slots_, ns, g.expert_size, nullptr });
        num_slots_ += ns;
    }

    size_t total_commit = 0;
    for (const auto& sp : subpools_) total_commit += static_cast<size_t>(sp.n_slots) * sp.expert_size;
    pool_bytes_ = total_commit;

    pool_base_ = static_cast<uint8_t*>(async_dio_engine::alloc_aligned(pool_bytes_));
    if (!pool_base_) {
        LOG_ERROR("expert_scheduler: pool alloc failed for " << (pool_bytes_ / (1024 * 1024)) << " MB");
        return false;
    }
    uint8_t* b = pool_base_;
    for (auto& sp : subpools_) { sp.base = b; b += static_cast<size_t>(sp.n_slots) * sp.expert_size; }

    slots_ = std::make_unique<slot_meta[]>(num_slots_);
    owner_.resize(num_slots_, {0, 0});
    dir_owned_ = std::make_unique<expert_directory>(topo.n_layer, topo.n_expert, num_slots_);
    dir_ = dir_owned_.get();

    // One staging buffer per expert group, sized to the group's MAX expert
    // layout (per-expert staging needs differ due to file-offset alignment).
    staging_per_group_.clear();
    staging_per_group_.reserve(topo.groups.size());
    for (const auto& g : topo.groups) {
        size_t max_sz = 0;
        for (uint32_t l : g.layers) {
            for (uint32_t e = 0; e < topo.n_expert; ++e) {
                size_t sz = topo.get_expert(l, e).read_plan.total_staging_size;
                if (sz > max_sz) max_sz = sz;
            }
        }
        staging_per_group_.push_back(make_aligned_buffer(max_sz > 0 ? max_sz : (1024 * 1024)));
    }
    stats_.init("", topo.n_layer, topo.n_expert, 8192);

    LOG_INFO("Expert pool: " << num_slots_ << " slots across " << subpools_.size()
             << " groups = " << (pool_bytes_ / (1024 * 1024)) << " MB committed (hard cap on expert residency)");
    for (const auto& sp : subpools_) {
        LOG_INFO("  subpool: slots [" << sp.slot_begin << ", " << (sp.slot_begin + sp.n_slots)
                 << ") x " << (sp.expert_size / 1024) << "KB");
    }
    return true;
}

uint32_t expert_scheduler::group_of(uint32_t layer) const {
    if (!topo_ || layer >= topo_->n_layer) return static_cast<uint32_t>(-1);
    for (size_t i = 0; i < topo_->groups.size(); ++i) {
        for (uint32_t l : topo_->groups[i].layers) {
            if (l == layer) return static_cast<uint32_t>(i);
        }
    }
    return static_cast<uint32_t>(-1);
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
    uint32_t gidx = group_of(layer);
    if (gidx == static_cast<uint32_t>(-1)) return -1;
    const subpool_t& sp = subpools_[gidx];
    const uint32_t lo = sp.slot_begin, hi = sp.slot_begin + sp.n_slots;

    // 1. free slot (within this group's sub-pool)
    for (uint32_t i = lo; i < hi; ++i) {
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
    for (uint32_t i = lo; i < hi; ++i) {
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
    uint32_t gidx = group_of(layer);
    if (gidx == static_cast<uint32_t>(-1) || !staging_per_group_[gidx]) return;
    bool ok = read_expert_sync(dio_, files_, info.read_plan, staging_per_group_[gidx].get(), slot_mem(slot));
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
    for (int retries = 0; retries < 100000; ++retries) {
        uint32_t s = dir_->find(layer, expert);
        if (s == SLOT_UNASSIGNED) {
            waited_for_load = true;
            uint32_t v = dir_->version(layer, expert);
            requests_.push({layer, expert, static_cast<uint32_t>(seq_.fetch_add(1, std::memory_order_relaxed))});
            dir_->wait_version(layer, expert, v);
            continue;
        }
        uint32_t st = slot_word_state(slots_[s].load());
        if (st == SLOT_FAILED) {
            LOG_ERROR("expert_scheduler: pin_expert FAILED for L" << layer << " E" << expert);
            return {-1, 0, layer, expert, false};
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
    LOG_ERROR("expert_scheduler: pin_expert retry limit hit for L" << layer << " E" << expert);
    return {-1, 0, layer, expert, false};
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

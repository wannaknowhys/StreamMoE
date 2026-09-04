#include "backend/scheduler.h"
#include "common/logger.h"
#include "common/tsc.h"
#include "common/types.h"

#include <cstring>
#include <algorithm>
#include <chrono>

namespace stream_moe {

expert_scheduler::~expert_scheduler() {
    stop();
    if (pool_base_) {
        async_dio_engine::free_aligned(pool_base_);
        pool_base_ = nullptr;
    }
    if (load_pool_) {
        async_dio_engine::free_aligned(load_pool_);
        load_pool_ = nullptr;
    }
}

bool expert_scheduler::init(const moe_model_topology_t& topo, async_dio_engine& dio,
                            const std::vector<dio_file_t*>& files, size_t pool_bytes,
                            const std::vector<vram_region_t>& vregions) {
    topo_ = &topo;
    dio_  = &dio;
    files_ = files;

    // Batch bitmap capacity check (2026-09): requests encode a layer's expert
    // set as a fixed-size bitmap. Fail fast here (load-time) instead of at a
    // random compute step when an expert index overflows the bitmap.
    if (topo.n_expert > MAX_EXPERTS_PER_LAYER) {
        LOG_ERROR("expert_scheduler: model has " << topo.n_expert
                  << " experts/layer, batch bitmap supports " << MAX_EXPERTS_PER_LAYER
                  << " - increase MAX_EXPERTS_PER_LAYER in slot.h");
        return false;
    }

    // Carve per-expert-group sub-pools. Hard floor per group: enough slots to
    // hold one full layer's expert set (per_layer x expert_size). A single
    // decode may pin any subset of one layer, and eviction cannot reclaim slots
    // pinned by the very layer being computed - a group smaller than one full
    // layer deadlocks (NO_VICTIM requeue storm) once a layer's active set
    // exceeds the group. Spare budget (pool - sum of floors) is split by byte
    // fraction so multi-layer groups keep eviction head-room. If the pool cannot
    // even meet the floors, fail fast at init instead of deadlocking at runtime.
    uint64_t total_bytes = 0;
    for (const auto& g : topo.groups) total_bytes += g.total_bytes;
    if (total_bytes == 0) {
        LOG_ERROR("expert_scheduler: no expert groups in topology");
        return false;
    }

    const uint32_t per_layer = topo.n_expert;   // experts per layer = one layer's worst-case active set
    size_t sum_floor = 0;
    std::vector<size_t> floor_bytes(topo.groups.size());
    for (size_t i = 0; i < topo.groups.size(); ++i) {
        floor_bytes[i] = static_cast<size_t>(per_layer) * topo.groups[i].expert_size;
        sum_floor += floor_bytes[i];
    }
    if (sum_floor > pool_bytes) {
        LOG_ERROR("expert_scheduler: pool " << (pool_bytes / (1024 * 1024)) << " MB too small: needs at least "
                  << (sum_floor / (1024 * 1024)) << " MB to hold one full expert layer per group ("
                  << per_layer << " experts/layer). Increase --moe-ram-pool.");
        return false;
    }
    const size_t spare = pool_bytes - sum_floor;

    num_slots_ = 0;
    const size_t n_groups = topo.groups.size();
    // Fill SoA column geometry for a region given its topo group columns and
    // slot count: column-major layout - col c occupies n_slots*stride_c bytes.
    auto fill_cols = [&](subpool_t& sp, uint32_t group, uint32_t ns) {
        sp.cols.clear();
        const auto& gcols = topo.groups[group].columns;
        size_t off = 0;
        for (size_t c = 0; c < gcols.size(); ++c) {
            column_t col;
            col.index = static_cast<uint32_t>(c);
            col.tag = gcols[c].tag;
            col.stride = gcols[c].per_expert;
            col.off = off;
            sp.cols.push_back(col);
            off += static_cast<size_t>(ns) * col.stride;
        }
    };
    for (size_t i = 0; i < n_groups; ++i) {
        const auto& g = topo.groups[i];
        // floor + byte-fraction share of the spare (double to avoid 64-bit overflow)
        size_t budget = floor_bytes[i] + static_cast<size_t>(
            static_cast<double>(spare) * (static_cast<double>(g.total_bytes) / static_cast<double>(total_bytes)));
        uint32_t ns = static_cast<uint32_t>(budget / g.expert_size);
        const uint32_t g_experts_total = static_cast<uint32_t>(g.layers.size()) * per_layer;
        if (ns < per_layer) ns = per_layer;             // never below one full layer
        if (ns > g_experts_total) ns = g_experts_total; // never more than the group's real experts
        subpool_t sp;
        sp.pool = 0;
        sp.group = static_cast<uint32_t>(i);
        sp.slot_begin = num_slots_;
        sp.n_slots = ns;
        sp.expert_size = g.expert_size;   // = sum of column strides
        fill_cols(sp, static_cast<uint32_t>(i), ns);
        subpools_.push_back(std::move(sp));
        num_slots_ += ns;
    }

    // Device (VRAM) regions: appended slot runs on top of the RAM sub-pools.
    // A region must carry a usable host-mapped base; anything else is skipped
    // with a warning (the pool still runs on CPU RAM alone).
    uint32_t max_pool = 0;
    for (const auto& reg : vregions) {
        if (reg.base == nullptr || reg.group >= n_groups || reg.n_slots == 0) {
            LOG_WARN("expert_scheduler: skipping vram region pool=" << reg.pool
                     << " group=" << reg.group << " (base=" << (void*)reg.base
                     << " n_slots=" << reg.n_slots << ")");
            continue;
        }
        const size_t es = topo.groups[reg.group].expert_size;
        subpool_t sp;
        sp.pool = reg.pool;
        sp.group = reg.group;
        sp.slot_begin = num_slots_;
        sp.n_slots = reg.n_slots;
        sp.expert_size = es;
        sp.base = reg.base;
        sp.dev_buf = reg.buf;
        fill_cols(sp, reg.group, reg.n_slots);
        subpools_.push_back(std::move(sp));
        num_slots_ += reg.n_slots;
        if (reg.pool > max_pool) max_pool = reg.pool;
    }
    n_pools_ = max_pool + 1;

    // RAM commit = the sum of the CPU-RAM regions only (device regions live in
    // their own host-mapped buffers, not in the RAM pool allocation).
    size_t total_commit = 0;
    for (const auto& sp : subpools_) {
        if (sp.pool == 0) total_commit += static_cast<size_t>(sp.n_slots) * sp.expert_size;
    }
    pool_bytes_ = total_commit;

    pool_base_ = static_cast<uint8_t*>(async_dio_engine::alloc_aligned(pool_bytes_));
    if (!pool_base_) {
        LOG_ERROR("expert_scheduler: pool alloc failed for " << (pool_bytes_ / (1024 * 1024)) << " MB");
        return false;
    }
    uint8_t* b = pool_base_;
    for (auto& sp : subpools_) {
        if (sp.pool != 0) continue;   // device bases come from the caller's buffers
        sp.base = b;
        b += static_cast<size_t>(sp.n_slots) * sp.expert_size;
    }

    slots_ = std::make_unique<slot_meta[]>(num_slots_);
    owner_.resize(num_slots_, {0, 0});
    dir_owned_ = std::make_unique<expert_directory>(topo.n_layer, topo.n_expert, n_pools_, num_slots_);
    dir_ = dir_owned_.get();

    // Initial hit rate prior = pool bytes / total expert bytes (recency-vs-freq
    // blend prior before real traffic; alpha = clamp(1 - hit_rate_ema, 0.1, 0.9)).
    hit_rate_ema_ = total_bytes > 0 ? (static_cast<double>(pool_bytes_) / static_cast<double>(total_bytes)) : 0.5;
    if (hit_rate_ema_ > 0.95) hit_rate_ema_ = 0.95;
    alpha_.store(1.0 - hit_rate_ema_, std::memory_order_relaxed);

    // Async-load ringbuffer pool (one per model). staging size = max per-expert
    // read-plan staging over all groups (non-4K columns need a staging window;
    // 4K-direct columns stage nothing). Element stride is 4K-aligned.
    load_staging_size_ = 0;
    for (const auto& g : topo.groups) {
        size_t gmax = 0;
        for (uint32_t l : g.layers) {
            for (uint32_t e = 0; e < topo.n_expert; ++e) {
                size_t sz = topo.get_expert(l, e).read_plan.total_staging_size;
                if (sz > gmax) gmax = sz;
            }
        }
        if (gmax > load_staging_size_) load_staging_size_ = gmax;
    }
    // Concurrency = one layer's worst-case expert touch set (n_expert) scaled by
    // device sub-pool count.
    const uint32_t dev_factor    = 1; // CPU-only today; GPU sub-pools multiply later
    const uint32_t layout_factor = load_staging_size_ ? 1u : 2u;
    max_in_flight_ = std::max<uint32_t>(16u, topo.n_expert * dev_factor * layout_factor);
    const size_t header_sz = align_ceil(sizeof(async_load_t), DIO_SECTOR_SIZE);
    load_stride_ = header_sz + align_ceil(load_staging_size_, DIO_SECTOR_SIZE);
    load_pool_ = static_cast<uint8_t*>(async_dio_engine::alloc_aligned(load_stride_ * max_in_flight_));
    if (!load_pool_) {
        LOG_ERROR("expert_scheduler: async load pool alloc failed");
        return false;
    }
    load_free_.resize(max_in_flight_);
    for (uint32_t i = 0; i < max_in_flight_; ++i) load_free_[i] = i;
    load_in_use_.assign(max_in_flight_, 0);
    stats_.init("", topo.n_layer, topo.n_expert, 8192);

    LOG_INFO("Expert pool: " << num_slots_ << " slots across " << n_groups
             << " expert groups (RAM " << (pool_bytes_ / (1024 * 1024))
             << " MB committed; " << (subpools_.size() - n_groups) << " device region(s))");
    for (const auto& sp : subpools_) {
        LOG_INFO("  subpool: pool=" << sp.pool << " group=" << sp.group
                 << " slots [" << sp.slot_begin << ", " << (sp.slot_begin + sp.n_slots)
                 << ") x " << (sp.expert_size / 1024) << "KB @ " << (void*) sp.base);
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

const expert_scheduler::subpool_t* expert_scheduler::ram_subpool(uint32_t group) const {
    for (const auto& sp : subpools_) {
        if (sp.pool == 0 && sp.group == group) return &sp;
    }
    return nullptr;
}

void expert_scheduler::start() {
    if (running_.exchange(true)) return;
    global_expert_scheduler::instance().register_scheduler(this);
}

void expert_scheduler::stop() {
    if (!running_.exchange(false)) return;
    global_expert_scheduler::instance().unregister_scheduler(this);
}

void expert_scheduler::clear_directory(uint32_t layer, uint32_t expert) {
    uint32_t pool = 0;
    uint32_t s = dir_->scan(layer, expert, &pool);
    if (s != SLOT_UNASSIGNED) {
        dir_->clear(layer, expert, pool);
    }
}

int32_t expert_scheduler::alloc_or_evict(uint32_t layer, uint32_t expert, uint32_t pool) {
    uint32_t gidx = group_of(layer);
    if (gidx == static_cast<uint32_t>(-1)) return -1;
    const subpool_t* sp = nullptr;
    for (const auto& s : subpools_) {
        if (s.pool == pool && s.group == gidx) { sp = &s; break; }
    }
    if (!sp) return -1;
    const uint32_t lo = sp->slot_begin, hi = sp->slot_begin + sp->n_slots;

    // 1. free slot (within this group's sub-pool)
    for (uint32_t i = lo; i < hi; ++i) {
        if (slot_word_state(slots_[i].load()) == SLOT_EMPTY) {
            slots_[i].begin_reload();
            owner_[i] = {layer, expert};
            return static_cast<int32_t>(i);
        }
    }

    // 2. evict victim: READY && refcount==0, lowest hybrid score then oldest.
    //    score = alpha * 1/(1+recency) + (1-alpha) * freq   (smaller = evict first)
    //    recency = max(0, cur_token - last_used_token); alpha = clamp(1-hit_rate_ema, .1, .9)
    //    Freshly-loaded, never-pinned slots are protected (evicting one would
    //    undo an in-flight load) UNLESS the pool has no other victim - a pool
    //    full of protected slots must still make room or the scheduler stalls.
    int32_t victim = -1;
    double  best_score = std::numeric_limits<double>::infinity();
    const uint64_t cur = token_.load(std::memory_order_relaxed);
    const double a = alpha_.load(std::memory_order_relaxed);
    for (int pass = 0; pass < 2 && victim < 0; ++pass) {
        const bool allow_fresh = (pass == 1);
        for (uint32_t i = lo; i < hi; ++i) {
            uint64_t w = slots_[i].load();
            if (slot_word_state(w) != SLOT_READY || slot_word_refcount(w) != 0) continue;
            const uint64_t last = dir_->last_used(owner_[i].first, owner_[i].second);
            if (!allow_fresh && last == 0) continue;
            const uint64_t recency = (cur > last) ? (cur - last) : 0;
        const double freq = stats_.get_adaptive_frequency(owner_[i].first, owner_[i].second);
        const double score = a * (1.0 / (1.0 + static_cast<double>(recency))) + (1.0 - a) * freq;
        if (score < best_score) {
            best_score = score;
            victim = static_cast<int32_t>(i);
        }
        }
    }
    if (victim < 0) return -1; // all slots pinned/in-flight

    // 3. evict: READY -> EVICTING, block new pins, drain refcount.
    if (!slots_[victim].begin_evict()) return -1;
    const uint32_t v_layer = owner_[victim].first;
    const uint32_t v_expert = owner_[victim].second;
    dir_->clear(v_layer, v_expert, sp->pool);
    while (slot_word_refcount(slots_[victim].load()) != 0) {
        std::this_thread::yield();
    }

    // 4. Demote, don't drop, device-region victims: move the content into the
    //    group's CPU-RAM region (which itself drops its coldest expert to make
    //    room). CPU-RAM victims are dropped (the disk is the next level). The
    //    content copy happens after the refcount drain, so no reader is pinned
    //    on the source slot; the destination slot is IO_INFLIGHT (reserved by
    //    alloc_or_evict above) and gets mark_ready() without any DIO.
    if (sp->pool != 0) {
        const subpool_t* rsp = ram_subpool(sp->group);
        if (rsp && rsp->expert_size == sp->expert_size && rsp->cols.size() == sp->cols.size()) {
            int32_t dst = alloc_or_evict(v_layer, v_expert, 0);   // RAM: make room
            if (dst >= 0) {
                // SoA: copy per column (device slot slices -> RAM slot slices)
                for (size_t c = 0; c < sp->cols.size(); ++c) {
                    std::memcpy(slot_col_mem(dst, static_cast<uint32_t>(c)),
                                slot_col_mem(victim, static_cast<uint32_t>(c)),
                                sp->cols[c].stride);
                }
                slots_[dst].mark_ready();
                dir_->set(v_layer, v_expert, 0, static_cast<uint32_t>(dst));
                LOG_DEBUG("expert_scheduler: demoted L" << v_layer << " E" << v_expert
                          << " device slot " << victim << " -> RAM slot " << dst);
            }
            // RAM full/pinned: fall through to dropping the device copy (the
            // content stays readable from disk).
        }
    }

    slots_[victim].begin_reload();
    owner_[victim] = {layer, expert};
    return victim;
}

const expert_scheduler::subpool_t* expert_scheduler::subpool_of_slot(int32_t slot) const {
    if (slot < 0 || slot >= static_cast<int32_t>(num_slots_)) return nullptr;
    for (const auto& sp : subpools_) {
        if (slot >= static_cast<int32_t>(sp.slot_begin) && slot < static_cast<int32_t>(sp.slot_begin + sp.n_slots)) {
            return &sp;
        }
    }
    return nullptr;
}

async_load_t* expert_scheduler::start_async_load(int32_t slot, uint32_t layer, uint32_t expert) {
    if (load_free_.empty()) return nullptr;   // pool full: keep draining completions
    const uint32_t idx = load_free_.back();
    load_free_.pop_back();
    load_in_use_[idx] = 1;
    async_load_t* t = load_task(idx);
    t->layer = layer;
    t->expert = expert;
    t->slot = static_cast<uint32_t>(slot);
    const subpool_t* osp = subpool_of_slot(slot);
    t->pool = osp ? osp->pool : 0;   // directory entry lands on the owning pool
    t->sched = this;   // shared-dio completions are dispatched by owner
    const expert_info_t& info = topo_->get_expert(layer, expert);
    t->plan = info.read_plan;
    t->failed = false;
    t->direct = false;   // per-slice direct now; staging when any slice is not direct
    const size_t header_sz = align_ceil(sizeof(async_load_t), DIO_SECTOR_SIZE);
    t->staging = load_pool_ + static_cast<size_t>(idx) * load_stride_ + header_sz;
    t->pending = 0;
    t->req_tsc = tsc_now();   // [TMR] request time: submission begins
    for (uint32_t s = 0; s < t->plan.num_tensors; ++s) {
        const auto& sl = t->plan.slices[s];
        aio_req_t& r = t->reqs[s];
        r = aio_req_t{};
        r.file = files_[sl.shard_idx];
        r.file_offset = sl.file_read_start;
        if (sl.direct) {
            // source aligned + len 4K multiple: DIO straight into the column slot
            const uint8_t* dst = slot_col_mem(slot, sl.column);
            if (!dst) { t->failed = true; break; }
            r.aligned_buf = const_cast<uint8_t*>(dst) + sl.copy_dst_offset;
            t->direct = true;
        } else {
            r.aligned_buf = t->staging + sl.staging_offset;
        }
        r.aligned_len = sl.file_read_len;
        r.user_data = t;
        if (dio_->submit_batch(&r, 1) > 0) {
            ++t->pending;
        } else {
            t->failed = true;
            break;
        }
    }
    if (t->pending == 0) {
        load_free_.push_back(idx);
        load_in_use_[idx] = 0;
        return nullptr;
    }
    return t;
}

void expert_scheduler::drain_completions(aio_req_t** done, uint32_t n) {
    // IOCP has no batch completion - we aggregate per async_load_t via `pending`.
    for (uint32_t i = 0; i < n; ++i) {
        async_load_t* t = static_cast<async_load_t*>(done[i]->user_data);
        const int slice = static_cast<int>(done[i] - t->reqs);
        if (done[i]->error_code != 0) {
            t->failed = true;
        } else if (!t->failed && !t->plan.slices[slice].direct) {
            const auto& sl = t->plan.slices[slice];
            const uint8_t* dst = slot_col_mem(static_cast<int32_t>(t->slot), sl.column);
            if (dst) {
                std::memcpy(const_cast<uint8_t*>(dst) + sl.copy_dst_offset,
                            t->staging + sl.copy_src_offset, sl.copy_byte_len);
            }
        }
        --t->pending;
        if (t->pending == 0) {
            t->dio_tsc = tsc_now();   // [TMR] all sub-tensor DIO reads complete
            if (!t->failed) {
                slots_[t->slot].mark_ready();
                dir_->set(t->layer, t->expert, t->pool, t->slot);
                n_misses_.fetch_add(1, std::memory_order_relaxed);
            } else {
                slots_[t->slot].mark_failed();
                LOG_ERROR("expert_scheduler: async DIO load failed for L" << t->layer << " E" << t->expert);
            }
            // wake-once: every settle (success OR failure) advances the batch
            // counter so exec never spins forever; exec rescans and pins what
            // is actually READY, retrying failures.
            if (t->batch_ready) {
                std::atomic<uint32_t>* c = t->batch_ready;
                c->fetch_add(1, std::memory_order_acq_rel);
                slot_wake_all(c);
            }
            t->done_tsc = tsc_now();   // [TMR] slot settled (ready or failed)
            const size_t off = static_cast<size_t>(reinterpret_cast<uint8_t*>(t) - load_pool_);
            const uint32_t idx = static_cast<uint32_t>(off / load_stride_);
            load_free_.push_back(idx);
            load_in_use_[idx] = 0;
        }
    }
}

bool expert_scheduler::accept_requests() {
    bool any = false;
    slot_request_t req;
    while (requests_.pop(req)) {
        any = true;
        // One request = one whole layer's missing expert set (bitmap). Each set
        // bit counts once toward the batch's completion word: already-resident
        // bits (raced) bump immediately, others bump when their async load
        // settles (drain_completions). Placement: device region first (keeps the
        // active set on one region while the executor has no per-region split
        // yet), RAM fallback.
        if (load_free_.empty()) { requests_.push(req); break; }
        const uint32_t n_experts = topo_->n_expert;
        uint64_t leftover[BITMAP_WORDS] = { 0 };
        uint32_t n_left = 0;
        auto bump = [&]() {   // one processed bit -> one toward the batch target
            if (req.batch_ready) {
                req.batch_ready->fetch_add(1, std::memory_order_acq_rel);
                slot_wake_all(req.batch_ready);
            }
        };
        for (uint32_t e = 0; e < n_experts; ++e) {
            if (!bit_test(req.needed, e)) continue;
            if (load_free_.empty()) { bit_set(leftover, e); ++n_left; continue; }
            uint32_t pool = 0;
            if (dir_->scan(req.layer, e, &pool) != SLOT_UNASSIGNED) {
                bump();   // raced with another submit: resident already
                continue;
            }
            uint32_t gidx = group_of(req.layer);
            int32_t slot = -1;
            const subpool_t* dsp = (gidx == static_cast<uint32_t>(-1)) ? nullptr : vram_subpool(gidx);
            if (dsp) slot = alloc_or_evict(req.layer, e, dsp->pool);
            if (slot < 0) slot = alloc_or_evict(req.layer, e, 0);
            if (slot < 0) { bit_set(leftover, e); ++n_left; continue; }
            async_load_t* t = start_async_load(slot, req.layer, e);
            if (!t) { bit_set(leftover, e); ++n_left; continue; }
            t->batch_ready = req.batch_ready;   // drain bumps when this settles
        }
        // requeue whatever could not be placed this pass (ring full / no victim)
        if (n_left) {
            slot_request_t rq = {};
            rq.layer = req.layer;
            rq.seq = req.seq;                    // keep the SAME completion target
            rq.batch_ready = req.batch_ready;
            for (uint32_t w = 0; w < BITMAP_WORDS; ++w) rq.needed[w] = leftover[w];
            requests_.push(rq);
            break;   // backpressure: don't starve other queues in this tick
        }
    }
    return any;
}

void expert_scheduler::update_alpha() {
    // alpha = clamp(1 - hit_rate_ema, 0.1, 0.9) from overall hit rate EMA.
    static thread_local uint64_t last_lookups = 0, last_hits = 0;
    const uint64_t lookups = n_lookups_.load(std::memory_order_relaxed);
    const uint64_t hits = n_hits_.load(std::memory_order_relaxed);
    if (lookups != last_lookups && lookups > 0) {
        last_lookups = lookups;
        last_hits = hits;
        const double hr = static_cast<double>(hits) / static_cast<double>(lookups);
        hit_rate_ema_ = hit_rate_ema_ * 0.9 + hr * 0.1;
        double a = 1.0 - hit_rate_ema_;
        if (a < 0.1) a = 0.1;
        if (a > 0.9) a = 0.9;
        alpha_.store(a, std::memory_order_relaxed);
    }
}

int32_t expert_scheduler::pin_layer(uint32_t layer, const uint64_t* needed, batch_await_t& await,
                                    expert_handle_t* out, uint32_t out_cap) {
    if (!topo_ || layer >= topo_->n_layer) return -1;
    const uint32_t n_experts = topo_->n_expert;
    const uint32_t want = bit_count(needed, n_experts);
    if (want == 0) return 0;
    if (want > out_cap) {
        LOG_ERROR("expert_scheduler: pin_layer out_cap " << out_cap << " < needed " << want);
        return -1;
    }

    // Two-pass: split the set into already-resident (pin now) and missing
    // (one batch request). Because the directory is read-only for compute and
    // only the scheduler mutates residency, do the split here by scanning.
    uint64_t missing[BITMAP_WORDS] = { 0 };
    uint32_t n_missing = 0, n_hit = 0;
    uint32_t o = 0;
    for (uint32_t e = 0; e < n_experts; ++e) {
        if (!bit_test(needed, e)) continue;
        uint32_t pool = 0;
        uint32_t s = dir_->scan(layer, e, &pool);
        if (s == SLOT_UNASSIGNED) {
            bit_set(missing, e);
            ++n_missing;
            continue;
        }
        if (slot_word_state(slots_[s].load()) != SLOT_READY) {
            // in-flight/evicting: not pin-able yet; request it too (idempotent -
            // the scheduler skips already-resident bits).
            bit_set(missing, e);
            ++n_missing;
            continue;
        }
        int64_t gen = slots_[s].try_pin();
        if (gen >= 0) {
            out[o++] = { static_cast<int32_t>(s), static_cast<uint32_t>(gen), layer, e, pool, true };
            ++n_hit;
            const uint64_t t = token_.fetch_add(1, std::memory_order_relaxed) + 1;
            dir_->touch_last_used(layer, e, t);
        } else {
            bit_set(missing, e);   // transient (another pin racing); request it
            ++n_missing;
        }
    }

    if (n_missing == 0) {
        n_lookups_.fetch_add(want, std::memory_order_relaxed);
        n_hits_.fetch_add(n_hit, std::memory_order_relaxed);
        return static_cast<int32_t>(want);
    }

    // Submit ONE batch request for the missing subset. `await` counts down per
    // completed expert; exec sleeps once until seq (== n_missing) is reached.
    slot_request_t req;
    req.layer = layer;
    req.seq = n_missing;
    for (uint32_t w = 0; w < BITMAP_WORDS; ++w) req.needed[w] = missing[w];
    await.reset();
    await.target = n_missing;
    req.batch_ready = &await.done;

    // Wait for the batch to settle (wake-once). Re-scan on completion: some
    // bits may have been raced/resident meanwhile; pin whatever is ready.
    n_lookups_.fetch_add(want, std::memory_order_relaxed);
    n_hits_.fetch_add(n_hit, std::memory_order_relaxed);

    for (uint32_t round = 0; round < 2; ++round) {
        requests_.push(req);                 // MPSC batch submit (blocking if full)
        await.wait();                        // single wake: done == target
        // pin the bits that are now resident
        uint32_t still_missing = 0;
        for (uint32_t e = 0; e < n_experts; ++e) {
            if (!bit_test(missing, e)) continue;
            uint32_t pool = 0;
            uint32_t s = dir_->scan(layer, e, &pool);
            if (s == SLOT_UNASSIGNED || slot_word_state(slots_[s].load()) != SLOT_READY) {
                ++still_missing;
                continue;
            }
            int64_t gen = slots_[s].try_pin();
            if (gen >= 0) {
                out[o++] = { static_cast<int32_t>(s), static_cast<uint32_t>(gen), layer, e, pool, true };
                const uint64_t t = token_.fetch_add(1, std::memory_order_relaxed) + 1;
                dir_->touch_last_used(layer, e, t);
                bit_clear(missing, e);      // pinned
            } else {
                ++still_missing;
            }
        }
        if (still_missing == 0) return static_cast<int32_t>(want);
        if (round == 0) {
            // transient failure / extra load needed: rebuild a fresh request and
            // retry once. Guarded against infinite spin.
            req.seq = still_missing;
            for (uint32_t w = 0; w < BITMAP_WORDS; ++w) req.needed[w] = 0;
            for (uint32_t e = 0; e < n_experts; ++e) if (bit_test(missing, e)) bit_set(req.needed, e);
            await.reset();
            await.target = still_missing;
            req.batch_ready = &await.done;
        } else {
            LOG_ERROR("expert_scheduler: pin_layer retry limit for L" << layer
                      << " (still missing " << still_missing << ")");
            return -1;
        }
    }
    return -1;
}

void expert_scheduler::unpin(const expert_handle_t& h) {
    if (h.slot >= 0 && h.slot < static_cast<int32_t>(num_slots_)) {
        slots_[h.slot].unpin();
    }
}

// ---- global scheduler thread (multi-level cache controller) ----------------

global_expert_scheduler& global_expert_scheduler::instance() {
    static global_expert_scheduler g;
    return g;
}

global_expert_scheduler::~global_expert_scheduler() {
    { std::lock_guard<std::mutex> lk(mtx_); running_ = false; }
    if (worker_.joinable()) worker_.join();
}

void global_expert_scheduler::register_scheduler(expert_scheduler* s) {
    std::lock_guard<std::mutex> lk(mtx_);
    schedulers_.push_back(s);
    // One worker for the process lifetime: started on first registration and
    // never torn down per-pool (a pool unregistering while others remain, or
    // re-registering later, must not restart/race the thread).
    if (!worker_.joinable()) {
        running_ = true;
        worker_ = std::thread(&global_expert_scheduler::worker_loop, this);
    }
}

void global_expert_scheduler::unregister_scheduler(expert_scheduler* s) {
    std::lock_guard<std::mutex> lk(mtx_);
    schedulers_.erase(std::remove(schedulers_.begin(), schedulers_.end(), s), schedulers_.end());
    // worker keeps running (idle when no pools are registered) until process exit.
}

void global_expert_scheduler::worker_loop() {
    for (;;) {
        std::vector<expert_scheduler*> snap;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!running_) break;
            snap = schedulers_;
        }
        bool any = false;
        // 1. drain each pool's DIO completions (shared engine in production -
        //    a completion may belong to another pool; dispatch by owner).
        for (auto* s : snap) {
            if (!s->is_running()) continue;
            async_dio_engine* dio = s->dio_engine();
            if (!dio) continue;
            aio_req_t* done[64];
            const uint32_t n = dio->wait_events(done, 64, 0, 0);
            if (n) any = true;
            for (uint32_t i = 0; i < n; ++i) {
                async_load_t* t = static_cast<async_load_t*>(done[i]->user_data);
                if (t->sched && t->sched->is_running()) {
                    t->sched->drain_completions(&done[i], 1);
                }
            }
        }
        // 2. accept alloc requests from every registered pool.
        for (auto* s : snap) {
            if (s->is_running() && s->accept_requests()) any = true;
        }
        // 3. adapt eviction alpha (overall hit-rate EMA, scheduler-thread owned).
        for (auto* s : snap) {
            if (s->is_running()) s->update_alpha();
        }
        if (!any) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

} // namespace stream_moe

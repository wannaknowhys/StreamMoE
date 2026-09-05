#include "backend/scheduler.h"
#include "common/logger.h"
#include "common/tsc.h"
#include "common/types.h"

#include <cstring>
#include <algorithm>
#include <chrono>

// Transfer-queue DMA download of a VRAM buffer slice to a host pointer
// (defined in the route-B vk_dma frag, vendored ggml-vulkan.cpp). 0.02 GB/s
// rebar host reads vs ~14 GB/s via the device transfer queue.
void stmoe_vk_dma_read(void* vram_buffer, size_t off, void* dst, size_t bytes);

// Per-expert / per-event diagnostics (move, load, alloc, evict, pin ticks) are
// temporary debug output: compiled only under STREAM_MOE_TEMP (the
// StreamMoE_dump_dbg tag). Without the macro the argument expression is not
// even evaluated (no ostringstream cost). ring-full (:691) intentionally stays
// a plain LOG_DEBUG - it is a real retry traceback worth having in any build.
#ifdef STREAM_MOE_TEMP
#define SCHED_DIAG(msg) LOG_DEBUG(msg)
#else
#define SCHED_DIAG(msg) do { } while (0)
#endif

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

// ---- async move pipeline (M4) ---------------------------------------------

bool expert_scheduler::submit_move(uint32_t layer, uint32_t expert,
                                   uint32_t src_pool, uint32_t src_slot,
                                   uint32_t dst_pool, uint32_t dst_slot) {
    const subpool_t* ssp = subpool_of_slot(static_cast<int32_t>(src_slot));
    const subpool_t* dsp = subpool_of_slot(static_cast<int32_t>(dst_slot));
    if (!ssp || !dsp) return false;
    // both pools must belong to the same group (same column set / strides)
    if (ssp->group != dsp->group || ssp->cols.size() != dsp->cols.size()) return false;

    move_task_t t;
    t.layer = layer; t.expert = expert;
    t.src_pool = src_pool; t.src_slot = src_slot;
    t.dst_pool = dst_pool; t.dst_slot = dst_slot;
    t.owner = this;
    t.req_tsc = tsc_now();
    for (size_t c = 0; c < ssp->cols.size(); ++c) {
        const column_t& sc = ssp->cols[c];
        const column_t& dc = dsp->cols[c];
        if (sc.stride != dc.stride) return false;   // must be same layout
        move_task_t::copy_t cp;
        cp.src = col_slice_ptr(*ssp, sc, static_cast<int32_t>(src_slot));
        cp.dst = col_slice_ptr(*dsp, dc, static_cast<int32_t>(dst_slot));
        cp.bytes = sc.stride;
        if (!cp.src || !cp.dst) return false;
        // v2r: DMA the device source instead of reading the slow rebar host map.
        if (ssp->pool != 0 && ssp->dev_buf) {
            cp.dev_buf = ssp->dev_buf;
            cp.dev_off = static_cast<size_t>(cp.src - ssp->base);
        }
        t.cols.push_back(cp);
    }
    {
        std::lock_guard<std::mutex> lk(move_mtx_);
        move_submit_.push_back(std::move(t));
    }
    move_cv_.notify_one();
    return true;
}

void expert_scheduler::move_worker_main() {
    for (;;) {
        move_task_t t;
        {
            std::unique_lock<std::mutex> lk(move_mtx_);
            move_cv_.wait(lk, [&] { return move_stop_.load(std::memory_order_relaxed) || !move_submit_.empty(); });
            if (move_submit_.empty()) {
                if (move_stop_.load(std::memory_order_relaxed)) break;
                continue;
            }
            t = std::move(move_submit_.front());
            move_submit_.pop_front();
        }
        // per-column copy. Device (vram) sources go through the transfer-queue
        // DMA download (stmoe_vk_dma_read); RAM sources are a plain memcpy.
        // Never touches the scheduler control plane.
        const uint64_t cp0 = tsc_now();
        uint64_t dma_ns = 0, mc_ns = 0;
        for (const auto& cp : t.cols) {
            if (cp.dev_buf) {
                const uint64_t d0 = tsc_now();
                stmoe_vk_dma_read(cp.dev_buf, cp.dev_off, cp.dst, cp.bytes);
                dma_ns += tsc_now() - d0;
            } else {
                const uint64_t d0 = tsc_now();
                std::memcpy(cp.dst, cp.src, cp.bytes);
                mc_ns += tsc_now() - d0;
            }
        }
        t.done_tsc = tsc_now();
        SCHED_DIAG("sched: move L" << t.layer << " E" << t.expert
                  << (dma_ns ? " dma=" + std::to_string(tsc_delta_ns(dma_ns) / 1000) + "us" : "")
                  << " mc=" << (tsc_delta_ns(mc_ns) / 1000) << "us"
                  << " (queued=" << (tsc_delta_ns(t.done_tsc - t.req_tsc) / 1000) << "us)");
        {
            std::lock_guard<std::mutex> lk(move_mtx_);
            move_done_.push_back(std::move(t));
        }
        // wake the global worker via the shared DIO engine? No - drain_moves is
        // polled by worker_loop each tick; no explicit wake needed (1ms poll).
    }
}

void expert_scheduler::drain_moves() {
    // Control-plane tail for finished moves (global scheduler thread).
    for (;;) {
        move_task_t t;
        {
            std::lock_guard<std::mutex> lk(move_mtx_);
            if (move_done_.empty()) return;
            t = std::move(move_done_.front());
            move_done_.pop_front();
        }
        // dst slot: IO_INFLIGHT (reserved at submit) -> READY; dir entry lands.
        slots_[t.dst_slot].mark_ready();
        dir_->transition(t.layer, t.expert, t.dst_pool, EXPERT_READY, t.dst_slot);
        // src slot: content copied away (was EVICTING / exclusive during the
        // move). Release it to EMPTY so a requeued alloc can reuse it. If the
        // move was a r2v the src was a RAM copy that is now redundant - releasing
        // is also correct (RAM keeps the dst or drops; content is on disk).
        slots_[t.src_slot].release_to_empty();
        SCHED_DIAG("expert_scheduler: moved L" << t.layer << " E" << t.expert
                  << " pool " << t.src_pool << " slot " << t.src_slot
                  << " -> pool " << t.dst_pool << " slot " << t.dst_slot);
    }
}

void expert_scheduler::start() {
    if (running_.exchange(true)) return;
    // launch the move worker once (idle until submit_move is called)
    if (!move_thread_.joinable()) {
        move_stop_.store(false, std::memory_order_relaxed);
        move_thread_ = std::thread(&expert_scheduler::move_worker_main, this);
    }
    global_expert_scheduler::instance().register_scheduler(this);
}

void expert_scheduler::stop() {
    if (!running_.exchange(false)) return;
    global_expert_scheduler::instance().unregister_scheduler(this);
    // drain any in-flight moves before stopping the worker
    {
        std::unique_lock<std::mutex> lk(move_mtx_);
        while (!move_done_.empty()) {
            move_task_t t = std::move(move_done_.front());
            move_done_.pop_front();
            lk.unlock();
            slots_[t.dst_slot].mark_ready();
            dir_->transition(t.layer, t.expert, t.dst_pool, EXPERT_READY, t.dst_slot);
            lk.lock();
        }
    }
    move_stop_.store(true, std::memory_order_relaxed);
    move_cv_.notify_all();
    if (move_thread_.joinable()) {
        move_thread_.join();
        move_thread_ = std::thread();   // reset so start() can relaunch
    }
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
            // Ordering invariant (§3.4): publish LOADING (intent + slot) BEFORE
            // the physical slot goes IO_INFLIGHT, so a compute scan can never see
            // "slot busy but dir ABSENT" (which would double-submit the expert).
            dir_->transition(layer, expert, pool, EXPERT_LOADING, static_cast<uint32_t>(i));
            slots_[i].begin_reload();
            return static_cast<int32_t>(i);
        }
    }
    SCHED_DIAG("sched: alloc L" << layer << " E" << expert << " pool " << pool
              << " no free slot in [" << lo << "," << hi << ") - must evict");

    // 2. Evict victim, keyed on (L,E) with layer-distance preference (M3, design
    //    §6). We need a free slot for (layer, expert) in this pool; choose a
    //    resident expert to evict. Enumerate layers nearest to `layer` first
    //    (the just-finished layers are most likely stale in a decode), scan that
    //    layer's experts for a READY slot in THIS pool, pick the lowest hybrid
    //    score in the nearest layer that has any evictable candidate. owner_ is
    //    gone - the (L,E) comes from the traversal, the slot from dir.find.
    //    score = alpha * 1/(1+recency) + (1-alpha) * freq (smaller evicts first)
    //    recency = max(0, cur_token - last_used_token).
    int32_t victim = -1;
    uint32_t v_layer = 0, v_expert = 0;
    const uint64_t cur = token_.load(std::memory_order_relaxed);
    const double a = alpha_.load(std::memory_order_relaxed);
    // Freshly-loaded, never-pinned experts are protected (evicting one would undo
    // an in-flight load) UNLESS nothing else is evictable - a pool full of
    // protected slots must still make room or the scheduler stalls.
    for (int pass = 0; pass < 2 && victim < 0; ++pass) {
        const bool allow_fresh = (pass == 1);
        // Scan layers nearest to `layer` first: L-1, L-2, ... down to 0. The
        // model runs all layers per token, so a just-finished layer is stale
        // for many tokens while a not-yet-run layer (L+1) is about to be used -
        // do not prefer evicting it. Delta bounded by `layer` (no negative L).
        for (uint32_t delta = 1; delta <= layer; ++delta) {
            const uint32_t L_u = layer - delta;
            double best_in_layer = std::numeric_limits<double>::infinity();
            int32_t cand = -1;
            uint32_t cand_e = 0;
            for (uint32_t e = 0; e < topo_->n_expert; ++e) {
                const uint32_t s = dir_->find(L_u, e, pool);   // READY in this pool only
                if (s == SLOT_UNASSIGNED) continue;
                uint64_t w = slots_[s].load();
                if (slot_word_state(w) != SLOT_READY || slot_word_refcount(w) != 0) continue;
                const uint64_t last = dir_->last_used(L_u, e);
                if (!allow_fresh && last == 0) continue;
                const uint64_t recency = (cur > last) ? (cur - last) : 0;
                const double freq = stats_.get_adaptive_frequency(L_u, e);
                const double score = a * (1.0 / (1.0 + static_cast<double>(recency))) + (1.0 - a) * freq;
                if (score < best_in_layer) {
                    best_in_layer = score;
                    cand = static_cast<int32_t>(s);
                    cand_e = e;
                }
            }
            if (cand >= 0) { victim = cand; v_layer = L_u; v_expert = cand_e; break; }
            // no evictable expert in this layer; widen to the next-older layer.
        }
    }
    if (victim < 0) {
        SCHED_DIAG("sched: alloc L" << layer << " E" << expert << " pool " << pool
                  << " NO evictable victim (all pinned/fresh)");
        return -1; // all slots pinned/in-flight
    }
    SCHED_DIAG("sched: evict L" << v_layer << " E" << v_expert << " (slot " << victim
              << ") to make room for L" << layer << " E" << expert << " in pool " << pool);

    // 3. evict: READY -> EVICTING, block new pins, drain refcount.
    if (!slots_[victim].begin_evict()) return -1;
    dir_->clear(v_layer, v_expert, sp->pool);
    while (slot_word_refcount(slots_[victim].load()) != 0) {
        std::this_thread::yield();
    }

    // 4. Demote device-region victims asynchronously (M4): the victim's content
    //    is moved to the group's RAM region by the move worker; CPU-RAM victims
    //    are dropped (disk is the next level). The src (victim) slot cannot be
    //    reused for the new expert until the copy completes - so we submit the
    //    move and return -1 (the caller requeues); drain_moves releases the src
    //    slot to EMPTY once the copy is done, and the requeued expert picks it
    //    up on the next alloc pass.
    if (sp->pool != 0) {
        const subpool_t* rsp = ram_subpool(sp->group);
        if (rsp && rsp->expert_size == sp->expert_size && rsp->cols.size() == sp->cols.size()) {
            // reserve a RAM destination slot (this may itself evict a RAM expert)
            const int32_t dst = alloc_or_evict(v_layer, v_expert, 0);
            if (dst >= 0) {
                if (submit_move(v_layer, v_expert, sp->pool, static_cast<uint32_t>(victim), 0, static_cast<uint32_t>(dst))) {
                    // dst slot is IO_INFLIGHT; dir entry for v in RAM is LOADING
                    // (from alloc_or_evict) - flip to MOVING_IN for the async copy.
                    dir_->transition(v_layer, v_expert, 0, EXPERT_MOVING_IN, static_cast<uint32_t>(dst));
                    SCHED_DIAG("expert_scheduler: async demote L" << v_layer << " E" << v_expert
                              << " device slot " << victim << " -> RAM slot " << dst);
                    return -1;   // src slot stays EVICTING until drain_moves frees it
                }
                // submit failed (should not happen): drop the device copy and
                // release the reserved RAM dst slot.
                dir_->transition(v_layer, v_expert, 0, EXPERT_ABSENT, SLOT_UNASSIGNED);
                slots_[dst].release_to_empty();
            }
            // RAM full/pinned: fall through to dropping the device copy (the
            // content stays readable from disk).
        }
    }

    slots_[victim].begin_reload();
    // Ordering invariant (§3.4): the victim slot is now being (re)loaded for the
    // NEW (layer, expert). Publish LOADING before/with begin_reload (begin_reload
    // above already ran for the free-slot path; here publish intent now).
    dir_->transition(layer, expert, sp->pool, EXPERT_LOADING, static_cast<uint32_t>(victim));
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
    t->staging_mask = 0;   // reused task: clear per-slice staging bits from the prior load
    t->req_tsc = tsc_now();   // [TMR] request time: submission begins
    for (uint32_t s = 0; s < t->plan.num_tensors; ++s) {
        const auto& sl = t->plan.slices[s];
        aio_req_t& r = t->reqs[s];
        r = aio_req_t{};
        r.file = files_[sl.shard_idx];
        r.file_offset = sl.file_read_start;
        // Direct DIO into the column slot needs a 4K-aligned DESTINATION, not
        // just an aligned source/length. The plan's `direct` only checks the
        // file side; the slot address (column base + slot index * stride) can
        // drift out of alignment when a preceding column's per-expert stride is
        // not 4K and the region slot count moves the column start (e.g. a
        // device region with 69 slots puts the down column at mod 4096 = 2048).
        // Fall back to the staging path when the target is misaligned (every
        // slice keeps its staging bytes reserved either way).
        bool use_direct = sl.direct;
        if (use_direct) {
            const uint8_t* dst = slot_col_mem(slot, sl.column);
            if (!dst) { t->failed = true; break; }
            if (!is_aligned(dst + sl.copy_dst_offset)) use_direct = false;
        }
        if (use_direct) {
            const uint8_t* dst = slot_col_mem(slot, sl.column);
            r.aligned_buf = const_cast<uint8_t*>(dst) + sl.copy_dst_offset;
            t->direct = true;
        } else {
            r.aligned_buf = t->staging + sl.staging_offset;
            t->staging_mask |= static_cast<uint8_t>(1u << s);
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
        } else if (!t->failed && (t->staging_mask & (uint8_t)(1u << slice))) {
            // plan.direct but the runtime destination was not 4K aligned: the
            // read went to staging, so copy payload to the column slot now.
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
                SCHED_DIAG("sched: loaded L" << t->layer << " E" << t->expert
                          << " -> pool " << t->pool << " slot " << t->slot
                          << " (staging_mask=" << (unsigned)t->staging_mask << ")"
                          << " dio=" << (t->dio_tsc > t->req_tsc ? tsc_delta_ns(t->dio_tsc - t->req_tsc) / 1000 : 0) << "us");
            } else {
                slots_[t->slot].mark_failed();
                // Revert LOADING -> ABSENT so a later retry can reload the
                // expert (an ABSENT dir entry is what triggers a new load). A
                // FAILED dir state would make accept skip it forever.
                dir_->transition(t->layer, t->expert, t->pool, EXPERT_ABSENT, SLOT_UNASSIGNED);
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
        SCHED_DIAG("sched: accept req L" << req.layer << " needed-bits=" << bit_count(req.needed, topo_->n_expert)
                  << " target=" << req.n_load_target << " load_free=" << load_free_.size());
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
            // Branch on the expert's directory state across pools (M2):
            //   READY            -> already resident (raced): bump, nothing to load
            //   LOADING/MOVING_* -> in flight from a prior tick's alloc: do NOT
            //                        start a second load; the in-flight settle
            //                        bumps through drain. Skipping without bump
            //                        here is correct (drain does it once).
            //   ABSENT / FAILED  -> genuinely needs a load: alloc + submit.
            uint32_t gidx = group_of(req.layer);
            if (gidx == static_cast<uint32_t>(-1)) { bit_set(leftover, e); ++n_left; continue; }
            bool in_flight = false, resident = false;
            for (uint32_t p = 0; p < n_pools_; ++p) {
                const expert_state st = dir_->state(req.layer, e, p);
                if (st == EXPERT_READY) resident = true;
                else if (st == EXPERT_LOADING || st == EXPERT_MOVING_IN || st == EXPERT_MOVING_OUT) in_flight = true;
            }
            if (resident) { bump(); continue; }
            if (in_flight) continue;   // drain will bump when it settles
            int32_t slot = -1;
            const subpool_t* dsp = (gidx == static_cast<uint32_t>(-1)) ? nullptr : vram_subpool(gidx);
            if (dsp) slot = alloc_or_evict(req.layer, e, dsp->pool);
            if (slot < 0) slot = alloc_or_evict(req.layer, e, 0);
            if (slot < 0) {
                SCHED_DIAG("sched: L" << req.layer << " E" << e << " no slot in vram or ram (leftover)");
                bit_set(leftover, e); ++n_left; continue;
            }
            async_load_t* t = start_async_load(slot, req.layer, e);
            if (!t) {
                // No async-load ring slot: revert LOADING -> ABSENT so a later
                // pass can retry (an in-flight dir entry with no load would be
                // skipped forever). The physical slot stays IO_INFLIGHT (the
                // ring-full case is transient; next alloc picks an EMPTY one).
                dir_->transition(req.layer, e, subpool_of_slot(slot) ? subpool_of_slot(slot)->pool : 0,
                                 EXPERT_ABSENT, SLOT_UNASSIGNED);
                LOG_DEBUG("sched: L" << req.layer << " E" << e << " ring-full (slot " << slot << "), reverted to ABSENT");
                bit_set(leftover, e); ++n_left; continue;
            }
            t->batch_ready = req.batch_ready;   // drain bumps when this settles
        }
        // requeue whatever could not be placed this pass (ring full / no victim)
        if (n_left) {
            slot_request_t rq = {};
            rq.layer = req.layer;
            rq.n_load_target = req.n_load_target;    // keep the SAME completion target
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
    // completed expert; exec sleeps once until n_load_target (== n_missing) is
    // reached.
    SCHED_DIAG("exec->sched: pin_layer L" << layer << " want=" << want
              << " hit=" << n_hit << " miss=" << n_missing);
    slot_request_t req;
    req.layer = layer;
    req.n_load_target = n_missing;
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
        SCHED_DIAG("exec<-sched: pin_layer L" << layer << " round " << round
                  << " pinned=" << (want - still_missing) << "/" << want
                  << " still_missing=" << still_missing);
        if (still_missing == 0) return static_cast<int32_t>(want);
        if (round == 0) {
            // transient failure / extra load needed: rebuild a fresh request and
            // retry once. Guarded against infinite spin.
            req.n_load_target = still_missing;
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
        // 2b. finish async cross-pool moves (dst READY + dir, src released).
        for (auto* s : snap) {
            if (s->is_running()) s->drain_moves();
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

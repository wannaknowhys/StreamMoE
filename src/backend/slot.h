#pragma once

// Route B control plane: slot_meta 64-bit atomic word + expert directory +
// bounded MPSC alloc-request queue. Cross-platform (Windows / Linux), model-agnostic:
// "expert" is just an opaque (layer, expert) pair; no DeepSeek-specific logic.
// Design source: docs/Backend.md §2/§4.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define _WIN32_WINNT 0x0603
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace stream_moe {

// ---- slot_meta: single 64-bit atomic word -------------------------------
// bit 63........32 | 31........3 | 2 1 0
//     generation   |  refcount   | state
enum slot_state : uint32_t {
    SLOT_EMPTY       = 0,
    SLOT_IO_INFLIGHT = 1,
    SLOT_READY       = 2,
    SLOT_EVICTING    = 3,
    SLOT_FAILED      = 4,
};

static constexpr uint64_t SLOT_STATE_MASK  = 0x7ull;
static constexpr uint64_t SLOT_REFCNT_MASK = 0xFFFFFFF8ull; // bits 3..31
static constexpr uint64_t SLOT_GEN_MASK    = 0xFFFFFFFF00000000ull;

inline uint32_t slot_word_state(uint64_t w)   { return static_cast<uint32_t>(w & SLOT_STATE_MASK); }
inline uint32_t slot_word_refcount(uint64_t w){ return static_cast<uint32_t>((w & SLOT_REFCNT_MASK) >> 3); }
inline uint32_t slot_word_generation(uint64_t w){ return static_cast<uint32_t>(w >> 32); }

inline uint64_t slot_word(uint32_t state, uint32_t refcount, uint32_t generation) {
    return (static_cast<uint64_t>(generation) << 32) |
           (static_cast<uint64_t>(refcount) << 3) |
           (static_cast<uint64_t>(state) & SLOT_STATE_MASK);
}

// Blocking wait/wake helpers on an arbitrary aligned word (uint32 or uint64).
// Windows: WaitOnAddress/WakeByAddressAll (Win8+). Linux: futex on the 32-bit
// word AT `addr` (for 64-bit words this is the low half on little-endian; callers
// re-check the full word after waking). Other: bounded spin+yield fallback.
inline void slot_wait_addr(const void* addr, const void* expected, size_t size) {
#if defined(_WIN32)
    WaitOnAddress(const_cast<void*>(addr), const_cast<void*>(expected), size, INFINITE);
#elif defined(__linux__)
    const int32_t* lo = static_cast<const int32_t*>(addr);
    int32_t expected_lo = *static_cast<const int32_t*>(expected);
    syscall(SYS_futex, const_cast<int32_t*>(lo), FUTEX_WAIT, expected_lo, nullptr, nullptr, 0);
#else
    (void)addr; (void)expected; (void)size;
    std::this_thread::yield();
#endif
}

inline void slot_wake_all(const void* addr) {
#if defined(_WIN32)
    WakeByAddressAll(const_cast<void*>(addr));
#elif defined(__linux__)
    const int32_t* lo = static_cast<const int32_t*>(addr);
    syscall(SYS_futex, const_cast<int32_t*>(lo), FUTEX_WAKE, INT32_MAX, nullptr, nullptr, 0);
#else
    (void)addr;
#endif
}

struct slot_meta {
    std::atomic<uint64_t> word{slot_word(SLOT_EMPTY, 0, 0)};

    uint64_t load() const { return word.load(std::memory_order_acquire); }

    // pin: require READY, refcount++, returns generation on success, -1 if not pin-able
    int64_t try_pin() {
        uint64_t w = word.load(std::memory_order_acquire);
        while (true) {
            if (slot_word_state(w) != SLOT_READY) return -1;
            uint32_t rc = slot_word_refcount(w);
            if (rc >= ((1u << 29) - 1)) return -1;
            uint64_t nw = slot_word(SLOT_READY, rc + 1, slot_word_generation(w));
            if (word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return static_cast<int64_t>(slot_word_generation(nw));
            }
        }
    }

    void unpin() {
        uint64_t w = word.load(std::memory_order_relaxed);
        while (true) {
            uint32_t rc = slot_word_refcount(w);
            if (rc == 0) return; // defensive: never below zero
            uint64_t nw = slot_word(slot_word_state(w), rc - 1, slot_word_generation(w));
            if (word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_relaxed)) return;
        }
    }

    // IO_INFLIGHT -> READY (payload arrived); wake all waiters
    void mark_ready() {
        uint64_t w = word.load(std::memory_order_relaxed);
        while (true) {
            uint32_t st = slot_word_state(w);
            if (st == SLOT_READY) break;
            if (st != SLOT_IO_INFLIGHT) return; // only IO_INFLIGHT may transition
            uint64_t nw = slot_word(SLOT_READY, slot_word_refcount(w), slot_word_generation(w));
            if (word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_relaxed)) break;
        }
        slot_wake_all(&word);
    }

    // READY -> EVICTING: stop new pins. Returns true if this thread won the CAS.
    bool begin_evict() {
        uint64_t w = word.load(std::memory_order_acquire);
        while (true) {
            if (slot_word_state(w) != SLOT_READY) return false;
            uint64_t nw = slot_word(SLOT_EVICTING, slot_word_refcount(w), slot_word_generation(w));
            if (word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
        }
    }

    // assign (layer, expert) for (re)loading: -> IO_INFLIGHT, generation++
    uint32_t begin_reload() {
        uint64_t w = word.load(std::memory_order_relaxed);
        while (true) {
            uint32_t gen = slot_word_generation(w) + 1;
            uint64_t nw = slot_word(SLOT_IO_INFLIGHT, 0, gen);
            if (word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return gen;
            }
        }
    }

    void mark_failed() {
        uint64_t w = word.load(std::memory_order_relaxed);
        while (true) {
            uint64_t nw = slot_word(SLOT_FAILED, slot_word_refcount(w), slot_word_generation(w));
            if (word.compare_exchange_weak(w, nw, std::memory_order_acq_rel, std::memory_order_relaxed)) break;
        }
        slot_wake_all(&word);
    }

    // wait until READY (and optionally generation matches) - used by compute threads
    bool wait_ready(uint32_t expect_gen) {
        uint64_t w = word.load(std::memory_order_acquire);
        while (true) {
            uint32_t st = slot_word_state(w);
            if (st == SLOT_READY && slot_word_generation(w) == expect_gen) return true;
            if (st == SLOT_FAILED) return false;
            slot_wait_addr(&word, &w, sizeof(w));
            w = word.load(std::memory_order_acquire);
        }
    }
};

// ---- expert directory -----------------------------------------------------
// Maps (layer, expert) -> slot index (per pool) or UNASSIGNED. Two-dimensional:
// outer (layer, expert), inner pool (CPU RAM / GPU VRAM ...). The whole expert
// store is viewed as a read-only multi-level cache: a directory is kept per
// (layer, expert) per pool, plus one cross-pool last_used_token (recency for
// eviction scoring). Compute threads read-only; scheduler thread writes.
// version[e] bumps on every mapping change (any pool) so waiters can wake and
// re-scan without a lock.
static constexpr uint32_t SLOT_UNASSIGNED = 0xFFFFFFFFu;

class expert_directory {
public:
    expert_directory(uint32_t n_layers, uint32_t n_experts, uint32_t n_pools, uint32_t n_slots)
        : n_layers_(n_layers), n_experts_(n_experts), n_pools_(n_pools), n_slots_(n_slots),
          entries_(static_cast<size_t>(n_layers) * n_experts * n_pools),
          versions_(static_cast<size_t>(n_layers) * n_experts),
          last_used_(static_cast<size_t>(n_layers) * n_experts) {
        for (auto& e : entries_) e.store(SLOT_UNASSIGNED, std::memory_order_relaxed);
    }

    uint32_t n_pools() const { return n_pools_; }

    // (layer, expert) -> slot in a specific pool, or SLOT_UNASSIGNED.
    uint32_t find(uint32_t layer, uint32_t expert, uint32_t pool) const {
        return entries_[idx(layer, expert, pool)].load(std::memory_order_acquire);
    }

    // Scan all pools for (layer, expert). Returns slot (global) and writes the
    // owning pool to *out_pool, or SLOT_UNASSIGNED if resident nowhere.
    uint32_t scan(uint32_t layer, uint32_t expert, uint32_t* out_pool) const {
        for (uint32_t p = 0; p < n_pools_; ++p) {
            uint32_t s = entries_[idx(layer, expert, p)].load(std::memory_order_acquire);
            if (s != SLOT_UNASSIGNED) {
                if (out_pool) *out_pool = p;
                return s;
            }
        }
        return SLOT_UNASSIGNED;
    }

    void set(uint32_t layer, uint32_t expert, uint32_t pool, uint32_t slot) {
        entries_[idx(layer, expert, pool)].store(slot, std::memory_order_release);
        versions_[idx(layer, expert)].fetch_add(1, std::memory_order_acq_rel);
        slot_wake_all(&versions_[idx(layer, expert)]);
    }

    void clear(uint32_t layer, uint32_t expert, uint32_t pool) {
        set(layer, expert, pool, SLOT_UNASSIGNED);
    }

    uint32_t version(uint32_t layer, uint32_t expert) const {
        return versions_[idx(layer, expert)].load(std::memory_order_acquire);
    }

    void wait_version(uint32_t layer, uint32_t expert, uint32_t expected) const {
        const auto& v = versions_[idx(layer, expert)];
        uint32_t cur = v.load(std::memory_order_acquire);
        while (cur == expected) {
            slot_wait_addr(&v, &cur, sizeof(cur));
            cur = v.load(std::memory_order_acquire);
        }
    }

    // Cross-pool recency: bumped with the global token counter on every
    // successful pin; eviction scoring reads it (max(0, cur - last_used)).
    void touch_last_used(uint32_t layer, uint32_t expert, uint64_t token) {
        last_used_[idx(layer, expert)].store(token, std::memory_order_relaxed);
    }
    uint64_t last_used(uint32_t layer, uint32_t expert) const {
        return last_used_[idx(layer, expert)].load(std::memory_order_relaxed);
    }

private:
    size_t idx(uint32_t layer, uint32_t expert) const {
        return static_cast<size_t>(layer) * n_experts_ + expert;
    }
    size_t idx(uint32_t layer, uint32_t expert, uint32_t pool) const {
        return (static_cast<size_t>(layer) * n_experts_ + expert) * n_pools_ + pool;
    }
    uint32_t n_layers_, n_experts_, n_pools_, n_slots_;
    std::vector<std::atomic<uint32_t>> entries_;
    std::vector<std::atomic<uint32_t>> versions_;
    std::vector<std::atomic<uint64_t>> last_used_;
};

// ---- bounded MPSC alloc-request queue -------------------------------------
// Compute threads push a WHOLE-LAYER request (bitmap of needed experts);
// scheduler thread pops. A request is a fixed POD:
//   header   layer u32 + total_tokens u32 + start_rdtsc u64   (16B)
//   wake     n_load_target (load count) u32 + batch_ready ptr u64      (16B)
//   bitmap   needed[8]                                        (64B)
//            == 96B total (the batch_ready pointer is the wake-once carrier;
//            without it exec would sleep per-expert version word).
// Producer sets n_load_target = number of experts it expects the scheduler to
// load and passes a pointer to its own counter; scheduler bumps + wakes it
// once per completed expert; exec sleeps once until count == n_load_target.
// Ring elements are PLAIN PODs guarded by a per-slot publish generation +
// release/acquire on head/tail - NOT std::atomic<T> (12B was already
// non-lock-free -> hidden lock; 96B would be worse). Multi-producer safe.
static constexpr uint32_t MAX_EXPERTS_PER_LAYER = 512;
static constexpr uint32_t BITMAP_WORDS = MAX_EXPERTS_PER_LAYER / 64;

struct slot_request_t {
    uint32_t layer        = 0;   // layer index this request covers
    uint32_t total_tokens = 0;   // [profile] batch tokens (ids->ne[1])
    uint64_t start_rdtsc  = 0;   // [profile] batch submit time (raw TSC)
    // n_load_target (2026-09): number of experts this request expects the
    // scheduler to load. Compute waits on *batch_ready until it reaches this.
    uint32_t n_load_target = 0;
    // wake-once counter (compute-side batch_await_t::done). A raw pointer keeps
    // slot_request_t trivially copyable for the plain ring.
    std::atomic<uint32_t>* batch_ready = nullptr;
    uint64_t needed[BITMAP_WORDS] = { 0 };         // bit e -> expert e needed
};
static_assert(sizeof(slot_request_t) == 96, "slot_request_t layout drifted");

// Convenience for the exec side: the wake-once counter + expected count.
struct batch_await_t {
    std::atomic<uint32_t> done{0};   // scheduler fetch_add per completed expert
    uint32_t target = 0;             // == slot_request_t::n_load_target
    // Wait until `done` reaches `target` (wake-once; scheduler wakes this word).
    void wait() {
        uint32_t cur = done.load(std::memory_order_acquire);
        while (cur < target) {
            slot_wait_addr(&done, &cur, sizeof(cur));
            cur = done.load(std::memory_order_acquire);
        }
    }
    void reset() { done.store(0, std::memory_order_relaxed); target = 0; }
};

class mpsc_alloc_queue {
public:
    explicit mpsc_alloc_queue(uint32_t capacity) : cap_(capacity), ring_(capacity) {
        seqs_.reset(new std::atomic<uint32_t>[capacity]);
        for (uint32_t i = 0; i < capacity; ++i) seqs_[i].store(0, std::memory_order_relaxed);
    }

    void push(slot_request_t r) {
        // reserve a slot by claiming the tail counter (MPSC: fetch_add gives
        // each producer a unique slot index and the reservation's generation)
        uint32_t tail;
        for (;;) {
            const uint32_t head = head_.load(std::memory_order_acquire);
            tail = tail_.load(std::memory_order_relaxed);
            if (tail - head >= cap_) {   // full
                std::this_thread::yield();
                continue;
            }
            uint32_t expected = tail;
            if (tail_.compare_exchange_weak(expected, tail + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
                break;   // claimed tail
            }
        }
        const uint32_t slot = tail % cap_;
        ring_[slot] = r;   // plain write payload
        // publish: the slot's generation = this reservation (head+1 base).
        // Consumers wait until the generation matches before reading (no ABA).
        seqs_[slot].store(tail + 1, std::memory_order_release);
    }

    bool pop(slot_request_t& out) {
        for (;;) {
            const uint32_t head = head_.load(std::memory_order_relaxed);
            const uint32_t tail = tail_.load(std::memory_order_acquire);
            if (head == tail) return false;   // empty
            const uint32_t slot = head % cap_;
            // wait for this slot's producer to publish (generation == head+1)
            if (seqs_[slot].load(std::memory_order_acquire) != head + 1) {
                std::this_thread::yield();
                continue;
            }
            out = ring_[slot];   // plain read, ordered after the acquire above
            // release the slot (only the consumer owns head)
            uint32_t expected = head;
            if (head_.compare_exchange_weak(expected, head + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
                // no need to clear seqs_[slot]; the next producer for this
                // physical slot writes a fresh generation (head+1+cap) and the
                // consumer only proceeds when it matches that generation.
                return true;
            }
            // lost head CAS (another consumer); retry with fresh head
        }
    }

private:
    uint32_t cap_;
    std::atomic<uint32_t> head_{0}, tail_{0};
    std::vector<slot_request_t>         ring_;    // plain PODs
    std::unique_ptr<std::atomic<uint32_t>[]> seqs_;   // publish generation per slot
};

} // namespace stream_moe

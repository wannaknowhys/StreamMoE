#pragma once

// Route B control plane: slot_meta 64-bit atomic word + expert directory +
// bounded MPSC alloc-request queue. Cross-platform (Windows / Linux), model-agnostic:
// "expert" is just an opaque (layer, expert) pair; no DeepSeek-specific logic.
// Design source: docs/Backend.md §2/§4.

#include <atomic>
#include <cstdint>
#include <cstring>
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
// Maps (layer, expert) -> slot index or UNASSIGNED. Compute threads read-only;
// scheduler thread writes. version[e] bumps on every mapping change so waiters
// can wake and re-scan without a lock.
static constexpr uint32_t SLOT_UNASSIGNED = 0xFFFFFFFFu;

class expert_directory {
public:
    expert_directory(uint32_t n_layers, uint32_t n_experts, uint32_t n_slots)
        : n_layers_(n_layers), n_experts_(n_experts), n_slots_(n_slots),
          entries_(static_cast<size_t>(n_layers) * n_experts),
          versions_(static_cast<size_t>(n_layers) * n_experts) {
        for (auto& e : entries_) e.store(SLOT_UNASSIGNED, std::memory_order_relaxed);
    }

    uint32_t find(uint32_t layer, uint32_t expert) const {
        return entries_[idx(layer, expert)].load(std::memory_order_acquire);
    }

    void set(uint32_t layer, uint32_t expert, uint32_t slot) {
        entries_[idx(layer, expert)].store(slot, std::memory_order_release);
        versions_[idx(layer, expert)].fetch_add(1, std::memory_order_acq_rel);
        slot_wake_all(&versions_[idx(layer, expert)]);
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

private:
    size_t idx(uint32_t layer, uint32_t expert) const {
        return static_cast<size_t>(layer) * n_experts_ + expert;
    }
    uint32_t n_layers_, n_experts_, n_slots_;
    std::vector<std::atomic<uint32_t>> entries_;
    std::vector<std::atomic<uint32_t>> versions_;
};

// ---- bounded MPSC alloc-request queue -------------------------------------
// Compute threads push (layer, expert); scheduler thread pops. Producers block
// when full (wait-on-space), consumers block when empty (wait-on-item).
struct slot_request_t {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint32_t seq = 0;
};

class mpsc_alloc_queue {
public:
    explicit mpsc_alloc_queue(uint32_t capacity) : cap_(capacity), ring_(capacity) {}

    void push(slot_request_t r) {
        uint32_t tail = tail_.load(std::memory_order_relaxed);
        while (true) {
            uint32_t head = head_.load(std::memory_order_acquire);
            if (tail - head < cap_) break;
            // full: wait for space
            std::this_thread::yield();
        }
        ring_[tail % cap_].store(r, std::memory_order_release);
        tail_.fetch_add(1, std::memory_order_release);
    }

    bool pop(slot_request_t& out) {
        uint32_t head = head_.load(std::memory_order_relaxed);
        while (true) {
            uint32_t tail = tail_.load(std::memory_order_acquire);
            if (head == tail) return false;
            if (head_.compare_exchange_weak(head, head + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                out = ring_[head % cap_].load(std::memory_order_acquire);
                return true;
            }
        }
    }

private:
    uint32_t cap_;
    std::atomic<uint32_t> head_{0}, tail_{0};
    std::vector<std::atomic<slot_request_t>> ring_;
};

} // namespace stream_moe

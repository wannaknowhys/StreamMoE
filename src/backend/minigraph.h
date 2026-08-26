#pragma once

// Route B mini-graph scratch arena (docs/LLAMA_MOE_NO_MMAP_RESEARCH.md §4.5).
// One reusable ggml context over a fixed external buffer: reset() re-inits the
// context on the same memory (no per-call malloc). Model-agnostic.
// Per-expert mini-graph construction (MUL_MAT_ID / MUL_MAT delegation) is built
// on top of this arena in moe_backend.

#include "ggml.h"
#include "backend/alloc.h"

#include <cstddef>
#include <cstring>
#include <memory>

namespace stream_moe {

class scratch_arena {
public:
    scratch_arena() = default;
    ~scratch_arena() { release(); }

    scratch_arena(const scratch_arena&) = delete;
    scratch_arena& operator=(const scratch_arena&) = delete;

    // Allocate backing memory (aligned). No ggml context yet.
    bool init(size_t bytes, size_t alignment = 64) {
        if (bytes == 0) return false;
        size_ = (bytes + alignment - 1) & ~(alignment - 1);
        mem_ = static_cast<uint8_t*>(aligned_alloc_ptr(size_, alignment));
        return mem_ != nullptr;
    }

    // (Re)create the ggml context over the same backing memory.
    bool reset() {
        release_ctx();
        if (!mem_) return false;
        ggml_init_params p;
        std::memset(&p, 0, sizeof(p));
        p.mem_size = size_;
        p.mem_buffer = mem_;
        p.no_alloc = false;
        ctx_ = ggml_init(p);
        return ctx_ != nullptr;
    }

    ggml_context* ctx() const { return ctx_; }
    size_t size() const { return size_; }

    void release() {
        release_ctx();
        aligned_free_ptr(mem_);
        mem_ = nullptr;
        size_ = 0;
    }

private:
    void release_ctx() {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
    }

    uint8_t*      mem_ = nullptr;
    size_t        size_ = 0;
    ggml_context* ctx_ = nullptr;
};

// Wrap contiguous quantized bytes as a 2D ggml tensor with external data.
// Layout must be standard ggml row-major for `type`; caller guarantees `data`
// outlives the arena reset.
inline ggml_tensor* wrap_2d_weight(ggml_context* ctx, enum ggml_type type,
                                   int64_t ne0, int64_t ne1, const void* data) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    t->data = const_cast<void*>(data);
    return t;
}

} // namespace stream_moe

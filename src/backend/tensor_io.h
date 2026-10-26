#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstring>

namespace stream_moe {

// Iron rule (docs/GRAPH_PARTITION.md): never dereference or memcpy a
// ggml_tensor::data on the host. A tensor may live on any backend (device-only
// memory, or a host mapping). These wrap the backend-agnostic copies; for a
// host-resident tensor they degrade to a plain copy.
//
// Mid-graph (inside a backend's graph_compute, while another backend may still
// be streaming) prefer the async variants + ggml_backend_synchronize, resolving
// the backend with ggml_backend_sched_get_tensor_backend. The synchronous form
// is fine during graph_reserve and for host-resident tensors.
//
// A tensor with no backend buffer is host memory (arena / heap, e.g. route-B
// relay shells and export captures before a buffer is bound). The ggml backend
// calls assert on a null buffer, so that single case is handled here as the
// documented raw-copy exception instead of at every call site.

static inline ggml_backend_buffer_t tensor_resolve_buffer(const ggml_tensor * t) {
    return t->view_src ? t->view_src->buffer : t->buffer;
}

static inline void tensor_read_host(const ggml_tensor * t, void * dst, size_t off, size_t size) {
    if (tensor_resolve_buffer(t)) {
        ggml_backend_tensor_get(t, dst, off, size);
        return;
    }
    GGML_ASSERT(t->data && "host tensor not allocated");
    GGML_ASSERT(off + size <= ggml_nbytes(t) && "host tensor read out of bounds");
    std::memcpy(dst, static_cast<const uint8_t *>(t->data) + off, size);  // iron-rule-exempt: host tensor, no backend buffer
}

static inline void tensor_write_host(ggml_tensor * t, const void * src, size_t off, size_t size) {
    if (tensor_resolve_buffer(t)) {
        ggml_backend_tensor_set(t, src, off, size);
        return;
    }
    GGML_ASSERT(t->data && "host tensor not allocated");
    GGML_ASSERT(off + size <= ggml_nbytes(t) && "host tensor write out of bounds");
    std::memcpy(static_cast<uint8_t *>(t->data) + off, src, size);  // iron-rule-exempt: host tensor, no backend buffer
}

} // namespace stream_moe

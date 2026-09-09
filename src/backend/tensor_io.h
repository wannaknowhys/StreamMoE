#pragma once

#include "ggml-backend.h"

#include <cstddef>

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
static inline void tensor_read_host(const ggml_tensor * t, void * dst, size_t off, size_t size) {
    ggml_backend_tensor_get(t, dst, off, size);
}

static inline void tensor_write_host(ggml_tensor * t, const void * src, size_t off, size_t size) {
    ggml_backend_tensor_set(t, src, off, size);
}

} // namespace stream_moe

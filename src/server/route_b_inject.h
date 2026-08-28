#pragma once

// Bridge between the upstream llama-server/llama-cli (vendored llama.cpp) and
// the StreamMoE route B expert pool. Called from common.cpp / speculative.cpp
// before llama_model_load_from_file when --expert-backend is passed.
// Multi-model (docs/MULTI_MODEL_POOL.md): the main model and the draft/MTP
// model each get their own pool instance + expert buft; overrides are
// per-model and never shared.
// See docs/UPSTREAM_TOOLS_MIGRATION.md §6.

#include "llama.h"

namespace stream_moe {

// Allocate the bounded expert pool for ONE model, register the stream_moe
// backend, and hand the MoE expert tensor patterns to this pool's weight buft.
// Idempotent per model path (returns the existing overrides on repeat calls).
// `pool_full_when_zero`: when ram_pool_mb == 0, size the pool to the FULL
// expert byte size (full residency, no eviction) instead of 75% free RAM.
// Returns a per-pool tensor_buft_overrides array (terminated by {nullptr,nullptr})
// or nullptr on failure.
llama_model_tensor_buft_override* route_b_setup(const char* model_path, size_t ram_pool_mb, int threads, bool pool_full_when_zero);

} // namespace stream_moe

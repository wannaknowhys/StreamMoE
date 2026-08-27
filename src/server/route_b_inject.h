#pragma once

// Bridge between the upstream llama-server/llama-cli (vendored llama.cpp) and
// the StreamMoE route B expert pool. Called from server-context.cpp load_model()
// before llama_model_load_from_file when --expert-backend is passed.
// See docs/UPSTREAM_TOOLS_MIGRATION.md §6.

#include "llama.h"

namespace stream_moe {

// Allocate the bounded expert pool, open model shards via DIO, start the
// scheduler, register the stream_moe backend, and hand the MoE expert tensor
// patterns to this backend's weight buft. Returns a static tensor_buft_overrides
// array (terminated by {nullptr,nullptr}) or nullptr on failure.
// The pool/backend live for the process lifetime (freed at exit).
llama_model_tensor_buft_override* route_b_setup(const char* model_path, size_t ram_pool_mb, int threads);

} // namespace stream_moe

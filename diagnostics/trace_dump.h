#pragma once

// Temporary per-layer tensor trace hook - SHORT-TERM DIAGNOSTIC ONLY.
// Lives in diagnostics/ (not src/) so it never pollutes the main codebase.
// Build a trace binary explicitly (see diagnostics/README.md); the normal
// build never compiles this.
//
// The only reference from main code is the one-line cb_eval hook in
// llama_engine.cpp, guarded by STREAM_MOE_TEMP (docs/PROJECT_STRUCTURE.md §10).

#include "ggml.h"

namespace stream_moe {

// ggml_backend_sched_eval_callback; dumps computed node outputs and MoE ids.
bool stream_moe_trace_cb(struct ggml_tensor* t, bool ask, void* ud);

} // namespace stream_moe

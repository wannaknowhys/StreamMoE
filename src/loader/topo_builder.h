#pragma once

#include "loader/moe_loader.h"
#include "loader/model.h"

#include <string>

namespace stream_moe {

// Convert a parsed model_t into the scheduler's moe_model_topology_t
// (expert_info_t / read_plan / sub-pool groups). Reading model_t means the
// DIO/staging/alignment policy is decided here:
//   - v2 / v2-chunk : whole (layer,expert) block, 4K-aligned, straight into slot
//   - v1            : per-branch slices, 4K-aligned (straight DIO)
//   - original      : per-branch slices, GGUF-align (staging + copy)
moe_model_topology_t build_topology(const model_t& m, const std::string& main_gguf_path);

} // namespace stream_moe

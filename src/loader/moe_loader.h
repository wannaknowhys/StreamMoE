#pragma once

#include "common/types.h"
#include "io/staging_reader.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace stream_moe {

struct sub_tensor_info_t {
    std::string name;
    uint32_t    shard_idx = 0;       // Index of the GGUF shard file containing this tensor
    uint64_t    abs_file_offset = 0; // Absolute file offset within that shard
    uint64_t    byte_size = 0;       // Bytes of this expert's slice
    uint64_t    slot_offset = 0;     // Offset inside the compact expert slot
    int32_t     ggml_type = 0;       // Quantization type
    int64_t     ne[4] = {0};         // Shape dimensions
};

struct expert_info_t {
    int32_t                        layer_idx = -1;
    int32_t                        expert_idx = -1;
    uint64_t                       total_expert_bytes = 0;
    std::vector<sub_tensor_info_t> sub_tensors;
    expert_read_plan_t             read_plan;
};

// GGUF layout of the expert storage (drives whether DIO needs a staging buffer).
// ORIGINAL / v1: expert slices are 32B-aligned -> stage + memcpy into the slot.
// V2 (expert-blocks): per-expert compact 4K-aligned blocks -> DIO straight into
// the slot (zero staging, zero copy). Known at load time per model (the draft
// and main models may use different layouts).
enum class gguf_layout_t : uint8_t {
    ORIGINAL          = 0,
    V1_SECTIONS       = 1,
    V2_EXPERT_BLOCKS  = 2,
};

struct moe_model_topology_t {
    std::string              arch_name;
    gguf_layout_t            layout = gguf_layout_t::ORIGINAL;
    uint32_t                 n_layer = 0;
    uint32_t                 n_expert = 0;
    uint32_t                 n_expert_used = 0;
    bool                     incomplete = false; // v2 chunk (strip files): dense tensors strip-scattered, need takeover
    bool needs_staging() const { return layout != gguf_layout_t::V2_EXPERT_BLOCKS; }
    
    // Attention & Context metadata
    uint32_t                 max_context_length = 4096;
    uint32_t                 embedding_length   = 2048;
    uint32_t                 head_count         = 32;
    uint32_t                 head_count_kv      = 32;
    uint32_t                 head_dim           = 64;

    // Multi-Head Latent Attention (MLA) & DeepSeek-V4 compression metadata
    bool                     is_mla             = false;
    uint32_t                 kv_lora_rank       = 0;  // Latent KV dimension d_c (e.g. 512)
    uint32_t                 qk_rope_dim        = 0;  // Decoupled RoPE dimension d_R (e.g. 64)
    uint32_t                 q_lora_rank        = 0;  // Q LoRA rank (e.g. 1024)
    std::vector<uint32_t>    dsv4_compress_ratios;    // Layer-wise compression ratios

    std::vector<std::string> shard_paths; // All discovered GGUF shard paths
    
    // Homogeneous layout metadata (group 0 unless heterogeneous)
    size_t      expert_slot_size = 0;
    size_t      expert_dio_staging_size = 0;
    uint32_t    num_sub_tensors_per_expert = 0;

    // Per-expert layout groups (docs/MULTI_SUBPOOL.md). Experts are grouped by
    // equal per-expert byte size; each group gets its own sub-pool whose slot
    // size equals the group's expert size. Homogeneous models have one group.
    struct expert_group_t {
        uint32_t              idx = 0;
        std::vector<uint32_t> layers;         // layers covered by this group
        size_t                expert_size = 0;  // bytes per expert (uniform within group)
        uint64_t              total_bytes  = 0; // layers.size()*n_expert*expert_size
    };
    std::vector<expert_group_t> groups;

    // Layer mapping
    std::vector<uint32_t> moe_layers;
    // Map from (layer_idx * n_expert + expert_idx) to expert_info_t
    std::vector<expert_info_t> experts;

    // Dense vs expert weight accounting, computed at parse time. Everything that
    // is NOT an expert tensor (`_exps` in name) stays on llama.cpp defaults
    // (mmap). For big MoE models the dense part is usually small - this drives
    // the "no-mmap" feasibility reasoning.
    std::vector<std::string> dense_tensor_names;
    uint64_t dense_total_bytes  = 0;
    uint64_t expert_total_bytes = 0;

    const expert_info_t& get_expert(uint32_t layer_idx, uint32_t expert_idx) const {
        size_t idx = static_cast<size_t>(layer_idx) * n_expert + expert_idx;
        return experts[idx];
    }
};

class moe_loader {
public:
    // Parse GGUF header & metadata across single or multi-shard files
    // Dynamically extracts MoE topology, MLA latent attention params, and performs homogeneity validation
    static moe_model_topology_t parse_gguf_topology(const std::string& main_gguf_path);
    // Multi-file variant (v2 chunk): explicit file list (c1.gguf;c2.gguf;...)
    static moe_model_topology_t parse_gguf_topology(const std::vector<std::string>& paths);
};

} // namespace stream_moe
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

struct moe_model_topology_t {
    std::string              arch_name;
    uint32_t                 n_layer = 0;
    uint32_t                 n_expert = 0;
    uint32_t                 n_expert_used = 0;
    
    // Attention & Context metadata
    uint32_t                 max_context_length = 4096;
    uint32_t                 embedding_length   = 2048;
    uint32_t                 head_count         = 32;
    uint32_t                 head_count_kv      = 32;
    uint32_t                 head_dim           = 64;

    std::vector<std::string> shard_paths; // All discovered GGUF shard paths
    
    // Homogeneous layout metadata
    size_t      expert_slot_size = 0;
    size_t      expert_dio_staging_size = 0;
    uint32_t    num_sub_tensors_per_expert = 0;

    // Layer mapping
    std::vector<uint32_t> moe_layers;
    // Map from (layer_idx * n_expert + expert_idx) to expert_info_t
    std::vector<expert_info_t> experts;

    const expert_info_t& get_expert(uint32_t layer_idx, uint32_t expert_idx) const {
        size_t idx = static_cast<size_t>(layer_idx) * n_expert + expert_idx;
        return experts[idx];
    }

    // Calculate total KV Cache memory footprint in bytes for a given context length
    // element_bytes: 2 for FP16/BF16/Q8_0, 1 for Q4_0
    size_t compute_kv_cache_bytes(uint32_t n_ctx, size_t element_bytes = 2) const {
        // KV Cache stores Key and Value tensors for all layers:
        // Size = 2 * n_ctx * n_layer * head_count_kv * head_dim * element_bytes
        uint32_t kv_heads = (head_count_kv > 0) ? head_count_kv : head_count;
        uint32_t h_dim = (head_dim > 0) ? head_dim : (embedding_length / (head_count > 0 ? head_count : 1));
        return 2ULL * n_ctx * n_layer * kv_heads * h_dim * element_bytes;
    }
};

class moe_loader {
public:
    // Parse GGUF header & metadata across single or multi-shard files
    // Dynamically extracts MoE topology, Attention metadata, and performs homogeneity validation
    static moe_model_topology_t parse_gguf_topology(const std::string& main_gguf_path);
};

} // namespace stream_moe
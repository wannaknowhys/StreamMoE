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

struct kv_cache_info_t {
    bool        is_mla = false;             // Multi-Head Latent Attention (DeepSeek)
    bool        has_v_tensor = true;        // Standard MHA has separate V tensor, MLA does not
    uint32_t    kv_lora_rank = 0;           // c_kv dimension (e.g. 512 for DeepSeek)
    uint32_t    qk_rope_dim = 0;            // Decoupled RoPE dimension (e.g. 64)
    uint32_t    total_latent_dim = 0;       // (kv_lora_rank + qk_rope_dim) or (2 * n_head_kv * head_dim)
    size_t      uncompressed_mha_bytes = 0; // Baseline standard MHA size
    size_t      actual_kv_bytes = 0;        // Actual compressed memory footprint
    double      compression_ratio = 1.0;    // actual_kv_bytes / uncompressed_mha_bytes
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

    // Multi-Head Latent Attention (MLA) & DeepSeek-V4 compression metadata
    bool                     is_mla             = false;
    uint32_t                 kv_lora_rank       = 0;  // Latent KV dimension d_c (e.g. 512)
    uint32_t                 qk_rope_dim        = 0;  // Decoupled RoPE dimension d_R (e.g. 64)
    uint32_t                 q_lora_rank        = 0;  // Q LoRA rank (e.g. 1024)
    std::vector<uint32_t>    dsv4_compress_ratios;    // Layer-wise compression ratios

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

    // Accurately compute KV Cache memory footprint accounting for MLA latent compression & DSV4 ratios
    // element_bytes: 2 for FP16/BF16, 1 for Q8_0/FP8, 0.5 for Q4_0
    kv_cache_info_t compute_kv_cache_info(uint32_t n_ctx, size_t element_bytes = 2) const {
        kv_cache_info_t info;
        info.is_mla = is_mla;
        info.kv_lora_rank = kv_lora_rank;
        info.qk_rope_dim = qk_rope_dim;

        // Baseline standard MHA size: 2 * n_ctx * n_layer * head_count * head_dim * element_bytes
        uint32_t h_dim = (head_dim > 0) ? head_dim : (embedding_length / (head_count > 0 ? head_count : 1));
        uint32_t h_count = (head_count > 0) ? head_count : 32;
        info.uncompressed_mha_bytes = 2ULL * n_ctx * n_layer * h_count * h_dim * element_bytes;

        if (is_mla) {
            // MLA architecture (DeepSeek-V2 / V3 / V4):
            // 1. NO separate V tensor stored!
            // 2. K tensor stores compressed latent vector (kv_lora_rank) + decoupled RoPE key (qk_rope_dim)
            info.has_v_tensor = false;
            info.total_latent_dim = (kv_lora_rank > 0 ? kv_lora_rank : 512) + (qk_rope_dim > 0 ? qk_rope_dim : 64);

            size_t total_bytes = 0;
            for (uint32_t l = 0; l < n_layer; ++l) {
                uint32_t effective_ctx = n_ctx;
                if (l < dsv4_compress_ratios.size() && dsv4_compress_ratios[l] > 1) {
                    // DeepSeek-V4 Layer-wise compressed attention
                    effective_ctx = (n_ctx + dsv4_compress_ratios[l] - 1) / dsv4_compress_ratios[l];
                }
                total_bytes += static_cast<size_t>(effective_ctx) * info.total_latent_dim * element_bytes;
            }
            info.actual_kv_bytes = total_bytes;
        } else {
            // Standard MHA / GQA (LLaMA, Qwen, Mistral):
            info.has_v_tensor = true;
            uint32_t kv_heads = (head_count_kv > 0) ? head_count_kv : head_count;
            info.total_latent_dim = 2 * kv_heads * h_dim;
            info.actual_kv_bytes = 2ULL * n_ctx * n_layer * kv_heads * h_dim * element_bytes;
        }

        if (info.uncompressed_mha_bytes > 0) {
            info.compression_ratio = static_cast<double>(info.actual_kv_bytes) / static_cast<double>(info.uncompressed_mha_bytes);
        }
        return info;
    }
};

class moe_loader {
public:
    // Parse GGUF header & metadata across single or multi-shard files
    // Dynamically extracts MoE topology, MLA latent attention params, and performs homogeneity validation
    static moe_model_topology_t parse_gguf_topology(const std::string& main_gguf_path);
};

} // namespace stream_moe
#pragma once

#include "common/types.h"
#include "loader/moe_loader.h"
#include <string>
#include <vector>
#include <memory>

namespace stream_moe {

#pragma pack(push, 1)
struct smkv_header_t {
    uint32_t magic;               // 0x534D4B56 ("SMKV")
    uint32_t version;             // 1
    char     arch_name[64];       // e.g. "deepseek4"
    uint32_t n_layer;             // Total layers
    uint32_t is_mla;              // 1 = MLA, 0 = MHA/GQA
    uint32_t total_latent_dim;    // e.g. 576
    uint32_t element_bytes;       // e.g. 2 for FP16
    uint32_t n_tokens;            // Active tokens stored
    uint32_t prompt_history_len;  // UTF-8 string bytes
    uint32_t token_ids_count;     // Number of token IDs
    uint64_t kv_payload_bytes;    // Total raw KV tensor payload bytes
    uint64_t timestamp_ns;        // Snapshot timestamp
};
#pragma pack(pop)

class dynamic_kv_buffer {
public:
    dynamic_kv_buffer(
        uint32_t n_layer,
        uint32_t total_latent_dim,
        uint32_t element_bytes = 2,
        uint32_t chunk_tokens = 4096
    );
    ~dynamic_kv_buffer();

    // Disable copy
    dynamic_kv_buffer(const dynamic_kv_buffer&) = delete;
    dynamic_kv_buffer& operator=(const dynamic_kv_buffer&) = delete;

    // Ensure capacity for at least target_tokens
    void reserve(uint32_t target_tokens);

    void*       data() { return buffer_; }
    const void* data() const { return buffer_; }
    size_t      allocated_bytes() const { return allocated_bytes_; }
    uint32_t    capacity_tokens() const { return capacity_tokens_; }
    uint32_t    current_tokens() const { return current_tokens_; }
    void        set_current_tokens(uint32_t n) { current_tokens_ = n; }

    size_t      bytes_per_token() const {
        return static_cast<size_t>(n_layer_) * total_latent_dim_ * element_bytes_;
    }

    size_t      used_bytes() const {
        return static_cast<size_t>(current_tokens_) * bytes_per_token();
    }

private:
    uint32_t n_layer_;
    uint32_t total_latent_dim_;
    uint32_t element_bytes_;
    uint32_t chunk_tokens_;
    uint32_t capacity_tokens_ = 0;
    uint32_t current_tokens_ = 0;
    size_t   allocated_bytes_ = 0;
    void*    buffer_ = nullptr;
};

class kv_cache_manager {
public:
    // Save KV Cache state, Token IDs, and Conversation Prompt History to a binary snapshot file
    static bool save_snapshot(
        const std::string& filepath,
        const std::string& prompt_history,
        const std::vector<int32_t>& token_ids,
        const dynamic_kv_buffer& kv_buf,
        const moe_model_topology_t& topo
    );

    // Load KV Cache state, Token IDs, and Prompt History from a snapshot file into the dynamic buffer
    static bool load_snapshot(
        const std::string& filepath,
        std::string& out_prompt_history,
        std::vector<int32_t>& out_token_ids,
        dynamic_kv_buffer& out_kv_buf,
        const moe_model_topology_t& topo
    );
};

} // namespace stream_moe
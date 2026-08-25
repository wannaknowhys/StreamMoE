#include "kv/kv_cache_manager.h"
#include "common/logger.h"

#include <fstream>
#include <cstring>
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#else
#include <stdlib.h>
#endif

namespace stream_moe {

static constexpr uint32_t SMKV_MAGIC = 0x534D4B56; // "SMKV"
static constexpr uint32_t SMKV_VERSION = 1;

dynamic_kv_buffer::dynamic_kv_buffer(
    uint32_t n_layer,
    uint32_t total_latent_dim,
    uint32_t element_bytes,
    uint32_t chunk_tokens
) : n_layer_(n_layer),
    total_latent_dim_(total_latent_dim),
    element_bytes_(element_bytes),
    chunk_tokens_(chunk_tokens > 0 ? chunk_tokens : 4096) {
    // Initial allocation for 1 chunk
    reserve(chunk_tokens_);
}

dynamic_kv_buffer::~dynamic_kv_buffer() {
    if (buffer_) {
#if defined(_WIN32)
        VirtualFree(buffer_, 0, MEM_RELEASE);
#else
        free(buffer_);
#endif
        buffer_ = nullptr;
    }
}

void dynamic_kv_buffer::reserve(uint32_t target_tokens) {
    if (target_tokens <= capacity_tokens_) return;

    // Round up to multiple of chunk_tokens_
    uint32_t new_capacity = ((target_tokens + chunk_tokens_ - 1) / chunk_tokens_) * chunk_tokens_;
    size_t new_bytes = static_cast<size_t>(new_capacity) * bytes_per_token();

#if defined(_WIN32)
    void* new_buf = VirtualAlloc(NULL, new_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* new_buf = malloc(new_bytes);
#endif

    if (!new_buf) {
        LOG_ERROR("Failed to allocate dynamic KV buffer of " << (new_bytes / (1024 * 1024)) << " MB");
        throw std::bad_alloc();
    }

    // Copy existing data if any
    if (buffer_ && current_tokens_ > 0) {
        size_t current_bytes = used_bytes();
        std::memcpy(new_buf, buffer_, current_bytes);
#if defined(_WIN32)
        VirtualFree(buffer_, 0, MEM_RELEASE);
#else
        free(buffer_);
#endif
    }

    buffer_ = new_buf;
    capacity_tokens_ = new_capacity;
    allocated_bytes_ = new_bytes;
}

bool kv_cache_manager::save_snapshot(
    const std::string& filepath,
    const std::string& prompt_history,
    const std::vector<int32_t>& token_ids,
    const dynamic_kv_buffer& kv_buf,
    const moe_model_topology_t& topo
) {
    auto t_start = std::chrono::steady_clock::now();

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open snapshot file for writing: " << filepath);
        return false;
    }

    smkv_header_t hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic = SMKV_MAGIC;
    hdr.version = SMKV_VERSION;
    std::strncpy(hdr.arch_name, topo.arch_name.c_str(), sizeof(hdr.arch_name) - 1);
    hdr.n_layer = topo.n_layer;
    hdr.is_mla = topo.is_mla ? 1 : 0;
    
    auto kv_info = topo.compute_kv_cache_info(kv_buf.current_tokens(), 2);
    hdr.total_latent_dim = kv_info.total_latent_dim;
    hdr.element_bytes = 2; // FP16
    hdr.n_tokens = kv_buf.current_tokens();
    hdr.prompt_history_len = static_cast<uint32_t>(prompt_history.size());
    hdr.token_ids_count = static_cast<uint32_t>(token_ids.size());
    hdr.kv_payload_bytes = kv_buf.used_bytes();
    hdr.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // 1. Write Header
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // 2. Write Prompt History string
    if (hdr.prompt_history_len > 0) {
        out.write(prompt_history.data(), hdr.prompt_history_len);
    }

    // 3. Write Token IDs
    if (hdr.token_ids_count > 0) {
        out.write(reinterpret_cast<const char*>(token_ids.data()), hdr.token_ids_count * sizeof(int32_t));
    }

    // 4. Write KV Cache raw payload
    if (hdr.kv_payload_bytes > 0 && kv_buf.data()) {
        out.write(reinterpret_cast<const char*>(kv_buf.data()), hdr.kv_payload_bytes);
    }

    out.close();

    auto t_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    LOG_INFO("Saved KV Cache Snapshot (" << hdr.n_tokens << " tokens, " 
             << (hdr.kv_payload_bytes / (1024.0 * 1024.0)) << " MB) to " << filepath << " in " << ms << " ms");
    return true;
}

bool kv_cache_manager::load_snapshot(
    const std::string& filepath,
    std::string& out_prompt_history,
    std::vector<int32_t>& out_token_ids,
    dynamic_kv_buffer& out_kv_buf,
    const moe_model_topology_t& topo
) {
    auto t_start = std::chrono::steady_clock::now();

    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        LOG_ERROR("Failed to open snapshot file for reading: " << filepath);
        return false;
    }

    smkv_header_t hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!in.good() || hdr.magic != SMKV_MAGIC || hdr.version != SMKV_VERSION) {
        LOG_ERROR("Invalid SMKV snapshot format or version mismatch: " << filepath);
        return false;
    }

    if (hdr.n_layer != topo.n_layer) {
        LOG_ERROR("Snapshot layer count mismatch: file has " << hdr.n_layer << ", model has " << topo.n_layer);
        return false;
    }

    // 1. Read Prompt History
    if (hdr.prompt_history_len > 0) {
        out_prompt_history.resize(hdr.prompt_history_len);
        in.read(&out_prompt_history[0], hdr.prompt_history_len);
    } else {
        out_prompt_history.clear();
    }

    // 2. Read Token IDs
    if (hdr.token_ids_count > 0) {
        out_token_ids.resize(hdr.token_ids_count);
        in.read(reinterpret_cast<char*>(out_token_ids.data()), hdr.token_ids_count * sizeof(int32_t));
    } else {
        out_token_ids.clear();
    }

    // 3. Read KV Cache Payload
    if (hdr.n_tokens > 0) {
        out_kv_buf.reserve(hdr.n_tokens);
        out_kv_buf.set_current_tokens(hdr.n_tokens);
        in.read(reinterpret_cast<char*>(out_kv_buf.data()), hdr.kv_payload_bytes);
    }

    in.close();

    auto t_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    LOG_INFO("Loaded KV Cache Snapshot (" << hdr.n_tokens << " tokens, " 
             << (hdr.kv_payload_bytes / (1024.0 * 1024.0)) << " MB) from " << filepath << " in " << ms << " ms");
    return true;
}

} // namespace stream_moe
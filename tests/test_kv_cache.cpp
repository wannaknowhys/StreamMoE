#include "kv/kv_cache_manager.h"
#include "common/logger.h"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace stream_moe;

void test_dynamic_kv_buffer() {
    std::cout << "[Test 1] Dynamic KV Buffer Allocation & Expansion...\n";
    // 43 layers, 576 latent dim, 2 bytes/elem (DeepSeek MLA layout)
    dynamic_kv_buffer buf(43, 576, 2, 4096);
    assert(buf.capacity_tokens() == 4096);
    assert(buf.current_tokens() == 0);
    assert(buf.allocated_bytes() == 4096ULL * 43 * 576 * 2);

    // Reserve for 10000 tokens -> should expand to 3 chunks (12288 tokens)
    buf.reserve(10000);
    assert(buf.capacity_tokens() == 12288);
    assert(buf.allocated_bytes() == 12288ULL * 43 * 576 * 2);

    std::cout << "[+] test_dynamic_kv_buffer PASSED!\n";
}

void test_kv_snapshot_persistence() {
    std::cout << "[Test 2] KV Cache & Prompt Snapshot Persistence...\n";
    moe_model_topology_t topo;
    topo.arch_name = "deepseek4";
    topo.n_layer = 43;
    topo.is_mla = true;
    topo.kv_lora_rank = 512;
    topo.qk_rope_dim = 64;

    dynamic_kv_buffer buf(43, 576, 2, 4096);
    buf.set_current_tokens(256);

    // Fill buffer with pseudo-data pattern
    uint8_t* ptr = static_cast<uint8_t*>(buf.data());
    for (size_t i = 0; i < buf.used_bytes(); ++i) {
        ptr[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);
    }

    std::string prompt_history = "User: You are an expert AI engineer.\nAssistant: Understood.";
    std::vector<int32_t> token_ids = {101, 2054, 2003, 102};

    std::string snapshot_path = "temp/test_snapshot.smkv";
    if (std::filesystem::exists(snapshot_path)) {
        std::filesystem::remove(snapshot_path);
    }

    bool saved = kv_cache_manager::save_snapshot(snapshot_path, prompt_history, token_ids, buf, topo);
    assert(saved);
    assert(std::filesystem::exists(snapshot_path));

    // Restore into a new buffer
    dynamic_kv_buffer restored_buf(43, 576, 2, 4096);
    std::string restored_prompt;
    std::vector<int32_t> restored_tokens;

    bool loaded = kv_cache_manager::load_snapshot(snapshot_path, restored_prompt, restored_tokens, restored_buf, topo);
    assert(loaded);
    assert(restored_buf.current_tokens() == 256);
    assert(restored_prompt == prompt_history);
    assert(restored_tokens.size() == token_ids.size());
    for (size_t i = 0; i < token_ids.size(); ++i) {
        assert(restored_tokens[i] == token_ids[i]);
    }

    // Verify binary tensor payload matches exactly
    const uint8_t* restored_ptr = static_cast<const uint8_t*>(restored_buf.data());
    for (size_t i = 0; i < restored_buf.used_bytes(); ++i) {
        assert(restored_ptr[i] == ptr[i]);
    }

    std::cout << "[+] test_kv_snapshot_persistence PASSED!\n";
}

int main() {
    std::cout << "===========================================\n"
              << "  Running StreamMoE KV Persistence Tests   \n"
              << "===========================================\n";
    test_dynamic_kv_buffer();
    test_kv_snapshot_persistence();
    std::cout << "===========================================\n"
              << "  ALL KV CACHE TESTS PASSED SUCCESSFULLY!  \n"
              << "===========================================\n";
    return 0;
}
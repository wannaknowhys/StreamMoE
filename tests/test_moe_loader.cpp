#include "loader/moe_loader.h"
#include "common/logger.h"

#include <iostream>
#include <cassert>
#include <filesystem>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_PASS(name) \
    std::cout << "[+] TEST PASSED: " << name << std::endl

// Test 1: Dense GGUF Model parsing (Qwen3-VL)
bool test_dense_gguf_parsing() {
    const std::string qwen_path = "F:/Dev/computer-use/Qwen3-VL-2B-Instruct-Q4_K_M.gguf";
    if (!std::filesystem::exists(qwen_path)) {
        LOG_WARN("Qwen3-VL test file not found (skipping)");
        return true;
    }

    auto topo = moe_loader::parse_gguf_topology(qwen_path);
    TEST_ASSERT(!topo.arch_name.empty(), "arch_name should not be empty");
    TEST_ASSERT(topo.n_layer > 0, "n_layer should be > 0");
    TEST_ASSERT(topo.n_expert == 0, "Dense model should have 0 experts");
    TEST_ASSERT(topo.moe_layers.empty(), "Dense model should have 0 MoE layers");

    LOG_INFO("Dense Model verified: arch=" << topo.arch_name << ", layers=" << topo.n_layer);
    TEST_PASS("test_dense_gguf_parsing");
    return true;
}

// Test 2: Real MoE Model Parsing & Homogeneity Verification (DeepSeek-V4 Shard 1)
bool test_real_moe_gguf_parsing() {
    const std::string ds_path = "N:/AI_LLM/DeepSeek-V4-Flash-0731/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf";
    if (!std::filesystem::exists(ds_path)) {
        LOG_WARN("DeepSeek-V4 Shard 1 test file not found (skipping)");
        return true;
    }

    auto topo = moe_loader::parse_gguf_topology(ds_path);
    TEST_ASSERT(!topo.arch_name.empty(), "arch_name should not be empty");
    TEST_ASSERT(topo.n_layer > 0, "n_layer should be > 0");
    TEST_ASSERT(topo.n_expert > 0, "MoE model must have n_expert > 0");
    TEST_ASSERT(topo.expert_slot_size > 0, "expert_slot_size must be > 0");
    TEST_ASSERT(is_aligned(topo.expert_slot_size, 4096), "expert_slot_size must be 4KB aligned");
    TEST_ASSERT(topo.expert_dio_staging_size > 0, "expert_dio_staging_size must be > 0");

    LOG_INFO("DeepSeek-V4 MoE Topology verified: "
             << "arch=" << topo.arch_name
             << ", layers=" << topo.n_layer
             << ", moe_layers=" << topo.moe_layers.size()
             << ", n_expert=" << topo.n_expert
             << ", n_expert_used=" << topo.n_expert_used
             << ", slot_size=" << (topo.expert_slot_size / 1024) << " KB"
             << ", staging_size=" << (topo.expert_dio_staging_size / 1024) << " KB");

    // Inspect first MoE layer expert 0 read plan
    if (!topo.moe_layers.empty()) {
        uint32_t first_moe_l = topo.moe_layers[0];
        const auto& exp0 = topo.get_expert(first_moe_l, 0);
        TEST_ASSERT(exp0.layer_idx == static_cast<int32_t>(first_moe_l), "Layer idx mismatch");
        TEST_ASSERT(exp0.expert_idx == 0, "Expert idx mismatch");
        TEST_ASSERT(!exp0.sub_tensors.empty(), "Sub-tensors should not be empty");
        TEST_ASSERT(exp0.read_plan.num_tensors == exp0.sub_tensors.size(), "Read plan tensor count mismatch");
        
        LOG_INFO("Expert 0 in Layer " << first_moe_l << " has " << exp0.sub_tensors.size() << " sub-tensors:");
        for (const auto& st : exp0.sub_tensors) {
            LOG_INFO("  - Sub-tensor " << st.name << ": size=" << st.byte_size << " bytes, file_offset=" << st.abs_file_offset);
        }
    }

    TEST_PASS("test_real_moe_gguf_parsing");
    return true;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Running StreamMoE Phase 3 Unit Tests     " << std::endl;
    std::cout << "===========================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_dense_gguf_parsing();
    all_passed &= test_real_moe_gguf_parsing();

    std::cout << "===========================================" << std::endl;
    if (all_passed) {
        std::cout << "  ALL PHASE 3 TESTS PASSED SUCCESSFULLY!    " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 0;
    } else {
        std::cerr << "  SOME PHASE 3 TESTS FAILED!                " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 1;
    }
}
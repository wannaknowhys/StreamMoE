#include "profile/profiler.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <string>

using namespace stream_moe;

void test_profiler_logging() {
    std::cout << "[Test 1] Testing High-Resolution Profiler & JSONL Serialization...\n";
    std::string test_log = "temp/test_profile_log.jsonl";
    if (std::filesystem::exists(test_log)) {
        std::filesystem::remove(test_log);
    }

    auto& logger = profile_logger::instance();
    logger.init(test_log);
    assert(logger.is_enabled());

    // 1. Log Request Ingest
    logger.log_request_ingest(1, 120, 32);

    // 2. Populate Turn Profile
    turn_profile_t p;
    p.turn_id = 1;
    p.timestamp_ns = read_timestamp_ns();
    p.prompt_tokens = 32;
    p.generated_tokens = 64;
    p.prefill_tps = 45.2;
    p.decode_tps = 12.8;
    p.total_duration_ms = 5000.0;

    p.total_lookups = 100;
    p.gpu_hits = 30;
    p.ram_hits = 50;
    p.disk_misses = 20;

    // Speculative accept histogram: 2 times 0 correct, 3 times 1 correct, 5 times 2 correct, 6 times 3 correct
    p.spec_accept_hist[0] = 2;
    p.spec_accept_hist[1] = 3;
    p.spec_accept_hist[2] = 5;
    p.spec_accept_hist[3] = 6;

    p.t_prefill_ns = 50000000;
    p.t_prefix_match_ns = 1200000;
    p.t_attn_layer_ns = 35000000;
    p.t_expert_total_ns = 400000000;
    p.t_expert_wait_io_ns = 150000000;
    p.t_expert_cpu_ns = 180000000;
    p.t_expert_gpu_ns = 70000000;
    p.t_sync_pcie_ns = 25000000;
    p.t_merge_reduce_ns = 15000000;

    logger.log_response_finish(p);
    logger.close();

    // Verify file content
    std::ifstream in(test_log);
    assert(in.is_open());
    std::string line1, line2;
    std::getline(in, line1);
    std::getline(in, line2);

    assert(line1.find("\"event\":\"request_ingest\"") != std::string::npos);
    assert(line1.find("\"turn\":1") != std::string::npos);

    assert(line2.find("\"event\":\"response_finish\"") != std::string::npos);
    assert(line2.find("\"gpu_hit_pct\":30.00") != std::string::npos);
    assert(line2.find("\"ram_hit_pct\":50.00") != std::string::npos);
    assert(line2.find("\"total_hit_pct\":80.00") != std::string::npos);
    assert(line2.find("\"speculative_hist\":[2,3,5,6,0,0,0,0,0]") != std::string::npos);
    assert(line2.find("\"expert_wait_io\":150000000") != std::string::npos);

    std::cout << "[+] test_profiler_logging PASSED!\n";
}

int main() {
    std::cout << "===========================================\n"
              << "  Running StreamMoE Profiler Tests         \n"
              << "===========================================\n";
    test_profiler_logging();
    std::cout << "===========================================\n"
              << "  ALL PROFILER TESTS PASSED SUCCESSFULLY!  \n"
              << "===========================================\n";
    return 0;
}
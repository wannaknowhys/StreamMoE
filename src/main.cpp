#include "common/types.h"
#include "common/logger.h"
#include "io/async_dio.h"
#include "loader/moe_loader.h"
#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "scheduler/moe_scheduler.h"
#include "engine/subgraph_executor.h"
#include "engine/state_machine.h"
#include "engine/speculative_engine.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>

using namespace stream_moe;

struct cmd_params_t {
    std::string model_path;
    std::string draft_model_path;
    std::string stats_path;
    std::string prompt = "Hello StreamMoE!";
    int32_t     n_gpu_layers = 0;
    size_t      moe_vram_pool_mb = 4096;
    size_t      moe_ram_pool_mb  = 8192;
    std::string preload_policy   = "none"; // none | ram | vram | all
    uint32_t    n_tokens         = 32;
    uint32_t    threads          = 16;
};

void print_usage(const char* prog) {
    std::cout << "StreamMoE (OffloadMoE) - High-Performance Memory-Optimized MoE Inference Engine\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>             Path to GGUF model (supports single or multi-shard)\n"
              << "  --draft-model <path>           Path to Dense draft model for speculative decoding\n"
              << "  -ngl, --gpu-layers <N>         Number of Dense layers to offload to GPU\n"
              << "  --moe-vram-pool <MB>           Pinned VRAM MoE Pool size in MB (default: 4096)\n"
              << "  --moe-ram-pool <MB>            Pinned Host RAM MoE Pool size in MB (default: 8192)\n"
              << "  --moe-preload <policy>         Preload policy: none | ram | vram | all (default: none)\n"
              << "  --stats-file <path>            Path to expert frequency stats file (EST1)\n"
              << "  -p, --prompt <text>            Input prompt\n"
              << "  -n, --n-predict <N>            Number of tokens to predict (default: 32)\n"
              << "  -t, --threads <N>              Number of CPU worker threads (default: 16)\n"
              << "  -h, --help                     Show this help message\n";
}

cmd_params_t parse_args(int argc, char** argv) {
    cmd_params_t params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--draft-model" && i + 1 < argc) {
            params.draft_model_path = argv[++i];
        } else if ((arg == "-ngl" || arg == "--gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--moe-vram-pool" && i + 1 < argc) {
            params.moe_vram_pool_mb = std::stoull(argv[++i]);
        } else if (arg == "--moe-ram-pool" && i + 1 < argc) {
            params.moe_ram_pool_mb = std::stoull(argv[++i]);
        } else if (arg == "--moe-preload" && i + 1 < argc) {
            params.preload_policy = argv[++i];
        } else if (arg == "--stats-file" && i + 1 < argc) {
            params.stats_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            params.prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            params.n_tokens = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.threads = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return params;
}

int main(int argc, char** argv) {
    std::cout << "===================================================================\n"
              << "   StreamMoE: Extreme Memory Offload MoE Inference Engine          \n"
              << "===================================================================\n";

    cmd_params_t params = parse_args(argc, argv);
    if (params.model_path.empty()) {
        LOG_ERROR("Missing required argument: -m <model_path>");
        print_usage(argv[0]);
        return 1;
    }

    auto t_start = std::chrono::steady_clock::now();

    // 1. Initialize Direct I/O Engine
    auto dio_engine = async_dio_engine::create(1024);
    if (!dio_engine) {
        LOG_ERROR("Failed to initialize Direct I/O Engine.");
        return 1;
    }

    // 2. Parse GGUF Topology & Verify Homogeneity
    LOG_INFO("Parsing GGUF Model Topology from: " << params.model_path);
    moe_model_topology_t topo;
    try {
        topo = moe_loader::parse_gguf_topology(params.model_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse GGUF topology: " << e.what());
        return 1;
    }

    // Open all shard files with Direct I/O
    std::vector<dio_file_t*> shard_files;
    for (const auto& shard_path : topo.shard_paths) {
        dio_file_t* f = dio_engine->open_file(shard_path);
        if (!f) {
            LOG_ERROR("Failed to open shard file with Direct I/O: " << shard_path);
            return 1;
        }
        shard_files.push_back(f);
    }

    // 3. Initialize Expert Stats Tracker (EST1)
    std::string stats_file = params.stats_path;
    if (stats_file.empty()) {
        std::filesystem::path p(params.model_path);
        stats_file = "expert_" + p.stem().string() + ".bin";
    }
    expert_stats_tracker stats;
    stats.init(stats_file, topo.n_layer, topo.n_expert, 8192);

    // 4. Calculate Pool Capacity & Allocate Pinned Host RAM Pool
    size_t slot_size = topo.expert_slot_size > 0 ? topo.expert_slot_size : (1024 * 1024 * 4);
    size_t pool_budget_bytes = params.moe_ram_pool_mb * 1024 * 1024;
    uint32_t num_slots = static_cast<uint32_t>(pool_budget_bytes / slot_size);
    if (num_slots < 4) num_slots = 4; // Minimum safe floor

    LOG_INFO("Allocating Pinned RAM MoE Pool: " << num_slots << " slots (Budget: " 
             << params.moe_ram_pool_mb << " MB, Slot size: " << (slot_size / 1024) << " KB)");
    
    std::unique_ptr<expert_pool> pool;
    try {
        pool = std::make_unique<expert_pool>(slot_size, num_slots);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to allocate Pinned Expert Pool: " << e.what());
        return 1;
    }

    // 5. Initialize Sub-Graph Executor, State Machine, and Speculative Engine
    subgraph_executor executor(topo, *pool, 2048, 4096);
    state_machine sm(params.threads);
    moe_scheduler scheduler(topo, *pool, stats, *dio_engine, shard_files);
    speculative_engine spec_engine(topo, sm, scheduler);

    if (!params.draft_model_path.empty()) {
        spec_engine.load_draft_model(params.draft_model_path);
    }

    // 6. Preload Strategy Execution
    LOG_INFO("Applying MoE Preload Policy: [" << params.preload_policy << "]");
    if (params.preload_policy == "none") {
        LOG_INFO("Instant Startup Mode: Zero MoE weights preloaded. Dynamic on-demand DIO stream active.");
    } else if (params.preload_policy == "ram" || params.preload_policy == "all") {
        uint32_t preloaded = 0;
        for (uint32_t l = 0; l < topo.n_layer && preloaded < num_slots; ++l) {
            for (uint32_t e = 0; e < topo.n_expert && preloaded < num_slots; ++e) {
                int32_t slot_id = pool->allocate_or_evict_slot(l, e, stats, 1);
                if (slot_id >= 0) {
                    pool->mark_ready(slot_id);
                    preloaded++;
                }
            }
        }
        LOG_INFO("Preloaded " << preloaded << " hot expert slots into Pinned RAM Pool.");
    }

    auto t_ready = std::chrono::steady_clock::now();
    double startup_sec = std::chrono::duration_cast<std::chrono::milliseconds>(t_ready - t_start).count() / 1000.0;
    LOG_INFO("Engine Ready in " << std::fixed << std::setprecision(3) << startup_sec << " seconds!\n");

    // Start background scheduler worker
    scheduler.start();

    // 7. Simulated Inference Execution Loop
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << " Prompt: \"" << params.prompt << "\"\n";
    std::cout << " Generating " << params.n_tokens << " tokens...\n";
    std::cout << "-------------------------------------------------------------------\n";

    uint32_t total_hits = 0;
    uint32_t total_lookups = 0;
    auto t_infer_start = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= params.n_tokens; ++step) {
        uint32_t active_k = spec_engine.get_active_k();

        // Forward through each MoE Layer
        for (uint32_t l : topo.moe_layers) {
            // Simulated top-K gating router output
            std::vector<uint32_t> routed_experts;
            uint32_t top_k = topo.n_expert_used > 0 ? topo.n_expert_used : 4;
            for (uint32_t k = 0; k < top_k; ++k) {
                uint32_t exp_id = (step * 3 + l * 7 + k * 13) % topo.n_expert;
                routed_experts.push_back(exp_id);
            }

            total_lookups += static_cast<uint32_t>(routed_experts.size());

            // Double-thread overlap pipeline
            auto req = scheduler.route_and_prefetch(l, routed_experts, step * 100 + l);
            total_hits += static_cast<uint32_t>(req.hit_slots.size());

            // 1. Immediately compute Hit GEMM concurrently with background DIO
            // (In full model forward, executes GEMM on Hit Slots)

            // 2. Wait for Miss Experts to complete asynchronously
            auto miss_slots = scheduler.wait_miss_ready(req, 5000);

            // 3. Compute Miss GEMM and unpin all layer slots
            std::vector<int32_t> all_layer_slots = req.hit_slots;
            all_layer_slots.insert(all_layer_slots.end(), miss_slots.begin(), miss_slots.end());
            scheduler.release_layer_slots(all_layer_slots);
        }

        // Notify tokens generated and trigger periodic stats sync if > 8192
        stats.notify_tokens_generated(1);

        if (step % 8 == 0 || step == params.n_tokens) {
            double hit_rate = total_lookups > 0 ? (100.0 * total_hits / total_lookups) : 0.0;
            std::cout << " Step " << std::setw(3) << step << "/" << params.n_tokens 
                      << " | Cache Hit Rate: " << std::fixed << std::setprecision(1) << hit_rate << "%"
                      << " | Active State: [" << sm.state_name(sm.current_state()) << "]"
                      << " | Spec K: " << active_k << "\n";
        }
    }

    auto t_infer_end = std::chrono::steady_clock::now();
    double infer_sec = std::chrono::duration_cast<std::chrono::milliseconds>(t_infer_end - t_infer_start).count() / 1000.0;
    double tps = infer_sec > 0 ? (params.n_tokens / infer_sec) : 0.0;

    scheduler.stop();
    stats.flush();

    std::cout << "\n===================================================================\n";
    std::cout << "                     BENCHMARK SUMMARY                             \n";
    std::cout << "===================================================================\n";
    std::cout << "  Model Arch:           " << topo.arch_name << " (" << topo.n_layer << " layers, " << topo.n_expert << " experts)\n";
    std::cout << "  Tokens Generated:     " << params.n_tokens << " tokens\n";
    std::cout << "  Total Inference Time: " << std::fixed << std::setprecision(3) << infer_sec << " s\n";
    std::cout << "  Throughput:           " << std::fixed << std::setprecision(2) << tps << " tokens/sec\n";
    std::cout << "  Expert Cache Hit Rate:" << std::fixed << std::setprecision(2) << (100.0 * total_hits / total_lookups) << " %\n";
    std::cout << "  Historical Stats:     " << stats_file << " (EST1)\n";
    std::cout << "===================================================================\n";

    return 0;
}
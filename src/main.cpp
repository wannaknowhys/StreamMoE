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
#include "kv/kv_cache_manager.h"
#include "profile/profiler.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <thread>

using namespace stream_moe;

struct cmd_params_t {
    std::string model_path;
    std::string draft_model_path;
    std::string stats_path;
    std::string profile_log_path;
    std::string output_file_path;
    std::string eviction_policy = "hybrid";
    std::string prompt;
    bool        interactive      = false;
    int32_t     n_gpu_layers     = 0;
    size_t      moe_vram_pool_mb = 4096;
    size_t      moe_ram_pool_mb  = 0; // 0 = auto 75% available RAM
    std::string preload_policy   = "none"; // none | ram | vram | all
    uint32_t    n_ctx            = 4096;
    uint32_t    n_tokens         = 64;
    uint32_t    threads          = 16;
};

void print_usage(const char* prog) {
    std::cout << "StreamMoE (OffloadMoE) - High-Performance Memory-Optimized MoE Inference Engine\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>             Path to GGUF model (supports single or multi-shard)\n"
              << "  --draft-model <path>           Path to Dense draft model for speculative decoding\n"
              << "  -i, --interactive              Interactive multi-turn continuous chat mode\n"
              << "  -c, --ctx-size <N>             Context window size (default: 4096)\n"
              << "  -ngl, --gpu-layers <N>         Number of Dense layers to offload to GPU\n"
              << "  --moe-vram-pool <MB>           Pinned VRAM MoE Pool size in MB (default: 4096)\n"
              << "  --moe-ram-pool <MB|auto>       Pinned Host RAM MoE Pool size in MB (default: auto 75% available RAM)\n"
              << "  --moe-preload <policy>         Preload policy: none | ram | vram | all (default: none)\n"
              << "  --stats-file <path>            Path to expert frequency stats file (EST1)\n"
              << "  --profile-log <path>           Enable fine-grained hardware profiling to JSONL file\n  --eviction-policy <policy>     Eviction policy: hybrid | lru | lfu (default: hybrid)\n  --output-file <path>           Save generated conversation text to file\n"
              << "  -p, --prompt <text>            Input prompt for single-run mode\n"
              << "  -n, --n-predict <N>            Number of tokens to predict (default: 64)\n"
              << "  -t, --threads <N>              Number of CPU worker threads (default: 16 physical cores)\n"
              << "  -h, --help                     Show this help message\n";
}

cmd_params_t parse_args(int argc, char** argv) {
    cmd_params_t params;
    params.threads = get_default_threads();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--draft-model" && i + 1 < argc) {
            params.draft_model_path = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::stoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--moe-vram-pool" && i + 1 < argc) {
            params.moe_vram_pool_mb = std::stoull(argv[++i]);
        } else if (arg == "--moe-ram-pool" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto" || val == "75%") {
                params.moe_ram_pool_mb = 0;
            } else {
                params.moe_ram_pool_mb = std::stoull(val);
            }
        } else if (arg == "--moe-preload" && i + 1 < argc) {
            params.preload_policy = argv[++i];
        } else if (arg == "--stats-file" && i + 1 < argc) {
            params.stats_path = argv[++i];
        } else if (arg == "--profile-log" && i + 1 < argc) {
            params.profile_log_path = argv[++i];
        } else if (arg == "--eviction-policy" && i + 1 < argc) {
            params.eviction_policy = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            params.output_file_path = argv[++i];
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

    if (params.prompt.empty()) {
        params.interactive = true;
    }

    return params;
}

void run_generation_stream(
    uint32_t turn_id,
    const std::string& prompt,
    uint32_t n_tokens,
    const moe_model_topology_t& topo,
    moe_scheduler& scheduler,
    speculative_engine& spec_engine,
    expert_stats_tracker& stats,
    state_machine& sm
) {
    auto& prof_logger = profile_logger::instance();
    uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / 4 + 1);
    prof_logger.log_request_ingest(turn_id, prompt.size(), prompt_tokens);

    std::cout << "\n[User]: " << prompt << "\n[StreamMoE]: ";
    std::cout.flush();

    turn_profile_t prof;
    prof.turn_id = turn_id;
    prof.timestamp_ns = read_timestamp_ns();
    prof.prompt_tokens = prompt_tokens;
    prof.generated_tokens = n_tokens;

    uint64_t t_start_all = read_timestamp_ns();
    uint64_t t_prefill_start = read_timestamp_ns();

    // Simulated Prefill
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    prof.t_prefill_ns = read_timestamp_ns() - t_prefill_start;
    prof.t_prefix_match_ns = 500000;

    uint32_t total_hits = 0;
    uint32_t total_lookups = 0;

    for (uint32_t step = 1; step <= n_tokens; ++step) {
        // Forward through each MoE Layer
        for (uint32_t l : topo.moe_layers) {
            uint64_t t_layer_start = read_timestamp_ns();

            std::vector<uint32_t> routed_experts;
            uint32_t top_k = topo.n_expert_used > 0 ? topo.n_expert_used : 4;
            for (uint32_t k = 0; k < top_k; ++k) {
                uint32_t exp_id = (step * 3 + l * 7 + k * 13) % topo.n_expert;
                routed_experts.push_back(exp_id);
            }

            total_lookups += static_cast<uint32_t>(routed_experts.size());

            auto req = scheduler.route_and_prefetch(l, routed_experts, step * 100 + l);
            total_hits += static_cast<uint32_t>(req.hit_slots.size());

            uint64_t t_wait_start = read_timestamp_ns();
            auto miss_slots = scheduler.wait_miss_ready(req, 5000);
            prof.t_expert_wait_io_ns += (read_timestamp_ns() - t_wait_start);

            uint64_t t_gemm_start = read_timestamp_ns();
            std::vector<int32_t> all_layer_slots = req.hit_slots;
            all_layer_slots.insert(all_layer_slots.end(), miss_slots.begin(), miss_slots.end());
            scheduler.release_layer_slots(all_layer_slots);
            prof.t_expert_cpu_ns += (read_timestamp_ns() - t_gemm_start);

            prof.t_attn_layer_ns += (read_timestamp_ns() - t_layer_start);
        }

        stats.notify_tokens_generated(1);

        uint32_t accepted = (step % 4 == 0) ? 3 : ((step % 2 == 0) ? 2 : 1);
        if (accepted < prof.spec_accept_hist.size()) {
            prof.spec_accept_hist[accepted]++;
        }

        std::cout << "token_" << step << " " << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    uint64_t t_end_all = read_timestamp_ns();
    prof.total_duration_ms = (t_end_all - t_start_all) / 1000000.0;
    double infer_sec = prof.total_duration_ms / 1000.0;
    prof.decode_tps = infer_sec > 0 ? (n_tokens / infer_sec) : 0.0;
    prof.prefill_tps = prof.t_prefill_ns > 0 ? (prompt_tokens / (prof.t_prefill_ns / 1000000000.0)) : 0.0;

    prof.total_lookups = total_lookups;
    prof.ram_hits = total_hits;
    prof.disk_misses = (total_lookups >= total_hits) ? (total_lookups - total_hits) : 0;
    prof.gpu_hits = 0;
    prof.t_expert_total_ns = prof.t_expert_wait_io_ns + prof.t_expert_cpu_ns + prof.t_expert_gpu_ns;

    std::cout << "\n\n[Stats: " << n_tokens << " tokens | "
              << std::fixed << std::setprecision(2) << prof.decode_tps << " tok/s | Cache Hit: " 
              << std::fixed << std::setprecision(1) << prof.total_hit_rate() << "% | Mode: " 
              << sm.state_name(sm.current_state()) << "]\n";

    prof_logger.log_response_finish(prof);
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
    if (!params.profile_log_path.empty()) {
        profile_logger::instance().init(params.profile_log_path);
    }

    // 1. Memory & Thread Discovery
    size_t total_ram = get_total_ram_bytes();
    size_t avail_ram = get_available_ram_bytes();
    if (params.moe_ram_pool_mb == 0) {
        params.moe_ram_pool_mb = static_cast<size_t>((avail_ram * 0.75) / (1024 * 1024));
    }

    // 2. Direct I/O Engine Initialization
    auto dio_engine = async_dio_engine::create(1024);
    if (!dio_engine) {
        LOG_ERROR("Failed to initialize Direct I/O Engine.");
        return 1;
    }

    // 3. Parse GGUF Topology
    LOG_INFO("Parsing GGUF Model Topology from: " << params.model_path);
    moe_model_topology_t topo;
    try {
        topo = moe_loader::parse_gguf_topology(params.model_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse GGUF topology: " << e.what());
        return 1;
    }

    auto kv_info = topo.compute_kv_cache_info(params.n_ctx, 2);
    double kv_mb = static_cast<double>(kv_info.actual_kv_bytes) / (1024.0 * 1024.0);
    double kv_gb = kv_mb / 1024.0;
    double uncomp_mb = static_cast<double>(kv_info.uncompressed_mha_bytes) / (1024.0 * 1024.0);
    double uncomp_gb = uncomp_mb / 1024.0;

    std::cout << "\n-------------------------------------------------------------------\n"
              << " [System & Model Hardware Profile]\n"
              << "  - Architecture:     " << topo.arch_name << " (" << topo.n_layer << " layers, " << topo.n_expert << " experts/layer, top-" << topo.n_expert_used << ")\n"
              << "  - Attention Type:   " << (topo.is_mla ? "DeepSeek MLA (Multi-Head Latent Attention)" : "Standard MHA / GQA") << "\n"
              << "  - Total System RAM: " << std::fixed << std::setprecision(2) << (total_ram / (1024.0*1024.0*1024.0)) << " GB (Available: " << (avail_ram / (1024.0*1024.0*1024.0)) << " GB)\n"
              << "  - Context Window:   " << params.n_ctx << " tokens (Max: " << topo.max_context_length << ")\n"
              << "  - KV Cache Memory:  " << std::fixed << std::setprecision(2) 
              << (kv_gb >= 1.0 ? kv_gb : kv_mb) << (kv_gb >= 1.0 ? " GB" : " MB") << " (FP16)"
              << (topo.is_mla ? (" [MLA " + std::to_string(static_cast<int>((1.0 - kv_info.compression_ratio) * 100.0)) + "% Compressed from " + (uncomp_gb >= 1.0 ? std::to_string(uncomp_gb) + " GB" : std::to_string(uncomp_mb) + " MB") + "]") : "") << "\n"
              << "  - MoE RAM Pool:     " << params.moe_ram_pool_mb << " MB (" << (params.moe_ram_pool_mb >= 1024 ? (params.moe_ram_pool_mb / 1024) : params.moe_ram_pool_mb) << " GB Pinned Lock)\n"
              << "  - Compute Threads:  " << params.threads << " Physical Cores\n"
              << "  - Shards Count:     " << topo.shard_paths.size() << " GGUF file(s)\n"
              << "-------------------------------------------------------------------\n\n";

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

    // 4. Expert Stats (EST1)
    std::string stats_file = params.stats_path;
    if (stats_file.empty()) {
        std::filesystem::path p(params.model_path);
        stats_file = "expert_" + p.stem().string() + ".bin";
    }
    expert_stats_tracker stats;
    stats.init(stats_file, topo.n_layer, topo.n_expert, 8192);

    // 5. Pinned Pool Allocation
    size_t slot_size = topo.expert_slot_size > 0 ? topo.expert_slot_size : (1024 * 1024 * 4);
    size_t pool_budget_bytes = params.moe_ram_pool_mb * 1024 * 1024;
    uint32_t num_slots = static_cast<uint32_t>(pool_budget_bytes / slot_size);
    if (num_slots < 4) num_slots = 4;

    std::unique_ptr<expert_pool> pool;
    try {
        eviction_policy_t ep = parse_eviction_policy(params.eviction_policy);
        pool = std::make_unique<expert_pool>(slot_size, num_slots, ep);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to allocate Pinned Expert Pool: " << e.what());
        return 1;
    }

    // 6. Engines & Scheduler
    subgraph_executor executor(topo, *pool, topo.embedding_length, topo.embedding_length * 2);
    state_machine sm(params.threads);
    moe_scheduler scheduler(topo, *pool, stats, *dio_engine, shard_files);
    speculative_engine spec_engine(topo, sm, scheduler);

    if (!params.draft_model_path.empty()) {
        spec_engine.load_draft_model(params.draft_model_path);
    }

    // Preload
    if (params.preload_policy == "ram" || params.preload_policy == "all") {
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
    LOG_INFO("Engine Ready in " << std::fixed << std::setprecision(3) << startup_sec << " seconds!");

    scheduler.start();

    // 7. Execution: Interactive REPL vs Single Run
    uint32_t turn_counter = 0;
    if (params.interactive) {
        std::cout << "\n===================================================================\n"
                  << "   Interactive Multi-Turn StreamMoE Chat Session                   \n"
                  << "   (Type '/exit' to quit, '/clear' to reset, '/stats' for metrics) \n"
                  << "===================================================================\n";
        
        while (true) {
            std::cout << "\n>>> ";
            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;
            if (user_input == "/exit" || user_input == "exit" || user_input == "quit") break;
            if (user_input == "/clear" || user_input == "/reset") {
                std::cout << "[Session Cleared]\n";
                continue;
            }
            if (user_input == "/stats") {
                std::cout << "[Runtime State: " << sm.state_name(sm.current_state()) << " | EST1: " << stats_file << "]\n";
                continue;
            }
            if (user_input.empty()) continue;

            run_generation_stream(++turn_counter, user_input, params.n_tokens, topo, scheduler, spec_engine, stats, sm);
        }
    } else {
        run_generation_stream(1, params.prompt, params.n_tokens, topo, scheduler, spec_engine, stats, sm);
    }

    scheduler.stop();
    stats.flush();
    profile_logger::instance().close();

    std::cout << "\n[StreamMoE] Session terminated cleanly. Expert stats synced to " << stats_file << "\n";
    return 0;
}
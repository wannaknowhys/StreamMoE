#include "tokenizer/tokenizer.h"
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
#include "server/http_server.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <csignal>

using namespace stream_moe;

std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int sig) {
    (void)sig;
    g_shutdown_requested = true;
}

struct server_cmd_params_t {
    std::string model_path;
    std::string draft_model_path;
    std::string stats_path;
    std::string profile_log_path;
    std::string output_file_path;
    std::string eviction_policy = "hybrid";
    std::string host = "127.0.0.1";
    uint16_t    port = 8080;
    size_t      moe_ram_pool_mb = 0; // 0 = auto 75% available RAM
    uint32_t    n_ctx = 4096;
    uint32_t    threads = 16;
};

void print_server_usage(const char* prog) {
    std::cout << "StreamMoE OpenAI-Compatible API Server\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>             Path to GGUF model (supports single or multi-shard)\n"
              << "  --draft-model <path>           Path to Dense draft model for speculative decoding\n"
              << "  --host <ip>                    Host IP to bind (default: 127.0.0.1)\n"
              << "  --port <port>                  Port to bind (default: 8080)\n"
              << "  -c, --ctx-size <N>             Context window size (default: 4096)\n"
              << "  --moe-ram-pool <MB|auto>       Pinned Host RAM Pool size in MB (default: auto 75% available RAM)\n"
              << "  --stats-file <path>            Path to expert frequency stats file (EST1)\n"
              << "  --profile-log <path>           Enable fine-grained hardware profiling to JSONL file\n  --eviction-policy <policy>     Eviction policy: hybrid | lru | lfu (default: hybrid)\n  --output-file <path>           Save generated conversation text to file\n"
              << "  -t, --threads <N>              Number of CPU worker threads (default: 16 physical cores)\n"
              << "  -h, --help                     Show this help message\n";
}

server_cmd_params_t parse_server_args(int argc, char** argv) {
    server_cmd_params_t params;
    params.threads = get_default_threads();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--draft-model" && i + 1 < argc) {
            params.draft_model_path = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            params.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            params.port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--moe-ram-pool" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto" || val == "75%") {
                params.moe_ram_pool_mb = 0;
            } else {
                params.moe_ram_pool_mb = std::stoull(val);
            }
        } else if (arg == "--stats-file" && i + 1 < argc) {
            params.stats_path = argv[++i];
        } else if (arg == "--profile-log" && i + 1 < argc) {
            params.profile_log_path = argv[++i];
        } else if (arg == "--eviction-policy" && i + 1 < argc) {
            params.eviction_policy = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            params.output_file_path = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.threads = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_server_usage(argv[0]);
            std::exit(0);
        }
    }
    return params;
}

int main(int argc, char** argv) {
    std::cout << "===================================================================\n"
              << "   StreamMoE: OpenAI-Compatible Streaming API Server               \n"
              << "===================================================================\n";

    server_cmd_params_t params = parse_server_args(argc, argv);
    if (params.model_path.empty()) {
        LOG_ERROR("Missing required argument: -m <model_path>");
        print_server_usage(argv[0]);
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (!params.profile_log_path.empty()) {
        profile_logger::instance().init(params.profile_log_path);
    }

    // 1. RAM Discovery
    size_t total_ram = get_total_ram_bytes();
    size_t avail_ram = get_available_ram_bytes();
    if (params.moe_ram_pool_mb == 0) {
        params.moe_ram_pool_mb = static_cast<size_t>((avail_ram * 0.75) / (1024 * 1024));
    }

    // 2. Direct I/O Engine
    auto dio_engine = async_dio_engine::create(1024);
    if (!dio_engine) {
        LOG_ERROR("Failed to initialize Direct I/O Engine.");
        return 1;
    }

    // 3. Topology
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
              << " [Server Hardware & Model Profile]\n"
              << "  - Architecture:     " << topo.arch_name << " (" << topo.n_layer << " layers, " << topo.n_expert << " experts/layer)\n"
              << "  - Attention Type:   " << (topo.is_mla ? "DeepSeek MLA (Multi-Head Latent Attention)" : "Standard MHA / GQA") << "\n"
              << "  - Host RAM:         " << std::fixed << std::setprecision(2) << (total_ram / (1024.0*1024.0*1024.0)) << " GB (Available: " << (avail_ram / (1024.0*1024.0*1024.0)) << " GB)\n"
              << "  - Context Window:   " << params.n_ctx << " tokens\n"
              << "  - KV Cache Memory:  " << std::fixed << std::setprecision(2) 
              << (kv_gb >= 1.0 ? kv_gb : kv_mb) << (kv_gb >= 1.0 ? " GB" : " MB")
              << (topo.is_mla ? (" [MLA " + std::to_string(static_cast<int>((1.0 - kv_info.compression_ratio) * 100.0)) + "% Compressed from " + (uncomp_gb >= 1.0 ? std::to_string(uncomp_gb) + " GB" : std::to_string(uncomp_mb) + " MB") + "]") : "") << "\n"
              << "  - MoE RAM Pool:     " << params.moe_ram_pool_mb << " MB (75% Available RAM)\n"
              << "  - Compute Threads:  " << params.threads << " Physical Cores\n"
              << "  - Server Endpoint:  http://" << params.host << ":" << params.port << "\n"
              << "-------------------------------------------------------------------\n\n";

    std::vector<dio_file_t*> shard_files;
    for (const auto& shard_path : topo.shard_paths) {
        dio_file_t* f = dio_engine->open_file(shard_path);
        if (!f) {
            LOG_ERROR("Failed to open shard: " << shard_path);
            return 1;
        }
        shard_files.push_back(f);
    }

    // 4. Stats & Pool
    std::string stats_file = params.stats_path;
    if (stats_file.empty()) {
        std::filesystem::path p(params.model_path);
        stats_file = "expert_" + p.stem().string() + ".bin";
    }
    expert_stats_tracker stats;
    stats.init(stats_file, topo.n_layer, topo.n_expert, 8192);

    size_t slot_size = topo.expert_slot_size > 0 ? topo.expert_slot_size : (1024 * 1024 * 4);
    size_t pool_budget_bytes = params.moe_ram_pool_mb * 1024 * 1024;
    uint32_t num_slots = static_cast<uint32_t>(pool_budget_bytes / slot_size);
    if (num_slots < 4) num_slots = 4;

    eviction_policy_t ep = parse_eviction_policy(params.eviction_policy);
    std::unique_ptr<expert_pool> pool = std::make_unique<expert_pool>(slot_size, num_slots, ep);
    state_machine sm(params.threads);
    moe_scheduler scheduler(topo, *pool, stats, *dio_engine, shard_files);
    speculative_engine spec_engine(topo, sm, scheduler);

    if (!params.draft_model_path.empty()) {
        spec_engine.load_draft_model(params.draft_model_path);
    }

    gguf_tokenizer tokenizer;
    tokenizer.init_from_gguf(params.model_path);
    scheduler.start();

    // 5. Start HTTP Server
    server_config_t s_cfg;
    s_cfg.host = params.host;
    s_cfg.port = params.port;
    s_cfg.n_ctx = params.n_ctx;
    s_cfg.threads = params.threads;

    http_server server(s_cfg, topo, *pool, stats, scheduler, spec_engine, sm, tokenizer);
    if (!server.start()) {
        LOG_ERROR("Failed to start HTTP server.");
        scheduler.stop();
        return 1;
    }

    std::cout << "[StreamMoE Server Ready] Press Ctrl+C to stop.\n";
    while (!g_shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "\n[StreamMoE Server] Shutting down...\n";
    server.stop();
    scheduler.stop();
    stats.flush();
    profile_logger::instance().close();

    std::cout << "[StreamMoE Server] Clean shutdown complete.\n";
    return 0;
}
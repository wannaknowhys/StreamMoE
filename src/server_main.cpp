#include "common/types.h"
#include "common/logger.h"
#include "loader/moe_loader.h"
#include "engine/llama_engine.h"
#include "server/http_server.h"
#include "profile/profiler.h"

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <csignal>
#include <cstring>

using namespace stream_moe;

std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int) {
    g_shutdown_requested = true;
}

struct server_cmd_params_t {
    std::string model_path;
    std::string host = "127.0.0.1";
    uint16_t    port = 8080;
    int32_t     n_gpu_layers = 0;
    size_t      moe_ram_pool_mb = 0; // 0 = auto 75% available RAM
    bool        kv_on_gpu = false;
    uint32_t    n_ctx = 4096;
    uint32_t    n_predict = 512;
    uint32_t    threads = 16;
    float       temp = -1.0f;
    float       top_p = -1.0f;
    int32_t     top_k = -1;
    bool        use_mlock = false;
    bool        use_expert_backend = false;
    std::string profile_log_path;
};

void print_server_usage(const char* prog) {
    std::cout << "StreamMoE OpenAI-Compatible API Server (libllama deepseek4 core)\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>             Path to GGUF model (single or multi-shard)\n"
              << "  --host <ip>                    Host IP to bind (default: 127.0.0.1)\n"
              << "  --port <port>                  Port to bind (default: 8080)\n"
              << "  -c, --ctx-size <N>             Context window size (default: 4096)\n"
              << "  -ngl, --gpu-layers <N>         Layers to offload to GPU (-1 = all, default: 0)\n"
              << "  --moe-ram-pool <MB|auto>       Expert residency budget in MB (enforced by Backend.md pool; reported)\n"
              << "  --kv-placement <ram|vram>      KV cache placement (default: ram)\n"
              << "  --moe-vram-pool <MB>           VRAM budget for expert pool (Backend.md phase; logged)\n"
              << "  -n, --n-predict <N>            Max tokens per reply (default: 512)\n"
              << "  --temp <F>                     Sampling temperature override\n"
              << "  --top-p <F>                    Nucleus sampling override\n"
              << "  --top-k <N>                    Top-K override (<=0 disables)\n"
              << "  --profile-log <path>           Per-turn telemetry JSONL log\n"
              << "  -t, --threads <N>              CPU threads (default: physical cores)\n"
              << "  -h, --help                     Show this help message\n";
}

server_cmd_params_t parse_server_args(int argc, char** argv) {
    server_cmd_params_t params;
    params.threads = get_default_threads();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            params.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            params.port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::stoul(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--moe-ram-pool" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto" || val == "75%") {
                params.moe_ram_pool_mb = 0;
            } else {
                params.moe_ram_pool_mb = std::stoull(val);
            }
        } else if (arg == "--moe-vram-pool" && i + 1 < argc) {
            // Accepted for compatibility; consumed by the Backend.md GPU expert pool phase
            ++i;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--expert-backend") {
            params.use_expert_backend = true;
        } else if (arg == "--kv-placement" && i + 1 < argc) {
            std::string v = argv[++i];
            std::string vl;
            for (char c : v) vl += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            params.kv_on_gpu = (vl == "vram" || vl == "gpu");
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            params.n_predict = std::stoul(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.top_k = std::stoi(argv[++i]);
        } else if (arg == "--profile-log" && i + 1 < argc) {
            params.profile_log_path = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.threads = std::stoul(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_server_usage(argv[0]);
            std::exit(0);
        }
    }
    return params;
}

int main(int argc, char** argv) {
    std::cout << "===================================================================\n"
              << "   StreamMoE: OpenAI-Compatible Streaming API Server (real core)   \n"
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

    size_t total_ram = get_total_ram_bytes();
    size_t avail_ram = get_available_ram_bytes();
    size_t ram_pool_mb = params.moe_ram_pool_mb;
    if (ram_pool_mb == 0) {
        ram_pool_mb = static_cast<size_t>((avail_ram * 0.75) / (1024 * 1024));
    }

    LOG_INFO("Parsing GGUF topology from: " << params.model_path);
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

    uint32_t slots_total = topo.expert_slot_size > 0
        ? static_cast<uint32_t>((ram_pool_mb * 1024ULL * 1024ULL) / topo.expert_slot_size)
        : 0;

    std::cout << "\n-------------------------------------------------------------------\n"
              << " [Server Hardware & Model Profile]\n"
              << "  - Architecture:     " << topo.arch_name << " (" << topo.n_layer << " layers, "
              << topo.n_expert << " experts/layer, top-" << topo.n_expert_used << ")\n"
              << "  - Attention Type:   " << (topo.is_mla ? "DeepSeek MLA" : "Standard MHA / GQA") << "\n"
              << "  - Total System RAM: " << std::fixed << std::setprecision(2)
              << (total_ram / (1024.0*1024.0*1024.0)) << " GB (Available: "
              << (avail_ram / (1024.0*1024.0*1024.0)) << " GB)\n"
              << "  - Context Window:   " << params.n_ctx << " tokens\n"
              << "  - KV Cache Memory:  " << std::fixed << std::setprecision(2)
              << (kv_gb >= 1.0 ? kv_gb : kv_mb) << (kv_gb >= 1.0 ? " GB" : " MB")
              << " -> placement: " << (params.kv_on_gpu ? "VRAM" : "Host RAM") << "\n"
              << "  - MoE RAM Pool:     " << ram_pool_mb << " MB (weights pinned via mmap+mlock)\n"
              << "  - Expert Slots:     ~" << slots_total << "\n"
              << "  - GPU Layers:       " << params.n_gpu_layers << "\n"
              << "  - Compute Threads:  " << params.threads << "\n"
              << "  - Server Endpoint:  http://" << params.host << ":" << params.port << "\n"
              << "-------------------------------------------------------------------\n\n";

    llama_engine_params eparams;
    eparams.model_path     = params.model_path;
    eparams.n_ctx          = params.n_ctx;
    eparams.n_gpu_layers   = params.n_gpu_layers;
    eparams.ram_pool_mb    = ram_pool_mb;
    eparams.use_mlock      = params.use_mlock;
    eparams.use_expert_backend = params.use_expert_backend;
    eparams.kv_on_gpu      = params.kv_on_gpu;
    eparams.threads        = params.threads;
    eparams.temp           = params.temp;
    eparams.top_p          = params.top_p;
    eparams.top_k          = params.top_k;

    llama_engine engine;
    if (!engine.init(eparams)) {
        LOG_ERROR("Engine init failed");
        return 1;
    }

    server_config_t s_cfg;
    s_cfg.host = params.host;
    s_cfg.port = params.port;
    s_cfg.n_ctx = params.n_ctx;
    s_cfg.threads = params.threads;
    s_cfg.n_predict = params.n_predict;
    s_cfg.ram_pool_mb = ram_pool_mb;
    s_cfg.slots_total = slots_total;

    http_server server(s_cfg, topo, engine);
    if (!server.start()) {
        LOG_ERROR("Failed to start HTTP server.");
        return 1;
    }

    std::cout << "[StreamMoE Server Ready] Press Ctrl+C to stop.\n";
    while (!g_shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "\n[StreamMoE Server] Shutting down...\n";
    server.stop();
    profile_logger::instance().close();
    std::cout << "[StreamMoE Server] Clean shutdown complete.\n";
    return 0;
}

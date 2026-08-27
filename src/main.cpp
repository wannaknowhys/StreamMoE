#include "common/types.h"
#include "common/logger.h"
#include "common/crash.h"
#include "engine/llama_engine.h"

#include <iostream>
#include <string>
#include <vector>

using namespace stream_moe;

struct cmd_params_t {
    std::string model_path;
    int32_t     n_gpu_layers = 0;
    size_t      moe_ram_pool_mb = 0;
    bool        kv_on_gpu = false;
    uint32_t    n_ctx = 4096;
    uint32_t    n_tokens = 512;
    uint32_t    threads = 16;
    float       temp = -1.0f;
    float       top_p = -1.0f;
    int32_t     top_k = -1;
    bool        use_mlock = false;
    bool        use_expert_backend = false;
    std::string prompt;
    bool        interactive = false;
};

void print_usage(const char* prog) {
    std::cout << "StreamMoE - Memory-Optimized MoE Inference CLI (libllama deepseek4 core)\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <path>             Path to GGUF model (single or multi-shard)\n"
              << "  -i, --interactive              Interactive multi-turn chat REPL\n"
              << "  -p, --prompt <text>            Single-run prompt mode\n"
              << "  -c, --ctx-size <N>             Context window (default: 4096)\n"
              << "  -ngl, --gpu-layers <N>         Layers offloaded to GPU (-1 = all, default: 0)\n"
              << "  --moe-ram-pool <MB|auto>       Expert residency budget in MB (enforced by Backend.md pool; reported)\n"
              << "  --kv-placement <ram|vram>      KV cache placement (default: ram)\n"
              << "  --moe-vram-pool <MB>           VRAM expert pool budget (Backend.md phase; logged)\n"
              << "  -n, --n-predict <N>            Max tokens per reply (default: 512)\n"
              << "  --temp <F>                     Sampling temperature override\n"
              << "  --top-p <F>                    Nucleus sampling override\n"
              << "  --top-k <N>                    Top-K override (<=0 disables)\n"
              << "  -t, --threads <N>              CPU threads (default: physical cores)\n"
              << "  -h, --help                     Show this help message\n";
}

cmd_params_t parse_args(int argc, char** argv) {
    cmd_params_t params;
    params.threads = get_default_threads();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            params.prompt = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::stoul(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--moe-ram-pool" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val != "auto" && val != "75%") {
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
            params.n_tokens = std::stoul(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.top_k = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.threads = std::stoul(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }

    if (params.prompt.empty() && !params.interactive) {
        params.interactive = true;
    }
    return params;
}

static void run_turn(
    llama_engine& engine,
    std::vector<chat_msg_t>& history,
    const std::string& user_input,
    uint32_t n_predict)
{
    history.push_back({"user", user_input});
    std::cout << "\n[StreamMoE]: ";
    std::cout.flush();

    std::string reply;
    auto metrics = engine.chat(history, n_predict, [&](const char* piece, size_t len) {
        std::cout.write(piece, static_cast<std::streamsize>(len));
        std::cout.flush();
        reply.append(piece, len);
        return true;
    });
    history.push_back({"assistant", reply});

    std::cout << "\n\n[Stats: " << metrics.generated_tokens << " tok | prefill "
              << metrics.prefill_tps << " tok/s (" << metrics.prompt_tokens << " tok) | decode "
              << metrics.decode_tps << " tok/s | ctx " << metrics.ctx_used << "/" << engine.n_ctx();
    if (metrics.truncated) std::cout << " | TRUNCATED";
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    stream_moe::install_crash_handlers(); // log SEH/signal/terminate to temp\stream_moe_fatal.log
    std::cout << "===================================================================\n"
              << "   StreamMoE: MoE Inference Engine (real libllama core)            \n"
              << "===================================================================\n";

    cmd_params_t params = parse_args(argc, argv);
    if (params.model_path.empty()) {
        LOG_ERROR("Missing required argument: -m <model_path>");
        print_usage(argv[0]);
        return 1;
    }

    llama_engine_params eparams;
    eparams.model_path   = params.model_path;
    eparams.n_ctx        = params.n_ctx;
    eparams.n_gpu_layers = params.n_gpu_layers;
    eparams.ram_pool_mb  = params.moe_ram_pool_mb > 0
        ? params.moe_ram_pool_mb
        : static_cast<size_t>((get_available_ram_bytes() * 0.75) / (1024 * 1024));
    eparams.use_mlock    = params.use_mlock;
    eparams.use_expert_backend = params.use_expert_backend;
    eparams.kv_on_gpu    = params.kv_on_gpu;
    eparams.threads      = params.threads;
    eparams.temp         = params.temp;
    eparams.top_p        = params.top_p;
    eparams.top_k        = params.top_k;

    llama_engine engine;
    if (!engine.init(eparams)) {
        LOG_ERROR("Engine init failed");
        return 1;
    }

    std::vector<chat_msg_t> history;

    if (!params.prompt.empty()) {
        run_turn(engine, history, params.prompt, params.n_tokens);
        return 0;
    }

    std::cout << "\nInteractive multi-turn chat. Commands: /exit quit | /clear reset session\n";
    while (true) {
        std::cout << "\n>>> ";
        std::string user_input;
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "/exit" || user_input == "exit" || user_input == "quit") break;
        if (user_input == "/clear") {
            engine.reset();
            history.clear();
            std::cout << "[Session Cleared]\n";
            continue;
        }
        if (user_input.empty()) continue;
        run_turn(engine, history, user_input, params.n_tokens);
    }
    return 0;
}

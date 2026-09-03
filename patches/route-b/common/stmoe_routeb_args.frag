    // RX590-class Vulkan drivers report a conservative Maintenance4
    // maxBufferSize that rejects expert-pool buffers before any real
    // allocation (size check, not an OOM). Lift ggml-vulkan's cap just above
    // the requested pool so the real vkAllocateMemory heap decides. Argument
    // parsing runs before llama_backend_init, hence before the vulkan device
    // initializes and reads the env - early enough. Floor at 3GB so a small
    // pool never *shrinks* the driver default for unrelated large buffers.
    // (Pool sizes come in MB.)
#define STMOE_VK_FORCE_MAX(mb) do { \
        size_t _mb = (size_t)(mb) + 256u; \
        if (_mb < 3072u) _mb = 3072u; \
        char _buf[32]; ::snprintf(_buf, sizeof _buf, "%zu", _mb * 1024ull * 1024ull); \
        ::_putenv_s("GGML_VK_FORCE_MAX_BUFFER_SIZE", _buf); \
    } while (0)
    add_opt(common_arg(
        {"--expert-backend"},
        "StreamMoE route B: route MoE expert tensors (ffn_*_exps / ffn_*_shexp) to the stream_moe expert pool (bounded RAM)",
        [](common_params & params) {
            params.expert_backend = true;
        }
    ));
    add_opt(common_arg(
        {"--moe-ram-pool"}, "MB",
        "StreamMoE: expert residency budget in MB (0 = auto 75% free RAM)",
        [](common_params & params, int value) {
            params.moe_expert_pools.push_back("RAM:" + std::to_string(value));
        }
    ));
    add_opt(common_arg(
        {"--moe-vram-pool"}, "MB",
        "StreamMoE: VRAM budget for the expert pool (GPU pool phase)",
        [](common_params & params, int value) {
            STMOE_VK_FORCE_MAX(value);
            params.moe_expert_pools.push_back("Vulkan0:" + std::to_string(value));
        }
    ));
    add_opt(common_arg(
        {"--moe-draft-ram-pool"}, "MB",
        "StreamMoE: DRAFT model expert residency budget in MB (0 = full resident, no eviction)",
        [](common_params & params, int value) {
            params.moe_draft_expert_pools.push_back("RAM:" + std::to_string(value));
        }
    ));
    add_opt(common_arg(
        {"--moe-draft-vram-pool"}, "MB",
        "StreamMoE: VRAM budget for the DRAFT expert pool (GPU pool phase)",
        [](common_params & params, int value) {
            STMOE_VK_FORCE_MAX(value);
            params.moe_draft_expert_pools.push_back("Vulkan0:" + std::to_string(value));
        }
    ));
    add_opt(common_arg(
        {"--prompt-log"}, "PATH",
        "StreamMoE: append every /v1/chat/completions request body to this file",
        [](common_params & params, const std::string & value) {
            params.prompt_log_path = value;
        }
    ));
    add_opt(common_arg(
        {"--expert-model-files"}, "PATHS",
        "StreamMoE v2-chunk: extra strip files (semicolon-joined) beyond the main -m model file",
        [](common_params & params, const std::string & value) {
            params.expert_model_files.clear();
            size_t start = 0;
            while (start <= value.size()) {
                size_t semi = value.find(';', start);
                std::string tok = value.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
                if (!tok.empty()) params.expert_model_files.push_back(tok);
                if (semi == std::string::npos) break;
                start = semi + 1;
            }
        }
    ));
    add_opt(common_arg(
        {"--kv-placement"}, "<RAM,VRAM0,...>",
        "StreamMoE: KV cache placement as a collection (comma-separated). "
        "Multi-replica mirrors are a future feature - today only the first "
        "element is honored, extra elements warn.",
        [](common_params & params, const std::string & value) {
            params.kv_placement.clear();
            size_t start = 0;
            while (start <= value.size()) {
                size_t comma = value.find(',', start);
                std::string tok = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) params.kv_placement.push_back(tok);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
    ));

    add_opt(common_arg(
        {"--moe-expert-pools"}, "DEV:MB,...",
        "StreamMoE: MAIN model expert pools as <device>:<MB> comma list, e.g. RAM:8192,Vulkan0:5120",
        [](common_params & params, const std::string & value) {
            params.moe_expert_pools.clear();
            size_t start = 0;
            while (start <= value.size()) {
                size_t comma = value.find(',', start);
                std::string tok = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) {
                    params.moe_expert_pools.push_back(tok);
                    if (tok.rfind("RAM:", 0) != 0 && tok.rfind("CPU:", 0) != 0) {
                        const size_t cl = tok.find(':');
                        if (cl != std::string::npos) STMOE_VK_FORCE_MAX(std::strtoull(tok.c_str() + cl + 1, nullptr, 10));
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
    ));
    add_opt(common_arg(
        {"--moe-draft-expert-pools"}, "DEV:MB,...",
        "StreamMoE: DRAFT model expert pools as <device>:<MB> comma list",
        [](common_params & params, const std::string & value) {
            params.moe_draft_expert_pools.clear();
            size_t start = 0;
            while (start <= value.size()) {
                size_t comma = value.find(',', start);
                std::string tok = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) {
                    params.moe_draft_expert_pools.push_back(tok);
                    if (tok.rfind("RAM:", 0) != 0 && tok.rfind("CPU:", 0) != 0) {
                        const size_t cl = tok.find(':');
                        if (cl != std::string::npos) STMOE_VK_FORCE_MAX(std::strtoull(tok.c_str() + cl + 1, nullptr, 10));
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
    ));

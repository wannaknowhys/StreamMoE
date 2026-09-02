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
            params.moe_ram_pool_mb = value;
        }
    ));
    add_opt(common_arg(
        {"--moe-vram-pool"}, "MB",
        "StreamMoE: VRAM budget for the expert pool (GPU pool phase)",
        [](common_params & params, int value) {
            params.moe_vram_pool_mb = value;
        }
    ));
    add_opt(common_arg(
        {"--moe-draft-ram-pool"}, "MB",
        "StreamMoE: DRAFT model expert residency budget in MB (0 = full resident, no eviction)",
        [](common_params & params, int value) {
            params.moe_draft_ram_pool_mb = value;
        }
    ));
    add_opt(common_arg(
        {"--moe-draft-vram-pool"}, "MB",
        "StreamMoE: VRAM budget for the DRAFT expert pool (GPU pool phase)",
        [](common_params & params, int value) {
            params.moe_draft_vram_pool_mb = value;
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

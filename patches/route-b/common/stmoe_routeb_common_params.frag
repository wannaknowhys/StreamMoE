    // StreamMoE route B (expert pool) opt-in
    bool    expert_backend   = false; // route MoE expert tensors to the stream_moe pool
    // Expert pool spec per model: "<device>:<MB>[,...]" (device = RAM | Vulkan0 | ...).
    // MAIN model pools. Legacy --moe-ram-pool N == "RAM:N", --moe-vram-pool N == "Vulkan0:N".
    std::vector<std::string> moe_expert_pools;
    // DRAFT model pools (legacy --moe-draft-ram-pool/--moe-draft-vram-pool feed here).
    std::vector<std::string> moe_draft_expert_pools;
    std::string prompt_log_path;      // append every /v1/chat/completions request body here

    // v2-chunk strip files beyond the main -m model (c2.gguf;c3.gguf;...): passed
    // to route_b so the expert pool opens every strip file.
    std::vector<std::string> expert_model_files;

    // KV placement as a collection, e.g. "RAM,VRAM0,VRAM1" (multi-replica
    // mirrors are a future feature; today only the first element is honored)
    std::vector<std::string> kv_placement;

    // StreamMoE route B (expert pool) opt-in
    bool    expert_backend   = false; // route MoE expert tensors to the stream_moe pool
    size_t  moe_ram_pool_mb  = 0;     // MAIN model expert residency budget in MB (0 = 75% free RAM)
    size_t  moe_vram_pool_mb = 0;     // VRAM budget for the MAIN expert pool (GPU pool phase)
    size_t  moe_draft_ram_pool_mb  = 0; // DRAFT model expert RAM budget in MB (0 = full resident, no eviction)
    size_t  moe_draft_vram_pool_mb = 0; // VRAM budget for the DRAFT expert pool (GPU pool phase)
    std::string prompt_log_path;      // append every /v1/chat/completions request body here

    // KV placement as a collection, e.g. "RAM,VRAM0,VRAM1" (multi-replica
    // mirrors are a future feature; today only the first element is honored)
    std::vector<std::string> kv_placement;

// v2-chunk strip files beyond the main -m model (c2.gguf;c3.gguf;...): passed
// to route_b so the expert pool opens every strip file.
std::vector<std::string> expert_model_files;

    if (!params.kv_placement.empty()) {
        throw std::runtime_error("--kv-placement is not supported; KV follows C1 placement, use --dense-placement");
    }

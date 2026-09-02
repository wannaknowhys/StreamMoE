    // StreamMoE KV placement collection: multi-replica mirrors are not
    // implemented yet - honor only the first element, warn if more are given.
    if (!params.kv_placement.empty()) {
        if (params.kv_placement.size() > 1) {
            COM_WRN("multi-replica KV placement not implemented yet; using first element '%s'",
                    params.kv_placement.front().c_str());
        }
        const std::string & p = params.kv_placement.front();
        if (p == "RAM" || p == "ram") {
            cparams.offload_kqv = false;   // KV stays in host RAM
        } else if (p.rfind("VRAM", 0) == 0) {
            cparams.offload_kqv = true;    // KV offloaded to GPU (device idx after "VRAM")
        }
    }

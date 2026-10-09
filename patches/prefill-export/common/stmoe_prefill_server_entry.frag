    // StreamMoE: prefill-only mode - no HTTP server, just prefill + export + exit
    if (!params.prefill_from.empty()) {
        return server_prefill_only(params);
    }

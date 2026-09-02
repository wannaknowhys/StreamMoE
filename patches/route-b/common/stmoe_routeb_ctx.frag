    // StreamMoE route B: when opted in, register the expert-pool backend and
    // route MoE expert tensors (ffn_*_exps) to its weight buft before load.
    // Skip when this common_params IS the draft model (speculative.cpp binds
    // the draft to its own pool, never the main-pool budget).
    if (params.expert_backend &&
            params.model.path != params.speculative.draft.mparams.path) {
        // common_params_parse pads tensor_buft_overrides with 4096 null entries
        // (llama_params_fit scratch). Our override must sit at the FRONT - the
        // loader reads from index 0 and stops at the first null pattern.
        params.tensor_buft_overrides.clear();
        auto * ovr = stream_moe::route_b_setup(params.model.path.c_str(), params.expert_model_files, params.moe_ram_pool_mb, params.cpuparams.n_threads, false);
        if (ovr) {
            for (auto * p = ovr; p->pattern != nullptr; ++p) {
                params.tensor_buft_overrides.push_back(*p);
            }
            params.tensor_buft_overrides.push_back({ nullptr, nullptr }); // terminator
        }
    }

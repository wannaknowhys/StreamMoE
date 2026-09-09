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
        auto * ovr = stream_moe::route_b_setup(params.model.path.c_str(), params.expert_model_files, params.moe_expert_pools, params.cpuparams.n_threads, false);
        if (ovr) {
            for (auto * p = ovr; p->pattern != nullptr; ++p) {
                params.tensor_buft_overrides.push_back(*p);
            }
            params.tensor_buft_overrides.push_back({ nullptr, nullptr }); // terminator
        }

        // Dense placement (docs/DENSE_PLACEMENT.md): route B owns dense residency,
        // so the loader's contiguous -ngl is not accepted. --dense-placement
        // decides C1/C2. Dense on a GPU needs op_offload so weightless dense ops
        // follow their weights (re-enabled after --expert-backend set it off).
        if (params.n_gpu_layers != -1) {
            fprintf(stderr, "route B: -ngl/--gpu-layers is not accepted with --expert-backend; "
                            "use --dense-placement C1:<dev>,C2:<dev> instead\n");
            exit(1);
        }
        mparams.n_gpu_layers = 0;   // dense default CPU; --dense-placement overrides
        if (!params.dense_placement.empty()) {
            if (!stream_moe::route_b_dense_placement_validate(params.dense_placement.c_str())) {
                fprintf(stderr, "route B: invalid --dense-placement '%s'\n", params.dense_placement.c_str());
                exit(1);
            }
            mparams.dense_placement = params.dense_placement.c_str();
            if (stream_moe::route_b_dense_placement_uses_gpu(params.dense_placement.c_str())) {
                params.no_op_offload = false;
            }
        }
    }

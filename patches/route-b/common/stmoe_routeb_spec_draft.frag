        // StreamMoE multi-model pools (docs/MULTI_MODEL_POOL.md): the draft
        // model gets its own expert pool + buft. Point the draft load at the
        // draft pool's overrides - never reuse the main-model overrides (the
        // same _exps pattern must not route the draft's experts to the main
        // pool). ram_pool_mb==0 -> full residency (no eviction).
        if (params.expert_backend) {
            auto * ovr = stream_moe::route_b_setup(model_path.c_str(), {}, params.moe_draft_expert_pools, params.cpuparams.n_threads, true, "");
            if (ovr) {
                mparams.tensor_buft_overrides = ovr; // draft pool's own (terminated) array
            }
        }

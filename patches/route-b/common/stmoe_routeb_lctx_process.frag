        stream_moe::moe_chain_verify_graph(gf);   // privatisation premise: no external consumer of MoE-chain intermediates
        for (auto& b : backends) {
            const char* bn = ggml_backend_name(b.get());
            if (bn && std::strcmp(bn, "STREAMMOE") == 0) {
                stream_moe::moe_chain_assign_backend(gf, sched.get(), b.get());
                break;
            }
        }

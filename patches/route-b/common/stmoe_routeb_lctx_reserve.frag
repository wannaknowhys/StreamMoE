    if (model.route_b_enabled()) {
        std::vector<ggml_backend_dev_t> physical_layers;
        for (int il = 0; il < model.hparams.n_layer_all; ++il) {
            physical_layers.push_back(model.dev_layer_physical(il));
        }
        for (const auto & node : res->get_fused_nodes()) {
            stream_moe::route_b_on_node(node.tensor, node.il, model.dev_layer_physical(node.il));
        }
        for (auto & b : backends) {
            auto * dev = ggml_backend_get_device(b.get());
            if (std::strcmp(ggml_backend_dev_name(dev), "STREAMMOE") != 0) {
                stream_moe::route_b_add_device_backend(ggml_backend_dev_name(dev), b.get());
            }
        }
        stream_moe::moe_chain_verify_graph(gf);
        for (auto & b : backends) {
            if (std::strcmp(ggml_backend_name(b.get()), "STREAMMOE") == 0) {
                stream_moe::moe_chain_assign_backend(gf, sched.get(), b.get(), physical_layers,
                    split_only && (cparams.auto_fa || cparams.auto_fgdn || cparams.auto_flid || cparams.auto_fhc));
                break;
            }
        }
    }

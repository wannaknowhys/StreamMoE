    if (model.route_b_enabled()) {
        stream_moe::route_b_on_node(cur, il, il >= 0 ? model.dev_layer_physical(il) : model.dev_output());
    }

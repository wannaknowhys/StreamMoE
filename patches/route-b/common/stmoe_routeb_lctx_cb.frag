    // StreamMoE route B: official layer attribution side channel. cb() is called
    // for every named node during model.build_graph() with the model's own layer
    // index; record (tensor, il) so moe_chain_assign_backend can use the official
    // layer instead of parsing the "-<il>" name suffix (docs/LAYER_EXECUTOR_DESIGN.md 4.5).
    stream_moe::route_b_on_node(cur, il);

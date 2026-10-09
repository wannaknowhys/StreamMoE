ggml_tensor * llama_kv_cache::get_v_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].v;
}

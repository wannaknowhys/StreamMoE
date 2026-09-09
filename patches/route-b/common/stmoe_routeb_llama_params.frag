        // StreamMoE route B: dense placement spec "C1:<dev>,C2:<dev>" (GLOBAL =
        // C2 alias; RAM/CPU = host). NULL = all dense on CPU. Consumed by
        // get_layer_buft_list in llama-model.cpp (docs/DENSE_PLACEMENT.md).
        const char * dense_placement;

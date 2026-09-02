    // StreamMoE export: the LM head must fit up to n_batch output rows per
    // decode call (all prefill tokens are marked as outputs).
    params_base.n_outputs_max = std::max(params_base.n_outputs_max, params_base.n_batch);


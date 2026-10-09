    // prefill cross-validation export (env LLM_EXPORT_DIR): accumulate per-token
    // LM head input (result_norm / t_embd) and hidden state (t_h_nextn), then
    // flush everything plus the final KV cache to a file on destruction.
    std::vector<uint8_t>    export_embd;         // raw element bytes (see export_embd_type)
    std::vector<uint32_t>   export_embd_pos;
    uint32_t                export_embd_dim = 0; // per-token element count
    enum ggml_type          export_embd_type = GGML_TYPE_F32;
    std::vector<uint8_t>    export_hidden;
    std::vector<uint32_t>   export_hidden_pos;
    uint32_t                export_hidden_dim = 0;
    enum ggml_type          export_hidden_type = GGML_TYPE_F32;
    void export_prefill_final();

    // expert access history (task 1, cache-policy simulation): one record per
    // routed expert per token per layer, appended in decode order. Collected on
    // the single decode thread after each ubatch (no multithread reordering).
    // Flushed to <LLM_EXPORT_DIR>/expert_history.bin on destruction.
    std::vector<uint32_t>   export_expert_layer;
    std::vector<uint32_t>   export_expert_token;
    std::vector<uint32_t>   export_expert_id;
    uint32_t                export_token_seq = 0; // global token seq across decode calls (all computed tokens, incl. speculative candidates)
    void export_expert_history_final();

    // STREAM_MOE_TEMP: graph-eval-callback export. When LLM_EXPORT_DIR is set,
    // cparams.cb_eval is replaced by moe_export_eval_cb (original kept) which
    // captures per-token top-4 logits + logsumexp, embd/hidden, and routing ids
    // at the moment each node finishes computing (no post-decode pull).
    bool                             export_active = false;
    std::string                      export_dir;
    ggml_backend_sched_eval_callback export_prev_cb_eval = nullptr;
    void *                           export_prev_cb_eval_ud = nullptr;
    struct ggml_tensor *             export_t_embd   = nullptr; // result_norm
    struct ggml_tensor *             export_t_hidden = nullptr; // t_h_nextn
    struct ggml_tensor *             export_t_logits = nullptr; // t_logits
    std::vector<uint32_t>            export_top4_id;            // n_rows x 4
    std::vector<float>               export_top4_logit;         // n_rows x 4
    std::vector<float>               export_lse;                // logsumexp per row
    std::vector<uint32_t>            export_lse_pos;            // per-row pos
    std::vector<int32_t>             export_tokens;             // input token ids captured from graph inp_tokens (matches embd pos)
    void export_capture_embd(struct ggml_tensor * t);
    void export_capture_hidden(struct ggml_tensor * t);
    void export_capture_logits(struct ggml_tensor * t);
    void export_capture_experts(struct ggml_tensor * t);
    friend bool moe_export_eval_cb(struct ggml_tensor * t, bool ask, void * ud);

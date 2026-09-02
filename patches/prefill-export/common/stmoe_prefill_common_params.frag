    std::string export_dir;   // StreamMoE: prefill/embedding/logit export directory (replaces LLM_EXPORT_DIR env)
    std::string prefill_from; // StreamMoE: prefill-only mode - read tokens (.bin) or prompt (.txt), prefill KV, export, exit

    add_opt(common_arg(
        {"--export-dir"}, "DIR",
        "StreamMoE: prefill/embedding/logit export directory (replaces LLM_EXPORT_DIR env). "
        "When set, embd/hidden/top-4 logits/tokens are captured in-graph and flushed on exit.",
        [](common_params & params, const std::string & value) {
            params.export_dir = value;
        }
    ));
    add_opt(common_arg(
        {"--prefill-from"}, "PATH",
        "StreamMoE: prefill-only mode - read tokens (.bin = u32 id array) or plain text (.txt, "
        "auto-tokenized), prefill the KV cache, export it, then exit.",
        [](common_params & params, const std::string & value) {
            params.prefill_from = value;
        }
    ));

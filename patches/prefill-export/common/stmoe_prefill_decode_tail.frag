        // LM head input (result_norm), hidden state (t_h_nextn) and routing ids
        // are captured inside the graph via moe_export_eval_cb (set in ctor when
        // LLM_EXPORT_DIR); here we only publish the target tensor pointers for
        // the current batch and advance the global token sequence.
        if (export_active) {
            export_t_embd   = res->get_embd();
            export_t_hidden = res->get_h_nextn();
            export_t_logits = res->get_logits();
            export_token_seq += ubatch.n_tokens;
        }

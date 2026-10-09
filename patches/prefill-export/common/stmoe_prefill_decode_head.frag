    // StreamMoE export: the LM head must produce logits for EVERY prefill token
    // so exported embd/logits cover the whole sequence with no gaps. The server
    // only marks the last token; force the whole prefill batch's logits flag on
    // here (before balloc derives n_outputs / out_ids from it). Some server
    // paths (first prefill ubatch) pass a null logits array - substitute a temp
    // all-1 array. Decode batches are single-token and left alone to keep
    // server-side sampling working. batch_inp.logits is int8_t*.
    if (export_active && batch_inp.n_tokens > 1) {
        if (batch_inp.logits) {
            for (int32_t i = 0; i < batch_inp.n_tokens; i++) {
                batch_inp.logits[i] = 1;
            }
        } else {
            static std::vector<int8_t> tmp;
            tmp.assign((size_t) batch_inp.n_tokens, 1);
            const_cast<llama_batch &>(batch_inp).logits = tmp.data();
        }
    }
    // StreamMoE: capture the input token ids directly from the batch (host
    // array, reliable) so the exported sequence matches embd pos 1:1 - the
    // graph inp_tokens buffer is not host-readable after compute.
    if (export_active && batch_inp.token) {
        for (int32_t i = 0; i < batch_inp.n_tokens; i++) {
            export_tokens.push_back(batch_inp.token[i]);
        }
    }

// StreamMoE: prefill-only mode. Reads tokens (.bin = u32 id array) or plain text
// (.txt, auto-tokenized), prefills the KV cache via llama_decode, then frees the
// context (which flushes the LLM_EXPORT_DIR export) and exits without serving.
static int server_prefill_only(common_params & params) {
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        SRV_ERR("prefill-from: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    std::vector<llama_token> toks;
    const bool is_bin = params.prefill_from.size() > 4 && params.prefill_from.substr(params.prefill_from.size() - 4) == ".bin";
    if (is_bin) {
        FILE * f = std::fopen(params.prefill_from.c_str(), "rb");
        if (!f) { SRV_ERR("prefill-from: cannot open %s\n", params.prefill_from.c_str()); llama_model_free(model); return 1; }
        uint32_t n = 0;
        if (fread(&n, 4, 1, f) == 1) {
            for (uint32_t i = 0; i < n && !std::feof(f); ++i) {
                uint32_t t = 0;
                if (fread(&t, 4, 1, f) != 1) break;
                toks.push_back((llama_token) t);
            }
        }
        std::fclose(f);
    } else {
        std::string text;
        {
            FILE * f = std::fopen(params.prefill_from.c_str(), "rb");
            if (!f) { SRV_ERR("prefill-from: cannot open %s\n", params.prefill_from.c_str()); llama_model_free(model); return 1; }
            char tmp[65536];
            size_t n;
            while ((n = std::fread(tmp, 1, sizeof(tmp), f)) > 0) text.append(tmp, n);
            std::fclose(f);
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<llama_token> buf(text.size() + 8);
        const int nt = llama_tokenize(vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(), true, false);
        toks.assign(buf.begin(), buf.begin() + (nt > 0 ? nt : 0));
    }
    if (toks.empty()) { SRV_ERR("prefill-from: no tokens in %s\n", params.prefill_from.c_str()); llama_model_free(model); return 1; }

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "prefill-from: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<int8_t> lg(toks.size(), 0);
    // use a full batch with explicit pos/seq_id - llama_batch_get_one leaves
    // pos/seq_id null (auto-tracked) which hits a different batch-allocr path;
    // the server's own decode always passes a fully populated batch.
    const int32_t n_toks = (int32_t) toks.size();
    llama_batch batch = llama_batch_init(n_toks, 0, 1);
    batch.n_tokens = n_toks;
    for (int32_t i = 0; i < n_toks; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    batch.logits[n_toks - 1] = 1; // output only the last token
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) SRV_WRN("prefill-from: llama_decode returned %d\n", rc);
    SRV_WRN("prefill-from: processed %zu tokens (rc=%d)\n", toks.size(), rc);

    llama_free(ctx);        // destructor flushes prefill export (KV, embd/hidden/top-4, tokens)
    llama_model_free(model);
    return rc == 0 ? 0 : 1;
}

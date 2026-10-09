// Expert access history (task 1): one (layer, global_token, expert) triple per
// routed expert, in decode order. Written once at destruction to
// <LLM_EXPORT_DIR>/expert_history.bin ("EXPHIST1"). Read by tools/simulate_cache.js
// to evaluate cache policies / pool sizes without re-running the model.
// STREAM_MOE_TEMP: graph-eval-callback export capture (LLM_EXPORT_DIR).
static bool moe_export_eval_cb(ggml_tensor * t, bool ask, void * ud) {
    llama_context * ctx = (llama_context *) ud;
    if (ctx->export_prev_cb_eval) {
        if (!ctx->export_prev_cb_eval(t, ask, ctx->export_prev_cb_eval_ud)) return false;
    }
    if (ask) {
#ifdef STREAM_MOE_TEMP
        // The per-node byte dump wants every node computed on its own.
        if (std::getenv("STREAM_MOE_TMP_CB_HASH") || std::getenv("STREAM_MOE_TMP_BIN_DIR")) return true;
#endif
        // Only the tensors we actually capture need a separate eval. Returning
        // true for every node makes the scheduler compute node by node; under
        // whole-layer ownership that re-runs a layer once per node.
        if (t == ctx->export_t_embd || t == ctx->export_t_hidden || t == ctx->export_t_logits) return true;
        if (t && t->op == GGML_OP_MUL_MAT_ID && t->src[0] && t->src[2] && t->src[0]->name
                && std::strstr(t->src[0]->name, "_exps")) return true;
        return false;
    }
#ifdef STREAM_MOE_TEMP
    // Route-B whole-layer diagnostic (env-gated, permanent): dump every node's
    // full bytes right after it is computed. STREAM_MOE_TMP_BIN_DIR enables it;
    // ubatch subdirectories avoid overwriting across decodes. Used for offline
    // cos comparison against the whole-layer executor's own dump.
    if (std::getenv("STREAM_MOE_TMP_BIN_DIR") && t && t->name && std::strcmp(t->name, "embd") == 0) {
        stream_moe::route_b_begin_ubatch();
    }
    if ((std::getenv("STREAM_MOE_TMP_CB_HASH") || std::getenv("STREAM_MOE_TMP_BIN_DIR")) && t && t->name && t->data) {
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(ctx->get_sched(), t);
        const size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> buf(nb);
        if (b) { ggml_backend_tensor_get_async(b, t, buf.data(), 0, nb); ggml_backend_synchronize(b); }
        else   { std::memcpy(buf.data(), t->data, nb); }
        stream_moe::route_b_dump_node_bin(-1, t->name, ggml_op_name(t->op), (int) t->type,
                                          t->ne[0], t->ne[1], buf.data(), nb);
        if (std::getenv("STREAM_MOE_TMP_CB_HASH")) {
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < nb; ++i) { h ^= buf[i]; h *= 1099511628211ull; }
            double v0 = 0.0;
            if (t->type == GGML_TYPE_F32) v0 = *(const float *) buf.data();
            else if (t->type == GGML_TYPE_I32) v0 = (double) *(const int32_t *) buf.data();
            std::fprintf(stderr, "[cbhash] %-30s %-12s %016llx v0=%.6g\n", t->name,
                    ggml_op_name(t->op), (unsigned long long) h, v0);
        }
    }
#endif
    if (t == ctx->export_t_embd)   { ctx->export_capture_embd(t);   return true; }
    if (t == ctx->export_t_hidden) { ctx->export_capture_hidden(t); return true; }
    if (t == ctx->export_t_logits) { ctx->export_capture_logits(t); return true; }
    if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] && t->src[2]
            && std::strstr(t->src[0]->name, "_exps")) {
        ctx->export_capture_experts(t);
    }
    return true;
}

void llama_context::export_capture_embd(ggml_tensor * t) {
    if (!t || !t->data || t->ne[1] <= 0) return;
    const uint32_t n = (uint32_t) t->ne[1], d = (uint32_t) t->ne[0];
    const size_t esz = ggml_element_size(t);
    const size_t need  = (size_t) n * d * esz;
    const size_t base = export_embd.size();
    export_embd.resize(base + need);
    // read through the sched backend (t->data may not be host-readable mid-graph)
    std::vector<uint8_t> buf(need);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(get_sched(), t);
    ggml_backend_tensor_get_async(b, t, buf.data(), 0, need);
    ggml_backend_synchronize(b);
    std::memcpy(export_embd.data() + base, buf.data(), need);
    export_embd_dim = d;
    export_embd_type = t->type;
    for (uint32_t i = 0; i < n; ++i) export_embd_pos.push_back(export_token_seq + i);
}

void llama_context::export_capture_hidden(ggml_tensor * t) {
    if (!t || !t->data || t->ne[1] <= 0) return;
    const uint32_t n = (uint32_t) t->ne[1], d = (uint32_t) t->ne[0];
    const size_t esz = ggml_element_size(t);
    const size_t need  = (size_t) n * d * esz;
    const size_t base = export_hidden.size();
    export_hidden.resize(base + need);
    // read through the sched backend (t->data may not be host-readable mid-graph)
    std::vector<uint8_t> buf(need);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(get_sched(), t);
    ggml_backend_tensor_get_async(b, t, buf.data(), 0, need);
    ggml_backend_synchronize(b);
    std::memcpy(export_hidden.data() + base, buf.data(), need);
    export_hidden_dim = d;
    export_hidden_type = t->type;
    for (uint32_t i = 0; i < n; ++i) export_hidden_pos.push_back(export_token_seq + i);
}

void llama_context::export_capture_logits(ggml_tensor * t) {
    if (!t || !t->data || t->ne[1] <= 0) return;
    const uint32_t n = (uint32_t) t->ne[1], v = (uint32_t) t->ne[0];
    const bool f32 = (t->type == GGML_TYPE_F32);
    if (!f32 && t->type != GGML_TYPE_F16) return; // top4 scan supports f32/f16 only
    // read through the sched backend (t->data may live on a device such as
    // Vulkan0 and is not host-readable mid-graph - same as embd/hidden capture)
    const size_t need = (size_t) n * v * (f32 ? 4 : 2);
    std::vector<uint8_t> buf(need);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(get_sched(), t);
    ggml_backend_tensor_get_async(b, t, buf.data(), 0, need);
    ggml_backend_synchronize(b);
    const float * p = f32 ? (const float *) buf.data() : nullptr;
    std::vector<float> rowbuf(f32 ? 0 : v);
    for (uint32_t i = 0; i < n; ++i) {
        const float * row;
        if (f32) {
            row = p + (size_t) i * v;
        } else {
            const ggml_fp16_t * p16 = ((const ggml_fp16_t *) buf.data()) + (size_t) i * v;
            for (uint32_t j = 0; j < v; ++j) rowbuf[j] = ggml_fp16_to_fp32(p16[j]);
            row = rowbuf.data();
        }
        int32_t top4[4] = { -1, -1, -1, -1 };
        float   topv[4] = { -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
        float   maxv = -FLT_MAX;
        double  sum = 0.0;
        for (uint32_t j = 0; j < v; ++j) {
            if (row[j] > maxv) maxv = row[j];
            if (row[j] > topv[0]) {
                topv[3] = topv[2]; top4[3] = top4[2];
                topv[2] = topv[1]; top4[2] = top4[1];
                topv[1] = topv[0]; top4[1] = top4[0];
                topv[0] = row[j];  top4[0] = (int32_t) j;
            } else if (row[j] > topv[1]) {
                topv[3] = topv[2]; top4[3] = top4[2];
                topv[2] = topv[1]; top4[2] = top4[1];
                topv[1] = row[j];  top4[1] = (int32_t) j;
            } else if (row[j] > topv[2]) {
                topv[3] = topv[2]; top4[3] = top4[2];
                topv[2] = row[j];  top4[2] = (int32_t) j;
            } else if (row[j] > topv[3]) {
                topv[3] = row[j];  top4[3] = (int32_t) j;
            }
        }
        for (uint32_t j = 0; j < v; ++j) sum += std::exp((double) row[j] - maxv);
        const float lse = maxv + (float) std::log(sum);
        for (int k = 0; k < 4; ++k) {
            export_top4_id.push_back((uint32_t) top4[k]);
            export_top4_logit.push_back(topv[k]);
        }
        export_lse.push_back(lse);
        export_lse_pos.push_back(export_token_seq + i);
    }
}

void llama_context::export_capture_experts(ggml_tensor * t) {
    ggml_tensor * w0 = t->src[0], * ids = t->src[2];
    if (!ids->data || !w0->name || std::strstr(w0->name, "blk.") == nullptr) return;
    const char * blk = std::strstr(w0->name, "blk.");
    const int layer = blk ? std::atoi(blk + 4) : -1;
    const size_t isz = ggml_element_size(ids);
    // ids may live on a device backend (e.g. Vulkan0) mid-graph; read via sched.
    // route-B minigraph tensors (STREAMMOE#...) are NOT in the sched hash - in
    // that case the ids data is host-valid, so fall back to a direct copy.
    std::vector<uint8_t> raw(ggml_nbytes(ids));
    ggml_backend_t ib = ggml_backend_sched_get_tensor_backend(get_sched(), ids);
    if (ib) {
        ggml_backend_tensor_get_async(ib, ids, raw.data(), 0, raw.size());
        ggml_backend_synchronize(ib);
    } else {
        memcpy(raw.data(), ids->data, raw.size());
    }
    const int32_t * base = (const int32_t *) raw.data();
    for (int tt = 0; tt < ids->ne[1]; ++tt) {
        for (int k = 0; k < ids->ne[0]; ++k) {
            const int32_t e = base[(size_t) tt * (ids->nb[1] / isz) + (size_t) k * (ids->nb[0] / isz)];
            export_expert_layer.push_back((uint32_t) layer);
            export_expert_token.push_back(export_token_seq + (uint32_t) tt);
            export_expert_id.push_back((uint32_t) e);
        }
    }
}

void llama_context::export_expert_history_final() {
    const char * dir = export_dir.empty() ? nullptr : export_dir.c_str();
    if (!dir || export_expert_id.empty()) return;
    // dual-MoE: main model and draft (MTP) each have their own context - write
    // to per-context files so they do not overwrite each other.
    const bool is_draft = (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) || (model.arch_name() == "dflash");
    std::string path = std::string(dir) + "/expert_history" + (is_draft ? "_draft" : "_main") + ".bin";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const uint32_t n = (uint32_t) export_expert_id.size();
    fwrite("EXPHIST1", 1, 8, f);
    fwrite(&n, 4, 1, f);
    for (uint32_t i = 0; i < n; ++i) {
        fwrite(&export_expert_layer[i], 4, 1, f);
        fwrite(&export_expert_token[i], 4, 1, f);
        fwrite(&export_expert_id[i],    4, 1, f);
    }
    std::fclose(f);
}

// Prefill cross-validation export (env LLM_EXPORT_DIR): flushed once from the
// destructor. Writes per-token LM head input (result_norm / t_embd) and hidden
// state (t_h_nextn), accumulated during decode, plus the final per-layer KV
// tensors of every dsv4 sub-cache. Binary "PREFEXP1". Compared by tools/verify_prefill.js.
void llama_context::export_prefill_final() {
    const char * dir = export_dir.empty() ? nullptr : export_dir.c_str();
    if (!dir) return;
    std::filesystem::create_directories(export_dir);
    // dual-MoE: separate main / draft prefill exports
    const bool is_draft = (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) || (model.arch_name() == "dflash");
    std::string path = std::string(dir) + "/prefill_export" + (is_draft ? "_draft" : "_main") + ".bin";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return;

    const uint32_t n_embd_rows = (uint32_t) export_embd_pos.size();
    const uint32_t n_embd_dim  = export_embd_dim;
    const uint32_t n_hid_rows  = (uint32_t) export_hidden_pos.size();
    const uint32_t n_hid_dim   = export_hidden_dim;
    const size_t   esz_embd    = ggml_type_size(export_embd_type);
    const size_t   esz_hidden  = ggml_type_size(export_hidden_type);

    fwrite("PREFEXP2", 1, 8, f);
    fwrite(&n_embd_rows, 4, 1, f);
    fwrite(&n_embd_dim, 4, 1, f);
    fwrite(&export_embd_type, 4, 1, f);
    for (uint32_t i = 0; i < n_embd_rows; ++i) {
        fwrite(&export_embd_pos[i], 4, 1, f);
        fwrite(export_embd.data() + (size_t) i * n_embd_dim * esz_embd, 1, (size_t) n_embd_dim * esz_embd, f);
    }
    fwrite(&n_hid_rows, 4, 1, f);
    fwrite(&n_hid_dim, 4, 1, f);
    fwrite(&export_hidden_type, 4, 1, f);
    for (uint32_t i = 0; i < n_hid_rows; ++i) {
        fwrite(&export_hidden_pos[i], 4, 1, f);
        fwrite(export_hidden.data() + (size_t) i * n_hid_dim * esz_hidden, 1, (size_t) n_hid_dim * esz_hidden, f);
    }

    std::vector<std::pair<llama_kv_cache *, const char *>> caches;
    llama_memory_i * mem = get_memory();
    llama_kv_cache_dsv4 * dsv4 = dynamic_cast<llama_kv_cache_dsv4 *>(mem);
    llama_kv_cache_iswa * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem);
    llama_kv_cache * kv_gen = dynamic_cast<llama_kv_cache *>(mem);
    if (std::getenv("STREAM_MOE_DBG")) fprintf(stderr, "[dbg] kv export: memory_type=%s dsv4=%d iswa=%d kv=%d\n", typeid(*mem).name(), dsv4 != nullptr, iswa != nullptr, kv_gen != nullptr);
    if (dsv4) {
        // DeepSeek dsv4: multiple sub-caches (raw_base/raw_swa/csa/hca/lid)
        caches.push_back({dsv4->get_raw()->get_base(), "raw_base"});
        caches.push_back({dsv4->get_raw()->get_swa (), "raw_swa" });
        caches.push_back({dsv4->get_csa(), "csa"});
        caches.push_back({dsv4->get_hca(), "hca"});
        caches.push_back({dsv4->get_lid(), "lid"});
    } else if (iswa) {
        // interleaved SWA (gemma4): base + swa sub-caches
        caches.push_back({iswa->get_base(), "base"});
        caches.push_back({iswa->get_swa (), "swa" });
    } else {
        // generic: any llama_kv_cache (all attention KV derive from it)
        if (kv_gen) {
            caches.push_back({kv_gen, "kv"});
        }
    }
    uint32_t n_cache = (uint32_t) caches.size();
    fwrite(&n_cache, 4, 1, f);
    for (auto & c : caches) {
        llama_kv_cache * kc = c.first;
        const char * name   = c.second;
        uint32_t nl = (uint32_t) std::strlen(name);
        fwrite(&nl, 4, 1, f);
        fwrite(name, 1, nl, f);
        std::vector<uint32_t> lids = kc->get_layer_ids();
        uint32_t n_layer = (uint32_t) lids.size();
        fwrite(&n_layer, 4, 1, f);
        for (uint32_t il : lids) {
            fwrite(&il, 4, 1, f);
            ggml_tensor * tk = kc->get_k_storage(il);
            ggml_tensor * tv = kc->get_v_storage(il);
            for (ggml_tensor * t : {tk, tv}) {
                uint64_t nb = t ? (uint64_t) ggml_nbytes(t) : 0;
                if (std::getenv("STREAM_MOE_DBG")) fprintf(stderr, "[dbg] kv t=%p nb=%llu cache=%s layer=%u\n", (void*) t, (unsigned long long) nb, name, il);
                if (t && (!t->data || nb > 1073741824ull)) { t = nullptr; nb = 0; }
                uint32_t ty = t ? (uint32_t) t->type : 0;
                // tensor layout (ne/nb) so readers can slice per-token
                const int64_t * ne4 = t ? t->ne : nullptr;
                const size_t  * nb4 = t ? t->nb : nullptr;
                for (int i = 0; i < 4; i++) { const int64_t v = ne4 ? ne4[i] : 0; fwrite(&v, 8, 1, f); }
                for (int i = 0; i < 4; i++) { const int64_t v = nb4 ? (int64_t) nb4[i] : 0; fwrite(&v, 8, 1, f); }
                fwrite(&ty, 4, 1, f);
                fwrite(&nb, 8, 1, f);
                if (t && nb > 0) {
                    std::vector<uint8_t> buf((size_t) nb);
                    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(get_sched(), t);
                    ggml_backend_tensor_get_async(b, t, buf.data(), 0, nb);
                    ggml_backend_synchronize(b);
                    fwrite(buf.data(), 1, nb, f);
                }
            }
        }
    }
    // STREAM_MOE_TEMP: top-4 logits + logsumexp per output row, then sampled
    // token ids (decode). Appended after KV so PREFEXP1 readers stay compatible.
    {
        const uint32_t n_top4 = (uint32_t) export_lse_pos.size();
        const uint32_t k_top4 = 4;
        fwrite(&n_top4, 4, 1, f);
        fwrite(&k_top4, 4, 1, f);
        for (uint32_t i = 0; i < n_top4; ++i) {
            fwrite(&export_lse_pos[i], 4, 1, f);
            for (int k = 0; k < 4; ++k) {
                fwrite(&export_top4_id[i * 4 + k], 4, 1, f);
            }
            for (int k = 0; k < 4; ++k) {
                fwrite(&export_top4_logit[i * 4 + k], 4, 1, f);
            }
            fwrite(&export_lse[i], 4, 1, f);
        }
    }
    // StreamMoE: input token ids (captured from graph inp_tokens) + detokenized
    // text. Sequence matches export_embd pos 1:1 (prefix-matching sequence).
    if (!export_tokens.empty()) {
        const std::string d = export_dir;
        const std::string tp = d + "/tokens_id.bin";
        FILE * tf = std::fopen(tp.c_str(), "wb");
        if (tf) {
            const uint32_t n = (uint32_t) export_tokens.size();
            fwrite(&n, 4, 1, tf);
            for (auto t : export_tokens) fwrite(&t, 4, 1, tf);
            std::fclose(tf);
        }
        const std::string ttp = d + "/tokens_text.txt";
        FILE * ttf = std::fopen(ttp.c_str(), "wb");
        if (ttf) {
            char buf[256];
            const llama_vocab * vocab = llama_model_get_vocab(&model);
            for (auto t : export_tokens) {
                const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
                if (n > 0) fwrite(buf, 1, (size_t) n, ttf);
            }
            std::fclose(ttf);
        }
    }
    // prefill_meta.json: describe the bins (dtype/layout) so readers (verify)
    // never hardcode element sizes.
    {
        FILE * mf = std::fopen((std::string(dir) + "/prefill_meta.json").c_str(), "w");
        if (mf) {
            std::string m;
            m += "{\n";
            m += "  \"format\": \"PREFEXP2\",\n";
            m += "  \"model\": \"" + std::string(model.arch_name()) + "\",\n";
            m += "  \"files\": {\n";
            m += "    \"prefill_export_main.bin\": {\n";
            m += "      \"byte_order\": \"little\",\n";
            m += "      \"sections\": [\n";
            m += "        { \"id\": \"embd\", \"dtype\": \"" + std::string(ggml_type_name(export_embd_type)) + "\", \"n\": " + std::to_string(n_embd_rows) + ", \"dim\": " + std::to_string(n_embd_dim) + " },\n";
            m += "        { \"id\": \"hidden\", \"dtype\": \"" + std::string(ggml_type_name(export_hidden_type)) + "\", \"n\": " + std::to_string(n_hid_rows) + ", \"dim\": " + std::to_string(n_hid_dim) + " },\n";
            m += "        { \"id\": \"top4\", \"id_dtype\": \"u32\", \"logit_dtype\": \"f32\", \"lse_dtype\": \"f32\", \"n\": " + std::to_string((uint32_t) export_lse_pos.size()) + " }\n";
            m += "      ]\n";
            m += "    },\n";
            m += "    \"expert_history_main.bin\": { \"record\": [\"layer u32\", \"token u32\", \"expert u32\"], \"n\": " + std::to_string((uint32_t) export_expert_id.size()) + " }\n";
            m += "  }\n";
            m += "}\n";
            fwrite(m.data(), 1, m.size(), mf);
            std::fclose(mf);
        }
    }
    std::fclose(f);
}

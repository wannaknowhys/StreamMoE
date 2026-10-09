        // STREAM_MOE: publish export target pointers BEFORE compute - the eval
        // callback runs during graph compute and must match against them.
        if (export_active) {
            export_t_embd   = res->get_embd();
            export_t_hidden = res->get_h_nextn();
            export_t_logits = res->get_logits();
        }
#ifdef STREAM_MOE_ROUTE_B
        // Whole-layer ownership reuses arena slots inside a layer, so the eval
        // callback's post-split read of an observed tensor can land on an
        // overwritten slot. Tell route B to retain every tensor the export reads
        // (never reuse their slot); an empty list (not exporting) clears the set.
        std::vector<const ggml_tensor *> export_observed;
        if (export_active) {
            export_observed.push_back(export_t_embd);
            export_observed.push_back(export_t_hidden);
            export_observed.push_back(export_t_logits);
            for (int i = 0; i < gf->n_nodes; ++i) {
                const ggml_tensor * nd = gf->nodes[i];
                if (nd && nd->op == GGML_OP_MUL_MAT_ID && nd->src[0] && nd->src[2]
                        && nd->src[0]->name && std::strstr(nd->src[0]->name, "_exps"))
                    export_observed.push_back(nd->src[2]);
            }
        }
        stream_moe::route_b_set_export_retained(export_observed);
#endif

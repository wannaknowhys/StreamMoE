    // STREAM_MOE_TEMP: when exporting, replace cb_eval with the graph-eval
    // capture callback (original kept and chained). Captures embd/hidden/top-4
    // logits/routing ids right after each node computes.
    export_dir = params.export_dir;
    if (!export_dir.empty()) {
        export_active            = true;
        export_prev_cb_eval      = params.cb_eval;
        export_prev_cb_eval_ud   = params.cb_eval_user_data;
        cparams.cb_eval          = moe_export_eval_cb;
        cparams.cb_eval_user_data = this;
    }

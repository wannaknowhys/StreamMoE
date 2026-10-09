    // StreamMoE: graceful shutdown endpoint - flushes prefill exports and
    // destructs contexts (used by the std/moe export orchestration).
    ctx_http.post("/shutdown", ex_wrapper([](const server_http_req &) -> server_http_res_ptr {
        llama_server_terminate();
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data = "ok";
        return res;
    }));

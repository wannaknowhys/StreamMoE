#pragma once

#include "engine/llama_engine.h"
#include "loader/moe_loader.h"

#include <mutex>
#include <string>
#include <thread>
#include <atomic>

namespace stream_moe {

struct server_config_t {
    std::string host = "127.0.0.1";
    uint16_t    port = 8080;
    uint32_t    n_ctx = 4096;
    uint32_t    threads = 16;
    uint32_t    n_predict = 512;
    size_t      ram_pool_mb = 0;   // reported via /stats
    uint32_t    slots_total = 0;   // derived from topology + pool budget
    std::string prompt_log_path;   // if set, append every /v1/chat/completions request body here
};

class http_server {
public:
    http_server(const server_config_t& config, const moe_model_topology_t& topo, llama_engine& engine);
    ~http_server();

    bool start();
    void stop();
    bool is_running() const { return running_; }

private:
    void listener_loop();
    void handle_client(uintptr_t client_socket);

    server_config_t             config_;
    const moe_model_topology_t& topo_;
    llama_engine&               engine_;

    std::mutex                  infer_mutex_;   // one inference at a time
    std::atomic<uint64_t>       total_turns_{0};
    std::atomic<bool>           running_{false};
    uintptr_t                   listen_socket_ = 0;
    std::thread                 listener_thread_;
};

} // namespace stream_moe

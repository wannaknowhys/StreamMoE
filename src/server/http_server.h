#pragma once

#include "common/types.h"
#include "loader/moe_loader.h"
#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "scheduler/moe_scheduler.h"
#include "engine/subgraph_executor.h"
#include "engine/state_machine.h"
#include "engine/speculative_engine.h"

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <functional>

namespace stream_moe {

struct server_config_t {
    std::string host = "127.0.0.1";
    uint16_t    port = 8080;
    uint32_t    n_ctx = 4096;
    uint32_t    threads = 16;
};

class http_server {
public:
    http_server(
        const server_config_t& config,
        const moe_model_topology_t& topo,
        expert_pool& pool,
        expert_stats_tracker& stats,
        moe_scheduler& scheduler,
        speculative_engine& spec_engine,
        state_machine& sm
    );
    ~http_server();

    // Disable copy/move
    http_server(const http_server&) = delete;
    http_server& operator=(const http_server&) = delete;

    // Start listening and serving HTTP/SSE requests
    bool start();

    // Stop HTTP server
    void stop();

    bool is_running() const { return running_; }

private:
    void listener_loop();
    void handle_client(uintptr_t client_socket);

    server_config_t             config_;
    const moe_model_topology_t& topo_;
    expert_pool&                pool_;
    expert_stats_tracker&       stats_;
    moe_scheduler&              scheduler_;
    speculative_engine&         spec_engine_;
    state_machine&              sm_;

    std::atomic<bool>           running_{false};
    std::thread                 listener_thread_;
    uintptr_t                   listen_socket_ = 0;
};

} // namespace stream_moe
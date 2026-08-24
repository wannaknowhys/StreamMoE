#include "server/http_server.h"
#include "common/logger.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <chrono>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

namespace stream_moe {

namespace {

void send_response(SOCKET sock, int status_code, const std::string& status_text, const std::string& content_type, const std::string& body) {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
       << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
       << "Connection: close\r\n\r\n"
       << body;
    std::string resp = ss.str();
    send(sock, resp.c_str(), static_cast<int>(resp.size()), 0);
}

} // namespace

http_server::http_server(
    const server_config_t& config,
    const moe_model_topology_t& topo,
    expert_pool& pool,
    expert_stats_tracker& stats,
    moe_scheduler& scheduler,
    speculative_engine& spec_engine,
    state_machine& sm
) : config_(config),
    topo_(topo),
    pool_(pool),
    stats_(stats),
    scheduler_(scheduler),
    spec_engine_(spec_engine),
    sm_(sm) {}

http_server::~http_server() {
    stop();
}

bool http_server::start() {
    if (running_.exchange(true)) return true;

#if defined(_WIN32)
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        LOG_ERROR("WSAStartup failed");
        running_ = false;
        return false;
    }
#endif

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        LOG_ERROR("Failed to create listening socket");
        running_ = false;
        return false;
    }

    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);
    inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr);

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        LOG_ERROR("Failed to bind server socket to " << config_.host << ":" << config_.port);
        closesocket(s);
        running_ = false;
        return false;
    }

    if (listen(s, 64) == SOCKET_ERROR) {
        LOG_ERROR("Failed to listen on socket");
        closesocket(s);
        running_ = false;
        return false;
    }

    listen_socket_ = static_cast<uintptr_t>(s);
    listener_thread_ = std::thread(&http_server::listener_loop, this);

    LOG_INFO("StreamMoE OpenAI API Server listening on http://" << config_.host << ":" << config_.port);
    return true;
}

void http_server::stop() {
    if (!running_.exchange(false)) return;

    if (listen_socket_ != 0) {
        closesocket(static_cast<SOCKET>(listen_socket_));
        listen_socket_ = 0;
    }

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

#if defined(_WIN32)
    WSACleanup();
#endif
    LOG_INFO("StreamMoE HTTP Server stopped.");
}

void http_server::listener_loop() {
    while (running_) {
        sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        SOCKET client_sock = accept(static_cast<SOCKET>(listen_socket_), reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_sock == INVALID_SOCKET) {
            if (!running_) break;
            continue;
        }

        std::thread([this, client_sock]() {
            handle_client(static_cast<uintptr_t>(client_sock));
        }).detach();
    }
}

void http_server::handle_client(uintptr_t client_socket) {
    SOCKET sock = static_cast<SOCKET>(client_socket);
    char buf[4096];
    int bytes_read = recv(sock, buf, sizeof(buf) - 1, 0);
    if (bytes_read <= 0) {
        closesocket(sock);
        return;
    }
    buf[bytes_read] = '\0';
    std::string req(buf);

    // Basic HTTP parsing
    std::istringstream stream(req);
    std::string method, path, version;
    stream >> method >> path >> version;

    if (method == "OPTIONS") {
        send_response(sock, 200, "OK", "text/plain", "");
        closesocket(sock);
        return;
    }

    // 1. GET /health
    if (path == "/health") {
        std::string json = "{\"status\":\"ok\",\"engine\":\"StreamMoE\",\"arch\":\"" + topo_.arch_name + "\"}";
        send_response(sock, 200, "OK", "application/json", json);
    }
    // 2. GET /v1/models
    else if (path == "/v1/models") {
        std::string json = "{\"object\":\"list\",\"data\":[{\"id\":\"" + topo_.arch_name + "\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"stream_moe\"}]}";
        send_response(sock, 200, "OK", "application/json", json);
    }
    // 3. GET /stats or GET /metrics
    else if (path == "/stats" || path == "/metrics") {
        std::ostringstream ss;
        ss << "{\n"
           << "  \"model\": \"" << topo_.arch_name << "\",\n"
           << "  \"layers\": " << topo_.n_layer << ",\n"
           << "  \"experts_per_layer\": " << topo_.n_expert << ",\n"
           << "  \"slots_total\": " << pool_.num_slots() << ",\n"
           << "  \"slot_size_kb\": " << (pool_.slot_size() / 1024) << ",\n"
           << "  \"pool_total_mb\": " << (pool_.total_bytes() / (1024 * 1024)) << ",\n"
           << "  \"runtime_state\": \"" << sm_.state_name(sm_.current_state()) << "\",\n"
           << "  \"speculative_k\": " << spec_engine_.get_active_k() << "\n"
           << "}";
        send_response(sock, 200, "OK", "application/json", ss.str());
    }
    // 4. POST /v1/chat/completions
    else if (path == "/v1/chat/completions" || path == "/v1/completions") {
        bool is_stream = req.find("\"stream\": true") != std::string::npos || req.find("\"stream\":true") != std::string::npos;

        if (is_stream) {
            // Server-Sent Events (SSE) Streaming Response
            std::string header = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: text/event-stream\r\n"
                                 "Cache-Control: no-cache\r\n"
                                 "Connection: close\r\n"
                                 "Access-Control-Allow-Origin: *\r\n\r\n";
            send(sock, header.c_str(), static_cast<int>(header.size()), 0);

            for (int step = 1; step <= 8; ++step) {
                std::ostringstream ss;
                ss << "data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\""
                   << topo_.arch_name << "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" token_" << step << "\"},\"finish_reason\":null}]}\n\n";
                std::string sse_chunk = ss.str();
                send(sock, sse_chunk.c_str(), static_cast<int>(sse_chunk.size()), 0);
            }

            std::string done_msg = "data: [DONE]\n\n";
            send(sock, done_msg.c_str(), static_cast<int>(done_msg.size()), 0);
        } else {
            // Non-streaming JSON response
            std::string json = "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"created\":1700000000,\"model\":\"" +
                               topo_.arch_name +
                               "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"StreamMoE: Extreme Memory Offload Engine is operational.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":12,\"total_tokens\":22}}";
            send_response(sock, 200, "OK", "application/json", json);
        }
    }
    else {
        send_response(sock, 404, "Not Found", "application/json", "{\"error\":\"Not Found\"}");
    }

    closesocket(sock);
}

} // namespace stream_moe
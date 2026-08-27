#include "server/http_server.h"
#include "profile/profiler.h"
#include "common/logger.h"

#include <nlohmann/json.hpp>

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

using json = nlohmann::json;

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

bool send_all(SOCKET sock, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        int n = send(sock, data + sent, static_cast<int>(len - sent), 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

// recv until the full HTTP request (headers + Content-Length body) is available
bool recv_request(SOCKET sock, std::string& out) {
    char buf[8192];
    size_t header_end = std::string::npos;
    size_t content_len = 0;
    while (true) {
        int n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) return !out.empty();
        out.append(buf, static_cast<size_t>(n));

        if (header_end == std::string::npos) {
            header_end = out.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                // parse Content-Length
                std::string lower = out.substr(0, header_end);
                for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                size_t cl = lower.find("content-length:");
                if (cl != std::string::npos) {
                    content_len = std::strtoull(out.c_str() + cl + 15, nullptr, 10);
                }
            }
        }
        if (header_end != std::string::npos) {
            size_t body_bytes = out.size() - (header_end + 4);
            if (body_bytes >= content_len) return true;
        }
        if (out.size() > (64ULL << 20)) return false; // sanity cap 64MB
    }
}

} // namespace

http_server::http_server(const server_config_t& config, const moe_model_topology_t& topo, llama_engine& engine)
    : config_(config),
      topo_(topo),
      engine_(engine) {}

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
        try {
            handle_client(static_cast<uintptr_t>(client_sock));
        } catch (const std::exception& e) {
            // record the exception reason (e.g. bad_alloc / malformed UTF-8) before
            // it becomes a hard crash; stack trace is appended by the crash handler
            LOG_ERROR("client handler exception: " << e.what());
            FILE* pf = std::fopen("temp/stream_moe_fatal.log", "ab");
            if (pf) {
                char ts[32];
                const std::time_t now = std::time(nullptr);
                std::tm tmv;
                localtime_s(&tmv, &now);
                std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);
                std::fprintf(pf, "[%s] HANDLER EXCEPTION: %s\n", ts, e.what());
                std::fclose(pf);
            }
            // reply 400 so the client sees an error instead of an abrupt ECONNRESET
            try {
                json err;
                err["error"] = e.what();
                send_response(client_sock, 400, "Bad Request", "application/json", err.dump());
            } catch (...) {}
            closesocket(client_sock);
        }
    }
}

void http_server::handle_client(uintptr_t client_socket) {
    SOCKET sock = static_cast<SOCKET>(client_socket);
    std::string req;
    if (!recv_request(sock, req)) {
        closesocket(sock);
        return;
    }

    std::istringstream stream(req);
    std::string method, path, version;
    stream >> method >> path >> version;

    if (method == "OPTIONS") {
        send_response(sock, 200, "OK", "text/plain", "");
        closesocket(sock);
        return;
    }

    if (path == "/health") {
        json j;
        j["status"] = "ok";
        j["engine"] = "StreamMoE";
        j["arch"] = topo_.arch_name;
        j["real_inference"] = true;
        send_response(sock, 200, "OK", "application/json", j.dump());
    }
    else if (path == "/v1/models") {
        json j;
        j["object"] = "list";
        json item;
        item["id"] = topo_.arch_name;
        item["object"] = "model";
        item["created"] = 1700000000;
        item["owned_by"] = "stream_moe";
        j["data"] = json::array({item});
        send_response(sock, 200, "OK", "application/json", j.dump());
    }
    else if (path == "/stats" || path == "/metrics") {
        json j;
        j["model"] = topo_.arch_name;
        j["model_name"] = engine_.model_name();
        j["layers"] = topo_.n_layer;
        j["experts_per_layer"] = topo_.n_expert;
        j["slots_total"] = config_.slots_total;
        j["slot_size_kb"] = topo_.expert_slot_size / 1024;
        j["pool_total_mb"] = config_.ram_pool_mb;
        j["runtime_state"] = "REAL_INFERENCE_LIBLLAMA";
        j["kv_on_gpu"] = engine_.kv_on_gpu();
        j["total_turns"] = total_turns_.load();
        j["ctx_used"] = 0;
        j["n_ctx"] = config_.n_ctx;
        send_response(sock, 200, "OK", "application/json", j.dump());
    }
    else if (path == "/v1/chat/completions" || path == "/v1/completions") {
        size_t body_pos = req.find("\r\n\r\n");
        if (body_pos == std::string::npos) {
            send_response(sock, 400, "Bad Request", "application/json", "{\"error\":\"missing body\"}");
            closesocket(sock);
            return;
        }
        std::string body = req.substr(body_pos + 4);

        // optional prompt backup: append every request body to a file
        if (!config_.prompt_log_path.empty()) {
            FILE* pf = std::fopen(config_.prompt_log_path.c_str(), "ab");
            if (pf) {
                char ts[32];
                const std::time_t now = std::time(nullptr);
                std::tm tmv;
                localtime_s(&tmv, &now);
                std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);
                std::fprintf(pf, "=== %s ===\n%s\n", ts, body.c_str());
                std::fclose(pf);
            }
        }

        json request;
        try {
            request = json::parse(body);
        } catch (const std::exception&) {
            send_response(sock, 400, "Bad Request", "application/json", "{\"error\":\"invalid JSON\"}");
            closesocket(sock);
            return;
        }

        std::vector<chat_msg_t> messages;
        if (request.contains("messages") && request["messages"].is_array()) {
            for (const auto& m : request["messages"]) {
                chat_msg_t msg;
                msg.role = m.value("role", "user");
                msg.content = m.value("content", "");
                messages.push_back(std::move(msg));
            }
        } else if (request.contains("prompt")) {
            messages.push_back({"user", request.value("prompt", "")});
        }
        if (messages.empty()) {
            send_response(sock, 400, "Bad Request", "application/json", "{\"error\":\"no messages\"}");
            closesocket(sock);
            return;
        }

        bool is_stream = request.value("stream", false);
        uint32_t n_predict = config_.n_predict;
        if (request.contains("max_tokens") && request["max_tokens"].is_number_unsigned()) {
            n_predict = std::min(n_predict, request["max_tokens"].get<uint32_t>());
        }

        uint32_t turn_id = static_cast<uint32_t>(++total_turns_);
        auto& prof_logger = profile_logger::instance();

        std::lock_guard<std::mutex> lock(infer_mutex_);

        prof_logger.log_request_ingest(turn_id, body.size(), 0);

        auto t_start = std::chrono::steady_clock::now();
        llama_turn_metrics metrics;

        if (is_stream) {
            std::string header =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n"
                "Access-Control-Allow-Origin: *\r\n\r\n";
            if (!send_all(sock, header.c_str(), header.size())) {
                closesocket(sock);
                return;
            }

            bool aborted = false;
            metrics = engine_.chat(messages, n_predict, [&](const char* piece, size_t len) {
                json chunk;
                chunk["id"] = "chatcmpl-streammoe";
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = 1700000000;
                chunk["model"] = topo_.arch_name;
                json choice;
                choice["index"] = 0;
                choice["delta"]["content"] = std::string(piece, len);
                choice["finish_reason"] = nullptr;
                chunk["choices"] = json::array({choice});
                std::string sse = "data: " + chunk.dump() + "\n\n";
                return send_all(sock, sse.c_str(), sse.size());
            });
            (void)aborted;

            const char* done = "data: [DONE]\n\n";
            send_all(sock, done, std::strlen(done));
        } else {
            std::string content;
            metrics = engine_.chat(messages, n_predict, [&](const char* piece, size_t len) {
                content.append(piece, len);
                return true;
            });

            json resp;
            resp["id"] = "chatcmpl-streammoe";
            resp["object"] = "chat.completion";
            resp["created"] = 1700000000;
            resp["model"] = topo_.arch_name;
            json choice;
            choice["index"] = 0;
            choice["message"]["role"] = "assistant";
            choice["message"]["content"] = content;
            choice["finish_reason"] = metrics.truncated ? "length" : "stop";
            resp["choices"] = json::array({choice});
            resp["usage"]["prompt_tokens"] = metrics.prompt_tokens;
            resp["usage"]["completion_tokens"] = metrics.generated_tokens;
            resp["usage"]["total_tokens"] = metrics.prompt_tokens + metrics.generated_tokens;
            std::string out = resp.dump();
            send_response(sock, 200, "OK", "application/json", out);
        }

        auto t_end = std::chrono::steady_clock::now();

        turn_profile_t prof;
        prof.turn_id = turn_id;
        prof.timestamp_ns = read_timestamp_ns();
        prof.prompt_tokens = metrics.prompt_tokens;
        prof.generated_tokens = metrics.generated_tokens;
        prof.total_duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double decode_sec = metrics.decode_ms / 1000.0;
        prof.decode_tps = decode_sec > 0 ? metrics.generated_tokens / decode_sec : 0.0;
        prof.prefill_tps = metrics.prefill_tps;
        prof.t_prefill_ns = static_cast<uint64_t>(metrics.prefill_ms * 1000000.0);
        prof.t_prefix_match_ns = 0;
        LOG_INFO("[turn " << turn_id << "] prompt=" << metrics.prompt_tokens
                 << " tok, gen=" << metrics.generated_tokens
                 << " tok, prefill=" << metrics.prefill_ms << " ms ("
                 << metrics.prefill_tps << " tok/s), decode="
                 << metrics.decode_ms << " ms (" << prof.decode_tps << " tok/s)"
                 << ", ctx_used=" << metrics.ctx_used);
        prof_logger.log_response_finish(prof);
    }
    else {
        send_response(sock, 404, "Not Found", "application/json", "{\"error\":\"Not Found\"}");
    }

    closesocket(sock);
}

} // namespace stream_moe

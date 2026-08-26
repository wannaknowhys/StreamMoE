#pragma once

// Real inference engine core backed by the vendored llama.cpp (deepseek4 support).
// Replaces the former mock generation loop: token IDs come from real model logits.

#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace stream_moe {

struct chat_msg_t {
    std::string role;
    std::string content;
};

struct llama_engine_params {
    std::string model_path;

    uint32_t n_ctx           = 4096;
    int32_t  n_gpu_layers    = 0;    // -1 = all layers to GPU
    size_t   ram_pool_mb     = 0;    // informational budget (Backend.md slot pool phase enforces it)
    bool     use_mlock       = false;// force-pin weights in RAM; requires model < available RAM
    bool     kv_on_gpu       = false;// offload KQV ops + KV cache to GPU
    uint32_t threads         = 16;

    // Sampling; negative values resolve from GGUF metadata defaults
    float    temp            = -1.0f;
    float    top_p           = -1.0f;
    int32_t  top_k           = -1;   // <=0 disables top-k
    uint32_t seed            = LLAMA_DEFAULT_SEED;

    // Route B opt-in: route MoE expert tensors (ffn_*_exps / ffn_*_shexp) to the
    // stream_moe expert-pool backend. Dense tensors keep llama.cpp defaults.
    // Default OFF - the backend's graph_compute delegation is not wired yet.
    bool     use_expert_backend = false;
};

struct llama_turn_metrics {
    uint32_t prompt_tokens    = 0;
    uint32_t generated_tokens = 0;
    double   prefill_ms       = 0.0;
    double   decode_ms        = 0.0;
    double   prefill_tps      = 0.0;
    double   decode_tps       = 0.0;
    uint32_t ctx_used         = 0;   // KV cells in use after the turn
    bool     truncated        = false;
};

class llama_engine {
public:
    llama_engine() = default;
    ~llama_engine();

    llama_engine(const llama_engine&) = delete;
    llama_engine& operator=(const llama_engine&) = delete;

    bool init(const llama_engine_params& p);

    // Multi-turn chat turn: `messages` must include the latest user message.
    // The assistant reply is NOT appended by the engine - caller owns history.
    // on_token receives streamed pieces; return false from callback to abort.
    llama_turn_metrics chat(
        const std::vector<chat_msg_t>& messages,
        uint32_t max_tokens,
        const std::function<bool(const char*, size_t)>& on_token);

    void reset(); // clears KV cache and prompt token cache

    // Metadata accessors
    std::string model_name() const { return model_name_; }
    std::string arch_name() const { return arch_name_; }
    uint32_t    n_ctx() const { return n_ctx_; }
    uint32_t    vocab_size() const { return llama_vocab_n_tokens(vocab_); }
    float       resolved_temp() const { return temp_; }
    float       resolved_top_p() const { return top_p_; }
    bool        kv_on_gpu() const { return kv_on_gpu_; }

private:
    std::vector<llama_token> tokenize_prompt(const std::string& text);
    int64_t common_prefix_len(const std::vector<llama_token>& a, const std::vector<llama_token>& b) const;
    bool decode_tokens(const std::vector<llama_token>& tokens, int64_t pos_begin, bool need_logits_last);

    llama_model*            model_   = nullptr;
    llama_context*          ctx_     = nullptr;
    const llama_vocab*      vocab_   = nullptr;
    llama_sampler*          sampler_ = nullptr;

    std::string             chat_template_;
    std::string             model_name_;
    std::string             arch_name_;

    uint32_t                n_ctx_        = 4096;
    uint32_t                n_batch_      = 2048;
    bool                    kv_on_gpu_    = false;
    float                   temp_         = 0.3f;
    float                   top_p_        = 0.95f;
    int32_t                 top_k_        = 0;

    // Tokenized state of everything currently resident in KV (prompt + generated)
    std::vector<llama_token> cache_tokens_;
};

} // namespace stream_moe

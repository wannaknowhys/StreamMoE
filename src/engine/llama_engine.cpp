#include "engine/llama_engine.h"
#include "common/logger.h"

#include <chrono>
#include <cstring>
#include <algorithm>

namespace stream_moe {

llama_engine::~llama_engine() {
    if (sampler_) llama_sampler_free(sampler_);
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

static std::string read_meta_str(const llama_model* model, const char* key) {
    // Two-pass: query required length first - templates can exceed any fixed buffer
    int32_t need = llama_model_meta_val_str(model, key, nullptr, 0);
    if (need <= 0) return "";
    std::string buf(static_cast<size_t>(need), '\0');
    int32_t n = llama_model_meta_val_str(model, key, buf.data(), static_cast<int32_t>(need + 1));
    if (n <= 0) return "";
    buf.resize(static_cast<size_t>(std::min<int32_t>(n, need)));
    return buf;
}

static float parse_float_or(const std::string& s, float fallback) {
    if (s.empty()) return fallback;
    try {
        size_t pos = 0;
        return std::stof(s, &pos);
    } catch (...) {
        return fallback;
    }
}

bool llama_engine::init(const llama_engine_params& p) {
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            LOG_INFO("[llama] " << text);
        }
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p.n_gpu_layers;
    // Repack extra bufts fully copy every repack-eligible tensor (Q4_K/Q5_K/Q6_K/Q2_K)
    // into private buffers at load - fatal for >RAM models like UD-Q8_K_XL (162GB).
    // Disabling keeps all weights zero-copy over mmap (OS page cache streams them).
    mparams.use_extra_bufts = false;
    // Explicit mlock only for models that fit in RAM; mmap stays the default so
    // expert weights stream from NVMe through the page cache.
    if (p.use_mlock) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    }

    LOG_INFO("Loading model (this streams all shards once): " << p.model_path);
    auto t0 = std::chrono::steady_clock::now();
    struct progress_ctx { std::chrono::steady_clock::time_point last; } pctx{std::chrono::steady_clock::now()};
    mparams.progress_callback_user_data = &pctx;
    mparams.progress_callback = [](float progress, void* ud) {
        auto* ctx = static_cast<progress_ctx*>(ud);
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - ctx->last).count() >= 2.0) {
            ctx->last = now;
            LOG_INFO("[load] " << static_cast<int>(progress * 100.0f) << "%");
        }
        return true;
    };
    model_ = llama_model_load_from_file(p.model_path.c_str(), mparams);
    if (!model_) {
        LOG_ERROR("llama_model_load_from_file failed for " << p.model_path);
        return false;
    }
    double load_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    LOG_INFO("Model loaded in " << load_sec << " s");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = p.n_ctx;
    cparams.n_batch         = std::min<uint32_t>(p.n_ctx, 2048);
    cparams.n_ubatch        = std::min<uint32_t>(cparams.n_batch, 512);
    cparams.n_threads       = static_cast<int32_t>(p.threads);
    cparams.n_threads_batch = static_cast<int32_t>(p.threads);
    cparams.offload_kqv     = p.kv_on_gpu;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        LOG_ERROR("llama_init_from_model failed");
        return false;
    }

    vocab_ = llama_model_get_vocab(model_);

    chat_template_ = read_meta_str(model_, "tokenizer.chat_template");
    arch_name_     = read_meta_str(model_, "general.architecture");
    model_name_    = read_meta_str(model_, "general.name");
    if (arch_name_.empty()) arch_name_ = "unknown";

    // Resolve sampling defaults: CLI override > GGUF metadata > hardcoded fallback
    temp_  = (p.temp  >= 0.0f) ? p.temp
           : parse_float_or(read_meta_str(model_, "general.sampling.temp"), 0.3f);
    top_p_ = (p.top_p >= 0.0f) ? p.top_p
           : parse_float_or(read_meta_str(model_, "general.sampling.top_p"), 0.95f);
    top_k_ = p.top_k;

    n_ctx_     = p.n_ctx;
    n_batch_   = cparams.n_batch;
    kv_on_gpu_ = p.kv_on_gpu;

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    if (temp_ > 0.0f) {
        if (top_k_ > 0) llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(top_k_));
        llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p_, 1));
        llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temp_));
        llama_sampler_chain_add(sampler_, llama_sampler_init_dist(p.seed));
    } else {
        llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
    }

    LOG_INFO("Engine ready: arch=" << arch_name_ << ", ctx=" << n_ctx_
             << ", threads=" << p.threads << ", ngl=" << p.n_gpu_layers
             << ", kv_on_gpu=" << kv_on_gpu_
             << ", temp=" << temp_ << ", top_p=" << top_p_);
    return true;
}

void llama_engine::reset() {
    llama_memory_clear(llama_get_memory(ctx_), 0);
    cache_tokens_.clear();
}

std::vector<llama_token> llama_engine::tokenize_prompt(const std::string& text) {
    int32_t n = llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, false, true);
    if (n < 0) n = -n;
    if (n == 0) return {};
    std::vector<llama_token> tokens(static_cast<size_t>(n));
    if (llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                       tokens.data(), n, false, true) < 0) {
        return {};
    }
    return tokens;
}

int64_t llama_engine::common_prefix_len(const std::vector<llama_token>& a, const std::vector<llama_token>& b) const {
    int64_t i = 0;
    while (i < static_cast<int64_t>(a.size()) && i < static_cast<int64_t>(b.size()) && a[i] == b[i]) {
        ++i;
    }
    return i;
}

bool llama_engine::decode_tokens(const std::vector<llama_token>& tokens, int64_t pos_begin, bool need_logits_last) {
    llama_batch batch = llama_batch_init(n_batch_, 0, 1);
    // seq_id slots are pre-allocated by llama_batch_init - only fill values,
    // never replace the pointers (llama_batch_free owns them)
    for (uint32_t bi = 0; bi < n_batch_; ++bi) {
        batch.n_seq_id[bi] = 1;
        if (batch.seq_id[bi]) batch.seq_id[bi][0] = 0;
    }

    bool ok = true;
    size_t idx = 0;
    while (idx < tokens.size()) {
        batch.n_tokens = 0;
        size_t end = std::min(tokens.size(), idx + static_cast<size_t>(n_batch_));
        for (size_t k = idx; k < end; ++k) {
            int32_t bi = batch.n_tokens++;
            batch.token [bi] = tokens[k];
            batch.pos   [bi] = pos_begin + static_cast<int64_t>(k);
            batch.logits[bi] = need_logits_last && (k == end - 1) && (end == tokens.size());
        }
        if (llama_decode(ctx_, batch) != 0) {
            LOG_ERROR("llama_decode failed during prefill at position "
                      << (pos_begin + static_cast<int64_t>(idx)));
            ok = false;
        }
        idx = end;
        if (!ok) break;
    }

    llama_batch_free(batch);
    return ok;
}

llama_turn_metrics llama_engine::chat(
    const std::vector<chat_msg_t>& messages,
    uint32_t max_tokens,
    const std::function<bool(const char*, size_t)>& on_token)
{
    llama_turn_metrics metrics;

    // 1. Apply the model's chat template over the full conversation
    std::vector<llama_chat_message> msgs;
    msgs.reserve(messages.size());
    for (const auto& m : messages) {
        msgs.push_back({m.role.c_str(), m.content.c_str()});
    }

    int32_t formatted_len = llama_chat_apply_template(
        chat_template_.c_str(), msgs.data(), msgs.size(), true, nullptr, 0);
    if (formatted_len < 0) {
        LOG_ERROR("llama_chat_apply_template sizing failed");
        return metrics;
    }
    std::string formatted(static_cast<size_t>(formatted_len), '\0');
    formatted_len = llama_chat_apply_template(
        chat_template_.c_str(), msgs.data(), msgs.size(), true,
        formatted.data(), static_cast<int32_t>(formatted.size()));
    if (formatted_len < 0) {
        LOG_ERROR("llama_chat_apply_template render failed");
        return metrics;
    }
    formatted.resize(static_cast<size_t>(formatted_len));

    // 2. Tokenize with special-token parsing; ensure BOS exactly once at position 0
    std::vector<llama_token> prompt_tokens = tokenize_prompt(formatted);
    if (prompt_tokens.empty()) {
        LOG_ERROR("tokenize produced no tokens");
        return metrics;
    }
    if (llama_vocab_get_add_bos(vocab_) && prompt_tokens[0] != llama_vocab_bos(vocab_)) {
        prompt_tokens.insert(prompt_tokens.begin(), llama_vocab_bos(vocab_));
    }
    metrics.prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());

    // 3. KV reuse: keep the common prefix, drop the divergent tail from cache
    int64_t reuse_len = common_prefix_len(cache_tokens_, prompt_tokens);
    llama_memory_t mem = llama_get_memory(ctx_);
    if (reuse_len < static_cast<int64_t>(cache_tokens_.size())) {
        llama_memory_seq_rm(mem, 0, reuse_len, -1);
    }
    cache_tokens_.resize(static_cast<size_t>(reuse_len));

    // 4. Prefill the suffix (full cache hit: replay last token to obtain fresh logits)
    auto t_prefill0 = std::chrono::steady_clock::now();
    std::vector<llama_token> suffix(prompt_tokens.begin() + reuse_len, prompt_tokens.end());
    if (!suffix.empty()) {
        if (!decode_tokens(suffix, reuse_len, true)) {
            reset();
            return metrics;
        }
    } else {
        int64_t last_pos = static_cast<int64_t>(cache_tokens_.size()) - 1;
        llama_token last_token = cache_tokens_.back();
        llama_memory_seq_rm(mem, 0, last_pos, -1);
        cache_tokens_.pop_back();
        std::vector<llama_token> one = { last_token };
        if (!decode_tokens(one, last_pos, true)) {
            reset();
            return metrics;
        }
        cache_tokens_.push_back(last_token);
    }
    auto t_prefill1 = std::chrono::steady_clock::now();
    metrics.prefill_ms = std::chrono::duration<double, std::milli>(t_prefill1 - t_prefill0).count();

    // 5. Decode loop with real sampling
    auto t_decode0 = std::chrono::steady_clock::now();
    int64_t next_pos = static_cast<int64_t>(prompt_tokens.size());
    uint32_t produced = 0;
    while (produced < max_tokens) {
        if (next_pos + 1 >= static_cast<int64_t>(n_ctx_)) {
            metrics.truncated = true;
            break;
        }
        llama_token next = llama_sampler_sample(sampler_, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, next)) break;

        char piece_buf[96];
        int32_t n = llama_token_to_piece(vocab_, next, piece_buf, sizeof(piece_buf), 0, true);
        if (n > 0 && on_token) {
            if (!on_token(piece_buf, static_cast<size_t>(n))) break;
        }

        cache_tokens_.push_back(next);
        ++produced;
        ++next_pos;
        if (produced >= max_tokens) break;

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens   = 1;
        batch.token[0]   = next;
        batch.pos[0]     = next_pos - 1;
        batch.n_seq_id[0] = 1;
        if (batch.seq_id[0]) batch.seq_id[0][0] = 0;
        batch.logits[0]  = true;
        int32_t rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            LOG_ERROR("llama_decode failed during generation");
            break;
        }
    }
    auto t_decode1 = std::chrono::steady_clock::now();
    metrics.decode_ms = std::chrono::duration<double, std::milli>(t_decode1 - t_decode0).count();

    metrics.generated_tokens = produced;
    metrics.ctx_used = static_cast<uint32_t>(llama_memory_seq_pos_max(mem, 0) + 1);
    if (metrics.prefill_ms > 0) {
        metrics.prefill_tps = metrics.prompt_tokens / (metrics.prefill_ms / 1000.0);
    }
    if (metrics.decode_ms > 0) {
        metrics.decode_tps = produced / (metrics.decode_ms / 1000.0);
    }
    return metrics;
}

} // namespace stream_moe

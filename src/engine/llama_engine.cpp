#include "engine/llama_engine.h"
#include "llama-ext.h"
#include "common/types.h"
#include "common/logger.h"
#include "backend/moe_backend.h"
#include "backend/scheduler.h"
#include "io/async_dio.h"
#include "loader/moe_loader.h"
#include "../diagnostics/trace_dump.h"

#include <cstdlib>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace stream_moe {

llama_engine::llama_engine() = default;

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

static ggml_type parse_cache_type(const std::string& s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    if (s == "q6_k") return GGML_TYPE_Q6_K;
    return GGML_TYPE_F16;
}

bool llama_engine::init(const llama_engine_params& p) {
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            LOG_INFO("[llama] " << text);
        }
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p.n_gpu_layers;

    // Route B: when opted in, register the stream_moe backend and point the
    // standard MoE expert tensor patterns at our expert-pool buft. Patterns are
    // model-agnostic (llama.cpp MoE schema); dense models have no matching
    // tensors so this is a no-op for them. Dense weights stay on defaults.
    static llama_model_tensor_buft_override moe_overrides[] = {
        {"blk\\..*\\.ffn_.*_exps\\.weight", nullptr},
        // shared experts (ffn_*_shexp) use plain MUL_MAT; routed to the pool in a
        // later milestone once the backend supports it.
        {nullptr, nullptr},
    };
    if (p.use_expert_backend) {
        stream_moe_register_backend();
        ggml_backend_buffer_type_t eb = stream_moe_register_backend_helper_expert_buft();
        if (eb) {
            moe_overrides[0].buft = eb;
            mparams.tensor_buft_overrides = moe_overrides;
            LOG_INFO("Expert backend enabled: routed MoE exps tensors -> stream_moe pool (shared experts stay default)");
        } else {
            LOG_WARN("Expert backend requested but stream_moe buft unavailable - using llama.cpp defaults");
        }
    }
    // into private buffers at load - fatal for >RAM models like UD-Q8_K_XL (162GB).
    // Disabling keeps all weights zero-copy over mmap (OS page cache streams them).
    mparams.use_extra_bufts = false;
    // Explicit mlock only for models that fit in RAM; mmap stays the default so
    // expert weights stream from NVMe through the page cache.
    if (p.use_mlock) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    }

    // Route B expert pool: parse topology, open shards via DIO, allocate the
    // bounded pool, and hand the scheduler to the backend before model load.
    if (p.use_expert_backend) {
        topo_ = std::make_unique<moe_model_topology_t>(moe_loader::parse_gguf_topology(p.model_path));
        dio_ = async_dio_engine::create(1024);
        for (const auto& shard : topo_->shard_paths) {
            dio_file_t* f = dio_->open_file(shard);
            if (!f) {
                LOG_ERROR("expert backend: cannot DIO-open shard " << shard);
                return false;
            }
            shard_files_.push_back(f);
        }
        size_t pool_bytes = p.ram_pool_mb > 0
            ? p.ram_pool_mb * 1024ull * 1024ull
            : (get_available_ram_bytes() * 3 / 4);
        scheduler_ = std::make_unique<expert_scheduler>();
        if (!scheduler_->init(*topo_, *dio_, shard_files_, pool_bytes)) return false;
        scheduler_->start();
        stream_moe_backend_set_scheduler(scheduler_.get());
        stream_moe_backend_set_threads(static_cast<int>(p.threads));
        LOG_INFO("Expert pool active: " << (pool_bytes / (1024ull * 1024ull * 1024ull))
                 << " GB cap, " << scheduler_->num_slots() << " slots");
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
    cparams.type_k          = parse_cache_type(p.cache_type);
    cparams.type_v          = cparams.type_k; // deepseek4/MLA requires type_k == type_v
    cparams.swa_full        = p.swa_full;

#ifdef STREAM_MOE_TEMP
    {
        const char* tf = std::getenv("STREAM_MOE_TRACE_FILE");
        static FILE* g_trace = std::fopen(tf ? tf : "stream_moe_trace.bin", "wb");
        if (g_trace) {
            cparams.cb_eval = stream_moe_trace_cb;
            cparams.cb_eval_user_data = g_trace;
            LOG_INFO("TRACE: per-layer tensor dump enabled -> " << (tf ? tf : "stream_moe_trace.bin"));
        } else {
            LOG_WARN("TRACE: cannot open trace file");
        }
    }
#endif

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

size_t llama_engine::kv_memory_bytes() const {
    if (!ctx_) return 0;
    size_t total = 0;
    for (const auto& [buft, mb] : llama_get_memory_breakdown(ctx_)) {
        (void) buft;
        total += mb.context;
    }
    return total;
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
            // when LLM_EXPORT_DIR is set (prefill cross-validation), mark every
            // token as output so the LM head input (result_norm) covers all tokens
            const bool export_all = std::getenv("LLM_EXPORT_DIR") != nullptr;
            batch.logits[bi] = export_all || (need_logits_last && (k == end - 1) && (end == tokens.size()));
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

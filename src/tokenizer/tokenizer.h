#pragma once

#include "common/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace stream_moe {

struct token_info_t {
    std::string text;
    float       score = 0.0f;
    int32_t     type  = 1; // 1 = normal, 2 = unknown, 3 = control, 4 = user_defined, 6 = byte
};

class gguf_tokenizer {
public:
    gguf_tokenizer() = default;
    ~gguf_tokenizer() = default;

    // Initialize vocabulary and merge rules from GGUF file
    bool init_from_gguf(const std::string& gguf_path);

    // Initialize standalone fallback vocabulary (standard GPT-2 / LLaMA / DeepSeek compatible)
    void init_fallback();

    // Tokenize UTF-8 string (English, Chinese, Code) into token IDs
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos = false) const;

    // Detokenize a single token ID to UTF-8 text chunk
    std::string detokenize(int32_t token_id) const;

    // Detokenize an array of token IDs to full UTF-8 text string
    std::string detokenize(const std::vector<int32_t>& tokens) const;

    // Accessors
    bool is_initialized() const { return initialized_; }
    size_t vocab_size() const { return id_to_token_.size(); }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }

private:
    bool                                  initialized_ = false;
    int32_t                               bos_id_ = 1;
    int32_t                               eos_id_ = 2;
    std::vector<token_info_t>             id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<uint64_t, int32_t> bpe_ranks_; // pair_hash -> rank
};

} // namespace stream_moe
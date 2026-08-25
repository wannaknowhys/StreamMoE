#include "tokenizer/tokenizer.h"
#include "common/logger.h"
#include "gguf.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace stream_moe {

namespace {

uint64_t make_pair_key(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

std::string format_byte_token(uint8_t byte) {
    std::ostringstream ss;
    ss << "<0x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte) << ">";
    return ss.str();
}

// GPT-2 / LLaMA byte-to-unicode reverse mapping
std::unordered_map<std::string, uint8_t> create_unicode_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    // Direct printable ASCII
    for (int b = L'!'; b <= L'~'; ++b) {
        map[std::string(1, static_cast<char>(b))] = static_cast<uint8_t>(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        // UTF-8 encoding of 0x00A1 - 0x00AC
        std::string s;
        s += static_cast<char>(0xC2);
        s += static_cast<char>(b);
        map[s] = static_cast<uint8_t>(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        std::string s;
        s += static_cast<char>(b < 0xC0 ? 0xC2 : 0xC3);
        s += static_cast<char>(b < 0xC0 ? b : (b - 0x40));
        map[s] = static_cast<uint8_t>(b);
    }

    // Remapped control bytes
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= L'!' && b <= L'~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            continue;
        }
        uint32_t cp = 256 + n;
        n++;
        // UTF-8 encode cp (256..383 -> 0xC4..0xC5)
        std::string s;
        if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        map[s] = static_cast<uint8_t>(b);
    }
    return map;
}

const std::unordered_map<std::string, uint8_t>& get_unicode_to_byte_map() {
    static const auto map = create_unicode_to_byte_map();
    return map;
}

} // namespace

bool gguf_tokenizer::init_from_gguf(const std::string& gguf_path) {
    gguf_init_params params = { true, nullptr };
    gguf_context* ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) {
        LOG_WARN("Could not open GGUF for tokenizer from " << gguf_path << ", using fallback.");
        init_fallback();
        return false;
    }

    int64_t key_tokens = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (key_tokens < 0) {
        LOG_WARN("tokenizer.ggml.tokens not found in GGUF metadata, using fallback.");
        gguf_free(ctx);
        init_fallback();
        return false;
    }

    size_t n_tokens = gguf_get_arr_n(ctx, key_tokens);
    id_to_token_.resize(n_tokens);
    token_to_id_.reserve(n_tokens);

    for (size_t i = 0; i < n_tokens; ++i) {
        const char* str = gguf_get_arr_str(ctx, key_tokens, i);
        std::string s = str ? str : "";
        id_to_token_[i].text = s;
        token_to_id_[s] = static_cast<int32_t>(i);
    }

    // BOS / EOS
    int64_t key_bos = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (key_bos >= 0) bos_id_ = gguf_get_val_u32(ctx, key_bos);

    int64_t key_eos = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (key_eos >= 0) eos_id_ = gguf_get_val_u32(ctx, key_eos);

    // Merge rules
    int64_t key_merges = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (key_merges >= 0) {
        size_t n_merges = gguf_get_arr_n(ctx, key_merges);
        for (size_t m = 0; m < n_merges; ++m) {
            const char* merge_str = gguf_get_arr_str(ctx, key_merges, m);
            if (!merge_str) continue;
            std::string line(merge_str);
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = line.substr(0, space_pos);
                std::string second = line.substr(space_pos + 1);
                auto it1 = token_to_id_.find(first);
                auto it2 = token_to_id_.find(second);
                if (it1 != token_to_id_.end() && it2 != token_to_id_.end()) {
                    uint64_t pair_key = make_pair_key(it1->second, it2->second);
                    bpe_ranks_[pair_key] = static_cast<int32_t>(m);
                }
            }
        }
    }

    gguf_free(ctx);
    initialized_ = true;
    LOG_INFO("Initialized GGUF Tokenizer: " << id_to_token_.size() << " tokens, " 
             << bpe_ranks_.size() << " merges, BOS=" << bos_id_ << ", EOS=" << eos_id_);
    return true;
}

void gguf_tokenizer::init_fallback() {
    id_to_token_.clear();
    token_to_id_.clear();
    bpe_ranks_.clear();

    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        id_to_token_.push_back({s, 0.0f, 1});
        token_to_id_[s] = i;
    }

    bos_id_ = 1;
    eos_id_ = 2;
    initialized_ = true;
    LOG_INFO("Initialized Fallback Byte Tokenizer (" << id_to_token_.size() << " base tokens)");
}

std::vector<int32_t> gguf_tokenizer::tokenize(const std::string& text, bool add_bos) const {
    if (!initialized_ || text.empty()) return {};

    std::vector<int32_t> tokens;
    if (add_bos) tokens.push_back(bos_id_);

    // Greedy forward maximal matching with byte-fallback
    size_t i = 0;
    while (i < text.size()) {
        int32_t matched_id = -1;
        size_t matched_len = 0;

        size_t max_lookahead = std::min(text.size() - i, static_cast<size_t>(64));
        for (size_t len = max_lookahead; len >= 1; --len) {
            std::string sub = text.substr(i, len);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                matched_id = it->second;
                matched_len = len;
                break;
            }
        }

        if (matched_id >= 0) {
            tokens.push_back(matched_id);
            i += matched_len;
        } else {
            uint8_t byte = static_cast<uint8_t>(text[i]);
            std::string byte_tok = format_byte_token(byte);
            auto bit = token_to_id_.find(byte_tok);
            if (bit != token_to_id_.end()) {
                tokens.push_back(bit->second);
            } else if (byte < id_to_token_.size()) {
                tokens.push_back(static_cast<int32_t>(byte));
            }
            i += 1;
        }
    }

    return tokens;
}

std::string gguf_tokenizer::detokenize(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(id_to_token_.size())) {
        return "";
    }
    const std::string& s = id_to_token_[token_id].text;

    // Handle byte token <0xXX>
    if (s.size() == 6 && s.rfind("<0x", 0) == 0 && s.back() == '>') {
        unsigned int byte_val = 0;
        std::stringstream ss;
        ss << std::hex << s.substr(3, 2);
        if (ss >> byte_val) {
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    // Standard LLaMA/GPT space markers (' ' or 'Ġ')
    std::string raw_bytes;
    const auto& u2b = get_unicode_to_byte_map();

    size_t i = 0;
    while (i < s.size()) {
        if (static_cast<unsigned char>(s[i]) == 0xe2 && i + 2 < s.size() &&
            static_cast<unsigned char>(s[i+1]) == 0x96 && static_cast<unsigned char>(s[i+2]) == 0x81) {
            raw_bytes += ' ';
            i += 3;
            continue;
        }
        if (static_cast<unsigned char>(s[i]) == 0xc4 && i + 1 < s.size() &&
            static_cast<unsigned char>(s[i+1]) == 0xa0) {
            raw_bytes += ' ';
            i += 2;
            continue;
        }

        // Check if multi-byte utf8 character maps to single raw byte
        bool mapped = false;
        for (size_t len = 4; len >= 1; --len) {
            if (i + len <= s.size()) {
                std::string sub = s.substr(i, len);
                auto it = u2b.find(sub);
                if (it != u2b.end()) {
                    raw_bytes += static_cast<char>(it->second);
                    i += len;
                    mapped = true;
                    break;
                }
            }
        }
        if (!mapped) {
            raw_bytes += s[i];
            i += 1;
        }
    }

    return raw_bytes;
}

std::string gguf_tokenizer::detokenize(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t t : tokens) {
        if (t == eos_id_ || t == bos_id_) continue;
        result += detokenize(t);
    }
    return result;
}

} // namespace stream_moe
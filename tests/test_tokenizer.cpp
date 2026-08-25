#include "tokenizer/tokenizer.h"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace stream_moe;

void test_tokenizer_live_gguf() {
    std::cout << "[Test 1] Testing GGUF Tokenizer Initialization & Tokenization...\n";

    std::string model_path = "N:/AI_LLM/DeepSeek-V4-Flash-0731/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf";
    gguf_tokenizer tok;
    
    if (std::filesystem::exists(model_path)) {
        bool ok = tok.init_from_gguf(model_path);
        assert(ok);
        assert(tok.vocab_size() > 1000);
        std::cout << "[INFO] Successfully loaded real GGUF vocabulary: " << tok.vocab_size() << " tokens.\n";
    } else {
        tok.init_fallback();
    }

    // English Test
    std::string en_text = "Hello, StreamMoE! High-Performance MoE Engine.";
    auto en_tokens = tok.tokenize(en_text);
    assert(!en_tokens.empty());
    std::string en_decoded = tok.detokenize(en_tokens);
    std::cout << "[EN Original]: " << en_text << "\n";
    std::cout << "[EN Tokens]  : " << en_tokens.size() << " tokens (";
    for (size_t i = 0; i < std::min(en_tokens.size(), (size_t)8); ++i) std::cout << en_tokens[i] << " ";
    std::cout << "...)\n";
    std::cout << "[EN Decoded] : " << en_decoded << "\n";
    assert(en_decoded.find("StreamMoE") != std::string::npos);

    // Chinese Test
    std::string zh_text = "你好，StreamMoE！这是针对极限显存卸载与70GB内存池的实时推理测试。";
    auto zh_tokens = tok.tokenize(zh_text);
    assert(!zh_tokens.empty());
    std::string zh_decoded = tok.detokenize(zh_tokens);
    std::cout << "\n[ZH Original]: " << zh_text << "\n";
    std::cout << "[ZH Tokens]  : " << zh_tokens.size() << " tokens\n";
    std::cout << "[ZH Decoded] : " << zh_decoded << "\n";
    assert(zh_decoded.find("StreamMoE") != std::string::npos);

    std::cout << "[+] test_tokenizer_live_gguf PASSED!\n";
}

int main() {
    std::cout << "===========================================\n"
              << "  Running StreamMoE Tokenizer Tests        \n"
              << "===========================================\n";
    test_tokenizer_live_gguf();
    std::cout << "===========================================\n"
              << "  ALL TOKENIZER TESTS PASSED SUCCESSFULLY! \n"
              << "===========================================\n";
    return 0;
}
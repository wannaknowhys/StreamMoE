#pragma once

// C++ converter write side: model_t (from parse_model) -> target GGUF.
// Pure C++ (replaces the old JS layout + convertd TCP service).

#include "loader/model.h"

#include <string>
#include <vector>

namespace stream_moe {

struct convert_opts_t {
    enum class target_t { V2, V3, V3_CHUNK };
    target_t target = target_t::V3;
    int chunks = 5;
    std::vector<int> ratio; // empty = uniform; else per-file ratio
};

// out = output file path (V2/V3) or output directory (V3_CHUNK).
// Throws std::runtime_error on any failure.
void convert_model(const model_t& model, const convert_opts_t& opts, const std::string& out);

} // namespace stream_moe

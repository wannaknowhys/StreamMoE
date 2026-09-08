#pragma once

// Shared layout primitives used by both the loader (read direction) and the
// C++ converter (write direction). Keeping them here is the single-source-of-
// truth guarantee that replaces the old JS/C++ duplication
// (docs/STREAMMOE_GGUF_FORMAT.md SS3.5).

#include "loader/model.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace stream_moe {

inline uint64_t sm_align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

inline int32_t sm_layer_of(const std::string& name) {
    const size_t p = name.find("blk.");
    if (p == std::string::npos) return -1;
    return std::atoi(name.c_str() + p + 4);
}

inline std::string sm_branch_of(const std::string& name) {
    if (name.find("gate_up") != std::string::npos) return "gate_up";
    if (name.find("ffn_gate_exps") != std::string::npos) return "gate";
    if (name.find("ffn_up_exps") != std::string::npos) return "up";
    if (name.find("down_exps") != std::string::npos) return "down";
    return "?";
}

// C1/C2/C3/C4 classification by name/structure (proxy for closure membership).
//   C2: name does not start with "blk."
//   C3: "blk.*_exps.weight" with ne[2] == n_expert (throws otherwise)
//   C4: "blk.*_exps.*" that is not ".weight" (per-expert scale/bias tables)
//   C1: all other "blk.*" (attn / norm / dense ffn / router / shexp)
inline tensor_category_t sm_classify(const std::string& name, const int64_t ne[4], uint32_t n_expert) {
    if (name.rfind("blk.", 0) != 0) return tensor_category_t::DENSE_GLOBAL;
    if (name.find("_exps") == std::string::npos) return tensor_category_t::DENSE_LAYER;
    const bool is_weight = name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0;
    if (!is_weight) return tensor_category_t::EXPERT_META;
    if (n_expert == 0 || ne[2] != static_cast<int64_t>(n_expert)) {
        throw std::runtime_error("expert tensor ne[2] != n_expert: " + name);
    }
    return tensor_category_t::EXPERT;
}

inline int sm_branch_order(const std::string& b) {
    for (int i = 0; i < EXPERT_BRANCH_ORDER_LEN; ++i)
        if (b == EXPERT_BRANCH_ORDER[i]) return i;
    return EXPERT_BRANCH_ORDER_LEN;
}

// Split B 4K blocks across N files: uniform base/rem (default) or largest-
// remainder by ratio (docs/STREAMMOE_GGUF_FORMAT.md SS7.2).
inline std::vector<int> sm_split_blocks(uint64_t B, int N, const std::vector<int>& ratio) {
    std::vector<int> out(static_cast<size_t>(N), 0);
    if (ratio.empty()) {
        const uint64_t base = B / static_cast<uint64_t>(N);
        const uint64_t rem  = B % static_cast<uint64_t>(N);
        for (int i = 0; i < N; ++i) out[static_cast<size_t>(i)] = static_cast<int>(base + (static_cast<uint64_t>(i) < rem ? 1 : 0));
    } else {
        if (static_cast<int>(ratio.size()) != N) throw std::runtime_error("split_blocks: ratio length != N");
        const double sum = [&] { double s = 0; for (int r : ratio) s += r; return s; }();
        if (sum <= 0) throw std::runtime_error("split_blocks: ratio sum <= 0");
        std::vector<double> quota(static_cast<size_t>(N));
        uint64_t total = 0;
        for (int i = 0; i < N; ++i) {
            quota[static_cast<size_t>(i)] = static_cast<double>(B) * ratio[static_cast<size_t>(i)] / sum;
            out[static_cast<size_t>(i)] = static_cast<int>(quota[static_cast<size_t>(i)]);
            total += static_cast<uint64_t>(out[static_cast<size_t>(i)]);
        }
        uint64_t diff = B - total;
        std::vector<int> order(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) order[static_cast<size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const double fa = quota[static_cast<size_t>(a)] - out[static_cast<size_t>(a)];
            const double fb = quota[static_cast<size_t>(b)] - out[static_cast<size_t>(b)];
            return fa > fb;
        });
        for (uint64_t k = 0; k < diff; ++k) out[static_cast<size_t>(order[static_cast<size_t>(k)])]++;
    }
    uint64_t s = 0;
    for (int v : out) s += static_cast<uint64_t>(v);
    if (s != B) throw std::runtime_error("split_blocks: strip sum != block count");
    return out;
}

} // namespace stream_moe

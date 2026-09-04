#pragma once

#include <cstdint>
#include <string>
#include <vector>

// C++ model description mirroring tools/stream_moe_layout.js buildModel() -
// the single source of truth for GGUF layout knowledge (read direction). The
// JS writer and this loader share the same Model shape so a future pure-C++
// convertd can reuse this file wholesale.
//
// A Model is built by a per-format parser that produces UNIFORM tensor
// descriptions with source segments; everything downstream (DIO plan, slot
// placement, staging decision) reads only this structure.

namespace stream_moe {

// Mirrors stream_moe.layout KV + incomplete flag.
enum class model_layout_t : uint8_t {
    ORIGINAL         = 0, // per-tensor contiguous, GGUF default align (needs staging)
    V1_SECTIONS      = 1, // per-tensor contiguous, per-expert slice 4K-aligned (straight DIO)
    V2_EXPERT_BLOCKS = 2, // per-(layer,expert) 4K-aligned block (1 straight DIO)
    V2_CHUNK         = 3, // v2 blocks strip-scattered across N files (incomplete=1)
};

// Branch order used by the converter (layout.js ORDER).
inline const char * const EXPERT_BRANCH_ORDER[] = { "gate_up", "gate", "up", "down" };
constexpr int EXPERT_BRANCH_ORDER_LEN = 4;

// A source segment: bytes [off, off+len) of source file `file`, located
// `in_off` bytes into its containing tensor / per-expert slice.
struct src_seg_t {
    uint32_t file    = 0;   // index into model_t::files
    uint64_t off     = 0;   // absolute file offset
    uint64_t len     = 0;   // byte length
    uint64_t in_off  = 0;   // offset of this segment inside the tensor / slice
};

struct dense_tensor_t {
    std::string name;
    int64_t     ne[4] = { 1, 1, 1, 1 };
    int32_t     type  = 0;
    uint64_t    size  = 0;
    std::vector<src_seg_t> srcs; // single-file => 1 seg; v2 chunk => N segs
};

struct expert_tensor_t {
    std::string name;
    int64_t     ne[4] = { 1, 1, 1, 1 };
    int32_t     type  = 0;
    uint64_t    size  = 0;        // whole tensor bytes
    uint64_t    per_expert = 0;   // per-expert slice bytes
    std::string branch;           // "gate_up" | "gate" | "up" | "down"
    int32_t     layer = -1;
    uint64_t    branch_off = 0;   // v2: offset of this branch inside the (layer,e) block
    // per_expert_srcs[e] = source segments for expert e's slice of this tensor.
    //   original / v1 : 1 segment (tensor.offset + e*per_expert, contiguous)
    //   v2 single-file : 1 segment (branchOff inside the (layer,e) block)
    //   v2 chunk       : N segments (block strip scattered across N files)
    std::vector<std::vector<src_seg_t>> per_expert_srcs;
};

// Uniform model description (mirrors layout.js buildModel output).
struct model_t {
    std::string arch;
    model_layout_t layout = model_layout_t::ORIGINAL;
    uint32_t n_layer      = 0;
    uint32_t n_expert     = 0;
    uint32_t n_expert_used = 0;
    bool     incomplete   = false; // v2 chunk (incomplete=1)

    std::vector<std::string> files;      // source file paths, indexed by src_seg.file
    std::vector<uint64_t>    data_offs;  // per-file GGUF data-area offset (header end, aligned)
    std::vector<dense_tensor_t>  dense;
    std::vector<expert_tensor_t> expert; // sorted (layer, branch ORDER)

    // v2 / v2-chunk layout metadata (from stream_moe.* KV), kept raw for
    // planners that need block/strip geometry.
    std::vector<uint64_t> dense_section;    // [0, denseEnd]
    std::vector<uint64_t> expert_sections;  // [off, size, nsub] per block
    std::vector<std::vector<src_seg_t>> block_srcs; // v2/v2chunk: whole-block source segments
                                          // (in_off = segment offset inside the block;
                                          //  v2 single-file = 1 seg, v2 chunk = N file segs)
    std::vector<std::vector<std::string>> branch_names; // per-layer branch names
    std::vector<std::vector<uint64_t>>    branch_sizes; // per-layer perExpert bytes
    std::vector<std::vector<uint64_t>>    chunk_slices; // per file: [denseBlocks, blockSlices...]
    bool branch_align = false; // v2: each branch (tensor) slice starts 4K-aligned
                               // inside the block (stream_moe.branch_align=1, 2026-09)

    // ---- helpers (concept) ----
    bool is_v2_blocks() const {
        return layout == model_layout_t::V2_EXPERT_BLOCKS || layout == model_layout_t::V2_CHUNK;
    }
    // A per-expert slice can be read straight into a 4K-aligned slot iff every
    // source segment start is 4K-aligned (v1/v2 guarantee; original does not).
    bool expert_slices_4k_aligned() const {
        return layout != model_layout_t::ORIGINAL;
    }
};

} // namespace stream_moe

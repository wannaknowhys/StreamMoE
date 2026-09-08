#pragma once

#include <cstdint>
#include <string>
#include <vector>

// C++ model description - the single source of truth for GGUF layout knowledge
// (read direction). The loader and the C++ converter share this file: the
// converter parses a source with parse_model() and writes a target from the
// same model_t (see docs/STREAMMOE_GGUF_FORMAT.md SS3).
//
// A Model is built by a per-format parser that produces UNIFORM tensor
// descriptions with source segments; everything downstream (DIO plan, slot
// placement, staging decision) reads only this structure.

namespace stream_moe {

// Mirrors stream_moe.layout KV + incomplete flag. v1 is removed (never loadable).
enum class model_layout_t : uint8_t {
    ORIGINAL         = 0, // per-tensor contiguous, GGUF default align (needs staging)
    V2_EXPERT_BLOCKS = 2, // per-(layer,expert) 4K-aligned block (1 straight DIO)
    V2_CHUNK         = 3, // v2 blocks strip-scattered across N files (incomplete=1)
    V3               = 4, // four category sections (global / layer / meta / expert blocks)
};

// v3 tensor category: split by whether the tensor is consumed by the MoE
// closure (docs/STREAMMOE_GGUF_FORMAT.md SS3.1). Name/structure proxy here.
enum class tensor_category_t : uint8_t {
    DENSE_LAYER  = 1, // C1: not closure, per-layer (attn/norm/router/shexp)
    DENSE_GLOBAL = 2, // C2: not closure, layer-independent (token_embd/output)
    EXPERT       = 3, // C3: closure, sliced per expert (pool weights)
    EXPERT_META  = 4, // C4: closure, not per-expert (per-expert scale/bias tables)
};

// Branch order used by the converter.
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
    std::string       name;
    int64_t           ne[4] = { 1, 1, 1, 1 };
    int32_t           type  = 0;
    uint64_t          size  = 0;
    tensor_category_t category = tensor_category_t::DENSE_LAYER; // C1/C2/C4
    int32_t           layer = -1;                               // C1/C4 (C2 = -1)
    std::vector<src_seg_t> srcs; // single-file => 1 seg; chunk => N segs
};

struct expert_tensor_t {
    std::string name;
    int64_t     ne[4] = { 1, 1, 1, 1 };
    int32_t     type  = 0;
    uint64_t    size  = 0;        // whole tensor bytes
    uint64_t    per_expert = 0;   // per-expert slice bytes
    std::string branch;           // "gate_up" | "gate" | "up" | "down"
    int32_t     layer = -1;
    uint64_t    branch_off = 0;   // v2/v3: offset of this branch inside the (layer,e) block
    // per_expert_srcs[e] = source segments for expert e's slice of this tensor.
    //   original : 1 segment (tensor.offset + e*per_expert, contiguous)
    //   v2/v3 single-file : 1 segment (branch_off inside the (layer,e) block)
    //   v2/v3 chunk       : N segments (block strip scattered across N files)
    std::vector<std::vector<src_seg_t>> per_expert_srcs;
};

// Uniform model description.
struct model_t {
    std::string arch;
    model_layout_t layout = model_layout_t::ORIGINAL;
    uint32_t n_layer      = 0;
    uint32_t n_expert     = 0;
    uint32_t n_expert_used = 0;
    bool     incomplete   = false; // chunk (incomplete=1)

    std::vector<std::string> files;      // source file paths, indexed by src_seg.file
    std::vector<uint64_t>    data_offs;  // per-file GGUF data-area offset (header end, aligned)
    std::vector<dense_tensor_t>  dense;  // source order (category set per tensor)
    std::vector<expert_tensor_t> expert; // sorted (layer, branch ORDER)

    // v2 / v2-chunk / v3 layout metadata (from stream_moe.* KV), kept raw for
    // planners that need block/strip geometry.
    std::vector<uint64_t> dense_section;    // v2: [0, denseEnd]; v3: [global off, size]
    std::vector<uint64_t> expert_sections;  // [off, size, nsub] per block
    std::vector<std::vector<src_seg_t>> block_srcs; // whole-block source segments
    std::vector<std::vector<std::string>> branch_names; // per-layer branch names
    std::vector<std::vector<uint64_t>>    branch_sizes; // per-layer perExpert bytes
    std::vector<std::vector<uint64_t>>    chunk_slices; // per file: [unitBlocks...]
    bool branch_align = false; // v2/v3: each branch slice starts 4K-aligned inside the block

    // v3 category sections (flat [layer, off, size, ...]).
    std::vector<uint64_t> dense_layer_sections;
    std::vector<uint64_t> expert_meta_sections;

    bool is_v2_blocks() const {
        return layout == model_layout_t::V2_EXPERT_BLOCKS || layout == model_layout_t::V2_CHUNK;
    }
    bool is_v3() const { return layout == model_layout_t::V3; }
    bool is_blocks() const { return is_v2_blocks() || is_v3(); }
    // A per-expert slice can be read straight into a 4K-aligned slot iff every
    // source segment start is 4K-aligned (v2/v3 guarantee; original does not).
    bool expert_slices_4k_aligned() const {
        return layout != model_layout_t::ORIGINAL;
    }
};

} // namespace stream_moe

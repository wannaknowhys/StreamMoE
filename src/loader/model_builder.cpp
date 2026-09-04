#include "loader/model_builder.h"
#include "common/logger.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <regex>
#include <stdexcept>

namespace stream_moe {

namespace {

constexpr uint64_t ALIGN = 4096;
constexpr uint64_t MB    = 1024ull * 1024ull;
constexpr uint64_t GB    = 1024ull * 1024ull * 1024ull;

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

int64_t kv_int(const gguf_context* ctx, const char* key, int64_t def = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, id));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        case GGUF_TYPE_FLOAT32: return static_cast<int64_t>(gguf_get_val_f32(ctx, id));
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(ctx, id) ? 1 : 0;
        default: return def;
    }
}

std::string kv_str(const gguf_context* ctx, const char* key, const std::string& def = "") {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(ctx, id);
}

// u64 array KV -> out (throws if missing or wrong type)
void kv_u64_arr(const gguf_context* ctx, const char* key, std::vector<uint64_t>& out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, id) != GGUF_TYPE_UINT64) {
        throw std::runtime_error(std::string("missing/invalid ") + key);
    }
    const uint64_t* d = static_cast<const uint64_t*>(gguf_get_arr_data(ctx, id));
    out.assign(d, d + gguf_get_arr_n(ctx, id));
}

// str array KV -> out (throws if missing or wrong type)
void kv_str_arr(const gguf_context* ctx, const char* key, std::vector<std::string>& out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("missing/invalid ") + key);
    }
    out.reserve(gguf_get_arr_n(ctx, id));
    for (size_t i = 0; i < static_cast<size_t>(gguf_get_arr_n(ctx, id)); ++i) {
        out.push_back(gguf_get_arr_str(ctx, id, i));
    }
}

// Discover original GGUF shards (-00001-of-N.gguf) like moe_loader.
std::vector<std::string> discover_shards(const std::string& main_path, const gguf_context* ctx0) {
    std::vector<std::string> shards{ main_path };
    const int64_t split_count = kv_int(ctx0, "split.count", 0);
    static const std::regex split_re("^(.*-)(\\d{5})-of-(\\d{5})(\\.gguf)$", std::regex::icase);
    std::smatch m;
    if (std::regex_match(main_path, m, split_re)) {
        const std::string prefix = m[1].str();
        const int total = std::stoi(m[3].str());
        const std::string suffix = m[4].str();
        if (split_count > 0 && split_count != total) {
            throw std::runtime_error("shard count mismatch: filename " + std::to_string(total)
                                     + " vs metadata " + std::to_string(split_count));
        }
        shards.clear();
        for (int i = 1; i <= total; ++i) {
            char buf[40];
            snprintf(buf, sizeof(buf), "%05d-of-%05d", i, total);
            const std::string p = prefix + buf + suffix;
            if (!std::filesystem::exists(p)) {
                throw std::runtime_error("missing shard " + std::to_string(i) + "/" + std::to_string(total) + ": " + p);
            }
            shards.push_back(p);
        }
    } else if (split_count > 1) {
        const std::filesystem::path pp(main_path);
        const std::string stem = pp.stem().string();
        const std::string ext = pp.extension().string();
        const std::string dir = pp.parent_path().string();
        shards.clear();
        for (int i = 1; i <= split_count; ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s-%05d-of-%05d%s", stem.c_str(), i, static_cast<int>(split_count), ext.c_str());
            const std::string p = dir.empty() ? buf : dir + "/" + buf;
            if (!std::filesystem::exists(p)) {
                throw std::runtime_error("missing shard " + std::to_string(i) + ": " + p);
            }
            shards.push_back(p);
        }
    }
    return shards;
}

// --- v2 / v2-chunk helpers (mirror layout.js buildBlocks/buildLayerBranches) ---

struct block_t { uint64_t off = 0, size = 0; uint32_t nsub = 0; };
struct branch_t { std::string name; std::string tag; uint64_t per_expert = 0; uint64_t branch_off = 0; };

std::vector<block_t> build_blocks(const std::vector<uint64_t>& sections) {
    std::vector<block_t> blocks;
    const size_t n = sections.size() / 3;
    blocks.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        block_t b;
        b.off = sections[i * 3];
        b.size = sections[i * 3 + 1];
        b.nsub = static_cast<uint32_t>(sections[i * 3 + 2]);
        blocks.push_back(b);
    }
    return blocks;
}

std::vector<std::vector<branch_t>> build_layer_branches(
        const std::vector<std::string>& bnames, const std::vector<uint64_t>& bsizes,
        const std::vector<uint64_t>& bcounts, uint32_t n_layer, bool branch_align) {
    std::vector<std::vector<branch_t>> layers(n_layer);
    size_t ni = 0, si = 0;
    for (uint32_t l = 0; l < n_layer; ++l) {
        const uint64_t cnt = l < bcounts.size() ? bcounts[l] : 0;
        uint64_t off = 0;
        for (uint64_t j = 0; j < cnt; ++j) {
            branch_t b;
            b.name = bnames[ni++];
            b.per_expert = bsizes[si++];
            b.tag = "?";
            if (b.name.find("gate_up") != std::string::npos) b.tag = "gate_up";
            else if (b.name.find("ffn_gate_exps") != std::string::npos) b.tag = "gate";
            else if (b.name.find("ffn_up_exps") != std::string::npos) b.tag = "up";
            else if (b.name.find("down_exps") != std::string::npos) b.tag = "down";
            b.branch_off = off;
            // branch_align=1: each branch slice starts 4K-aligned (mirror
            // layout.js computeV2Layout); legacy compact otherwise.
            off = branch_align ? align_up(off + b.per_expert, ALIGN) : off + b.per_expert;
            layers[l].push_back(b);
        }
    }
    return layers;
}

// Map a logical interval [rel, rel+len) of a v2-chunk strip layout into source
// segments (mirror layout.js rangeToSegs). rel is relative to the block strip
// area (dense section is excluded: base starts after per-file dense strips).
// kind='dense' or 'block'. Output offsets are ABSOLUTE file offsets.
std::vector<src_seg_t> range_to_segs(
        const model_t& model, bool is_block, uint32_t blk_index,
        uint64_t rel, uint64_t len) {
    std::vector<src_seg_t> segs;
    if (!model.incomplete) {
        segs.push_back({ 0, model.data_offs.empty() ? rel : model.data_offs[0] + rel, len, 0 });
        return segs;
    }
    // v2 chunk: strip layout across N files
    const size_t N = model.chunk_slices.size();
    std::vector<uint64_t> counts(N), base(N, 0);
    if (!is_block) {
        for (size_t i = 0; i < N; ++i) counts[i] = model.chunk_slices[i].empty() ? 0 : model.chunk_slices[i][0];
    } else {
        for (size_t i = 0; i < N; ++i) {
            counts[i] = (model.chunk_slices[i].size() > 1 + blk_index) ? model.chunk_slices[i][1 + blk_index] : 0;
            uint64_t s = 0;
            for (uint64_t k = 0; k < blk_index; ++k) {
                s += (model.chunk_slices[i].size() > 1 + k) ? model.chunk_slices[i][1 + k] : 0;
            }
            // per-file block-strip base = this file's dense strip + prefix blocks
            base[i] = (model.chunk_slices[i].empty() ? 0 : model.chunk_slices[i][0]) + s;
        }
    }
    std::vector<uint64_t> cum(N);
    { uint64_t c = 0; for (size_t i = 0; i < N; ++i) { cum[i] = c; c += counts[i]; } }
    uint64_t cur = rel / ALIGN;
    const uint64_t end = (rel + len + ALIGN - 1) / ALIGN;
    while (cur < end) {
        size_t fi = 0;
        while (fi + 1 < N && cur >= cum[fi] + counts[fi]) fi++;
        const uint64_t local = cur - cum[fi];
        const uint64_t avail = std::min(counts[fi] - local, end - cur);
        if (avail == 0) {
            throw std::runtime_error("range_to_segs: interval past strip coverage");
        }
        const uint64_t seg_start = std::max(rel, cur * ALIGN);
        const uint64_t seg_end   = std::min(rel + len, (cur + avail) * ALIGN);
        if (seg_end > seg_start) {
            src_seg_t s;
            s.file = static_cast<uint32_t>(fi);
            s.off = model.data_offs[fi] + base[fi] * ALIGN + local * ALIGN + (seg_start - cur * ALIGN);
            s.len = seg_end - seg_start;
            s.in_off = seg_start - rel;
            segs.push_back(s);
        }
        cur += avail;
    }
    return segs;
}

} // namespace

// ---------------------------------------------------------------------------

model_t parse_model(const std::vector<std::string>& paths) {
    if (paths.empty()) throw std::runtime_error("parse_model: no paths");

    const gguf_init_params params{ /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context* ctx0 = gguf_init_from_file(paths[0].c_str(), params);
    if (!ctx0) throw std::runtime_error("cannot open " + paths[0]);

    model_t model;
    model.arch = kv_str(ctx0, "general.architecture", "llama");
    const std::string layout = kv_str(ctx0, "stream_moe.layout", "");
    const bool incomplete = kv_int(ctx0, "stream_moe.incomplete", 0) == 1;

    if (layout == "expert-blocks-v2") {
        model.layout = incomplete ? model_layout_t::V2_CHUNK : model_layout_t::V2_EXPERT_BLOCKS;
    } else if (layout == "sections-v1") {
        model.layout = model_layout_t::V1_SECTIONS;
    } else {
        model.layout = model_layout_t::ORIGINAL;
    }
    model.incomplete = incomplete;
    model.n_layer = static_cast<uint32_t>(kv_int(ctx0, (model.arch + ".block_count").c_str(),
                                       kv_int(ctx0, "block_count", 0)));
    model.n_expert = static_cast<uint32_t>(kv_int(ctx0, (model.arch + ".expert_count").c_str(),
                                         kv_int(ctx0, "expert_count", 0)));
    model.n_expert_used = static_cast<uint32_t>(kv_int(ctx0, (model.arch + ".expert_used_count").c_str(), 0));

    // Source file list. Original multi-shard is discovered from paths[0];
    // v2 chunk passes all strip files explicitly.
    if (model.layout == model_layout_t::ORIGINAL) {
        model.files = discover_shards(paths[0], ctx0);
    } else {
        model.files = paths;
    }

    // record per-file data-area offsets (header end, 4K-aligned by converter)
    for (const auto& p : model.files) {
        gguf_context* fc = gguf_init_from_file(p.c_str(), params);
        if (!fc) throw std::runtime_error("cannot open " + p);
        model.data_offs.push_back(gguf_get_data_offset(fc));
        gguf_free(fc);
    }

    // v2 / v2-chunk layout KV
    if (model.is_v2_blocks()) {
        kv_u64_arr(ctx0, "stream_moe.expert_sections", model.expert_sections);
        std::vector<std::string> bnames;
        std::vector<uint64_t> bsizes, bcounts;
        kv_str_arr(ctx0, "stream_moe.expert_branch_names", bnames);
        kv_u64_arr(ctx0, "stream_moe.expert_branch_sizes", bsizes);
        kv_u64_arr(ctx0, "stream_moe.expert_branch_counts", bcounts);
        model.branch_align = kv_int(ctx0, "stream_moe.branch_align") == 1;
        const std::vector<block_t> blocks = build_blocks(model.expert_sections);
        const auto layers = build_layer_branches(bnames, bsizes, bcounts, model.n_layer, model.branch_align);
        model.dense_section = { 0, 0 };
        for (size_t i = 0; i < model.files.size(); ++i) model.chunk_slices.emplace_back();
        if (model.incomplete) {
            // dense section end = first block start (single-file) or file0 denseEnd
            const uint64_t dense_end = blocks.empty() ? 0 : blocks[0].off;
            model.dense_section = { 0, dense_end };
            for (size_t i = 0; i < model.files.size(); ++i) {
                gguf_context* fc = gguf_init_from_file(model.files[i].c_str(), params);
                if (!fc) throw std::runtime_error("cannot open chunk " + model.files[i]);
                kv_u64_arr(fc, "stream_moe.chunk_slices", model.chunk_slices[i]);
                gguf_free(fc);
            }
        }

        // Tensor metadata comes from file 0 (v2-chunk files each carry the full
        // tensor_info table). Classify dense vs expert, map per-expert slices.
        gguf_context* tf = gguf_init_from_file(model.files[0].c_str(), params);
        if (!tf) throw std::runtime_error("cannot open " + model.files[0]);
        const uint64_t data_off = gguf_get_data_offset(tf);
        // whole-block source segments: v2 single-file = 1 seg; v2 chunk = N file
        // strip segments (in_off = segment offset inside the block).
        model.block_srcs.resize(blocks.size());
        for (size_t bi = 0; bi < blocks.size(); ++bi) {
            if (model.incomplete) {
                model.block_srcs[bi] = range_to_segs(model, true, static_cast<uint32_t>(bi), 0, blocks[bi].size);
            } else {
                model.block_srcs[bi].push_back({ 0, data_off + blocks[bi].off, blocks[bi].size, 0 });
            }
        }
        const int n_t = gguf_get_n_tensors(tf);
        for (int i = 0; i < n_t; ++i) {
            const char* tname = gguf_get_tensor_name(tf, i);
            const std::string name(tname);
            const bool is_scale = name.find(".scale") != std::string::npos;
            const bool is_exp = name.find("_exps") != std::string::npos && !is_scale;
            const uint64_t toff = data_off + static_cast<uint64_t>(gguf_get_tensor_offset(tf, i));
            const uint64_t tsize = gguf_get_tensor_size(tf, i);
            const int32_t ttype = static_cast<int32_t>(gguf_get_tensor_type(tf, i));
            const int64_t* ne = gguf_get_tensor_ne(tf, i);

            if (is_exp) {
                const uint64_t per_expert = model.n_expert ? tsize / model.n_expert : 0;
                // layer + branch from name
                int32_t layer = -1;
                const size_t p = name.find("blk.");
                if (p != std::string::npos) layer = std::atoi(name.c_str() + p + 4);
                std::string branch = "?";
                if (name.find("gate_up") != std::string::npos) branch = "gate_up";
                else if (name.find("ffn_gate_exps") != std::string::npos) branch = "gate";
                else if (name.find("ffn_up_exps") != std::string::npos) branch = "up";
                else if (name.find("down_exps") != std::string::npos) branch = "down";

                expert_tensor_t et;
                et.name = name; et.type = ttype; et.size = tsize; et.per_expert = per_expert;
                et.branch = branch; et.layer = layer;
                for (int d = 0; d < 4; ++d) et.ne[d] = ne[d];
                et.per_expert_srcs.resize(model.n_expert);
                // branch offset inside the (layer,e) block (v2 / v2-chunk)
                uint64_t branch_off = 0;
                if (model.is_v2_blocks()) {
                    const auto& bl = layers[layer];
                    for (const auto& b : bl) { if (b.tag == branch) { branch_off = b.branch_off; break; } }
                }

                for (uint32_t e = 0; e < model.n_expert; ++e) {
                    if (model.layout == model_layout_t::V2_EXPERT_BLOCKS) {
                        // branch sits at branchOff inside the (layer,e) block
                        const uint32_t blk_idx = static_cast<uint32_t>(layer) * model.n_expert + e;
                        et.per_expert_srcs[e].push_back({ 0, data_off + blocks[blk_idx].off + branch_off, per_expert, 0 });
                    } else {
                        // V2_CHUNK: block strip scattered across N files
                        et.per_expert_srcs[e] = range_to_segs(model, true,
                                                              static_cast<uint32_t>(layer) * model.n_expert + e,
                                                              branch_off, per_expert);
                    }
                }
                et.branch_off = branch_off;
                model.expert.push_back(et);
            } else {
                dense_tensor_t dt;
                dt.name = name; dt.type = ttype; dt.size = tsize;
                for (int d = 0; d < 4; ++d) dt.ne[d] = ne[d];
                dt.srcs = model.incomplete ? range_to_segs(model, false, 0, toff - data_off, tsize)
                                           : std::vector<src_seg_t>{ { 0, toff, tsize, 0 } };
                model.dense.push_back(dt);
            }
        }
        gguf_free(tf);
    } else {
        // ORIGINAL / V1: per-tensor contiguous across (possibly multiple) shards.
        // Aggregate tensor metadata from every shard, then slice per expert.
        struct tentry { uint32_t shard; uint64_t off; uint64_t size; int32_t type; int64_t ne[4]; };
        std::vector<std::pair<std::string, tentry>> tensors;
        const uint32_t n_files = static_cast<uint32_t>(model.files.size());
        for (uint32_t s = 0; s < n_files; ++s) {
            gguf_context* sc = gguf_init_from_file(model.files[s].c_str(), params);
            if (!sc) throw std::runtime_error("cannot open " + model.files[s]);
            const uint64_t doff = gguf_get_data_offset(sc);
            const int nt = gguf_get_n_tensors(sc);
            for (int i = 0; i < nt; ++i) {
                const char* tname = gguf_get_tensor_name(sc, i);
                tentry e;
                e.shard = s;
                e.off = doff + static_cast<uint64_t>(gguf_get_tensor_offset(sc, i));
                e.size = gguf_get_tensor_size(sc, i);
                e.type = static_cast<int32_t>(gguf_get_tensor_type(sc, i));
                const int64_t* ne = gguf_get_tensor_ne(sc, i);
                for (int d = 0; d < 4; ++d) e.ne[d] = ne[d];
                tensors.push_back({ tname, e });
            }
            gguf_free(sc);
        }

        for (auto& [name, e] : tensors) {
            const bool is_scale = name.find(".scale") != std::string::npos;
            const bool is_exp = name.find("_exps") != std::string::npos && !is_scale;
            if (is_exp) {
                const uint64_t per_expert = model.n_expert ? e.size / model.n_expert : 0;
                // v1 target layout pads each per-expert slice to 4K; ORIGINAL does not.
                const uint64_t stride = model.layout == model_layout_t::V1_SECTIONS ? align_up(per_expert, ALIGN) : per_expert;
                int32_t layer = -1;
                const size_t p = name.find("blk.");
                if (p != std::string::npos) layer = std::atoi(name.c_str() + p + 4);
                std::string branch = "?";
                if (name.find("gate_up") != std::string::npos) branch = "gate_up";
                else if (name.find("ffn_gate_exps") != std::string::npos) branch = "gate";
                else if (name.find("ffn_up_exps") != std::string::npos) branch = "up";
                else if (name.find("down_exps") != std::string::npos) branch = "down";

                expert_tensor_t et;
                et.name = name; et.type = e.type; et.size = e.size; et.per_expert = per_expert;
                et.branch = branch; et.layer = layer;
                for (int d = 0; d < 4; ++d) et.ne[d] = e.ne[d];
                et.per_expert_srcs.resize(model.n_expert);
                for (uint32_t x = 0; x < model.n_expert; ++x) {
                    et.per_expert_srcs[x].push_back({ e.shard, e.off + static_cast<uint64_t>(x) * stride, per_expert, 0 });
                }
                model.expert.push_back(et);
            } else {
                dense_tensor_t dt;
                dt.name = name; dt.type = e.type; dt.size = e.size;
                for (int d = 0; d < 4; ++d) dt.ne[d] = e.ne[d];
                dt.srcs.push_back({ e.shard, e.off, e.size, 0 });
                model.dense.push_back(dt);
            }
        }
    }

    // Sort experts by (layer, branch ORDER) like layout.js.
    auto order_of = [](const std::string& b) {
        for (int i = 0; i < EXPERT_BRANCH_ORDER_LEN; ++i)
            if (b == EXPERT_BRANCH_ORDER[i]) return i;
        return EXPERT_BRANCH_ORDER_LEN;
    };
    std::sort(model.expert.begin(), model.expert.end(),
              [&](const expert_tensor_t& a, const expert_tensor_t& b) {
                  return a.layer != b.layer ? a.layer < b.layer : order_of(a.branch) < order_of(b.branch);
              });

    gguf_free(ctx0);
    return model;
}

model_t parse_model_path(const std::string& main_path) {
    return parse_model(std::vector<std::string>{ main_path });
}

} // namespace stream_moe

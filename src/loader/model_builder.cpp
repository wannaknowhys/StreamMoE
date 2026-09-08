#include "loader/model_builder.h"
#include "loader/layout_math.h"
#include "common/logger.h"

#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
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

// Discover the chunk set from any one strip file. Each chunk file carries
// stream_moe.chunk_no (0-based, its own position) + chunk_total, so the given
// file's metadata anchors the set; the filename supplies the prefix/ext and
// numbering. A directory scan matches by PARSED number, so zero-padded and
// unpadded names both work (on a number collision, prefer the given file's
// digit width). Only used when no explicit file list was given.
std::vector<std::string> discover_chunk_files(const std::string& main_path,
                                              uint32_t chunk_no, uint32_t chunk_total) {
    const std::filesystem::path p(main_path);
    const std::string stem = p.stem().string();
    const std::string ext  = p.extension().string();
    const std::string dir  = p.parent_path().string();
    size_t e = stem.size();
    while (e > 0 && std::isdigit(static_cast<unsigned char>(stem[e - 1]))) --e;
    if (e == stem.size()) {
        throw std::runtime_error("chunk source needs a trailing index in the filename (e.g. c1.gguf): " + main_path);
    }
    const std::string prefix = stem.substr(0, e);
    const std::string given_digits = stem.substr(e);
    const long given_num = std::stol(given_digits);
    const size_t given_width = given_digits.size();

    std::map<long, std::string> by_num;
    const std::string scan_dir = dir.empty() ? std::string(".") : dir;
    for (const auto& entry : std::filesystem::directory_iterator(scan_dir)) {
        const std::string fn = entry.path().filename().string();
        if (fn.size() <= prefix.size() + ext.size()) continue;
        if (fn.compare(0, prefix.size(), prefix) != 0) continue;
        if (fn.compare(fn.size() - ext.size(), ext.size(), ext) != 0) continue;
        const std::string mid = fn.substr(prefix.size(), fn.size() - prefix.size() - ext.size());
        if (mid.empty() || mid.find_first_not_of("0123456789") != std::string::npos) continue;
        const long num = std::stol(mid);
        auto it = by_num.find(num);
        if (it == by_num.end() || mid.size() == given_width) by_num[num] = entry.path().string();
    }

    const long start = given_num - static_cast<long>(chunk_no);
    std::vector<std::string> out;
    out.reserve(chunk_total);
    for (uint32_t i = 0; i < chunk_total; ++i) {
        const long target = start + static_cast<long>(i);
        auto it = by_num.find(target);
        if (it == by_num.end()) {
            throw std::runtime_error("missing chunk file #" + std::to_string(target) +
                                     " (prefix '" + prefix + "', ext '" + ext + "') in " + scan_dir);
        }
        out.push_back(it->second);
    }
    return out;
}

// --- v2 / v3 block helpers (mirror the converter writer) ---

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
            b.tag = sm_branch_of(b.name);
            b.branch_off = off;
            off = branch_align ? align_up(off + b.per_expert, ALIGN) : off + b.per_expert;
            layers[l].push_back(b);
        }
    }
    return layers;
}

// v3 chunk unit geometry: units are [C2 global] + [C1 layers] + [C4 layers] +
// [expert blocks], all in file order (mirror the writer).
struct v3_units_t {
    uint32_t n_c1 = 0, n_c4 = 0;
    uint32_t c1_unit(uint32_t li)    const { return 1 + li; }
    uint32_t c4_unit(uint32_t mi)    const { return 1 + n_c1 + mi; }
    uint32_t block_unit(uint32_t bi) const { return 1 + n_c1 + n_c4 + bi; }
};

v3_units_t v3_units_of(const model_t& model) {
    v3_units_t u;
    u.n_c1 = static_cast<uint32_t>(model.dense_layer_sections.size() / 3);
    u.n_c4 = static_cast<uint32_t>(model.expert_meta_sections.size() / 3);
    return u;
}

// Map a logical interval [rel, rel+len) inside chunk unit `unit` to source
// segments across the N strip files (mirror the converter strip writer).
// rel is relative to the unit start; output offsets are ABSOLUTE file offsets.
std::vector<src_seg_t> strip_range_to_segs(
        const model_t& model, uint32_t unit, uint64_t rel, uint64_t len) {
    std::vector<src_seg_t> segs;
    const size_t N = model.chunk_slices.size();
    std::vector<uint64_t> counts(N, 0), base(N, 0);
    for (size_t i = 0; i < N; ++i) {
        const auto& cs = model.chunk_slices[i];
        counts[i] = unit < cs.size() ? cs[unit] : 0;
        uint64_t s = 0;
        for (uint32_t k = 0; k < unit && k < cs.size(); ++k) s += cs[k];
        base[i] = s;
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
        if (avail == 0) throw std::runtime_error("strip_range_to_segs: interval past strip coverage");
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

// Find a layer's [index, base] in a flat [layer, off, size, ...] section table.
bool find_section(const std::vector<uint64_t>& secs, uint64_t layer, uint64_t& index, uint64_t& base) {
    for (size_t k = 0, idx = 0; k + 3 <= secs.size(); k += 3, ++idx) {
        if (secs[k] == layer) { index = idx; base = secs[k + 1]; return true; }
    }
    return false;
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
    } else if (layout == "v3") {
        model.layout = model_layout_t::V3;
    } else {
        model.layout = model_layout_t::ORIGINAL;
    }
    model.incomplete = incomplete;
    model.n_layer = static_cast<uint32_t>(kv_int(ctx0, (model.arch + ".block_count").c_str(),
                                       kv_int(ctx0, "block_count", 0)));
    model.n_expert = static_cast<uint32_t>(kv_int(ctx0, (model.arch + ".expert_count").c_str(),
                                         kv_int(ctx0, "expert_count", 0)));
    model.n_expert_used = static_cast<uint32_t>(kv_int(ctx0, (model.arch + ".expert_used_count").c_str(), 0));

    // Source file list. Original multi-shard is discovered from paths[0]; a
    // single chunk main file discovers its siblings (c1..cN) from chunk_total;
    // an explicit multi-path list (converter merge) is used as given.
    if (model.layout == model_layout_t::ORIGINAL) {
        model.files = discover_shards(paths[0], ctx0);
    } else if (incomplete && paths.size() == 1) {
        const uint32_t chunk_total = static_cast<uint32_t>(kv_int(ctx0, "stream_moe.chunk_total", 1));
        const uint32_t chunk_no    = static_cast<uint32_t>(kv_int(ctx0, "stream_moe.chunk_no", 0));
        model.files = chunk_total > 1 ? discover_chunk_files(paths[0], chunk_no, chunk_total) : paths;
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

    if (model.is_blocks()) {
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
        if (model.is_v3()) {
            kv_u64_arr(ctx0, "stream_moe.dense_global_section", model.dense_section);
            kv_u64_arr(ctx0, "stream_moe.dense_layer_sections", model.dense_layer_sections);
            kv_u64_arr(ctx0, "stream_moe.expert_meta_sections", model.expert_meta_sections);
        }
        for (size_t i = 0; i < model.files.size(); ++i) model.chunk_slices.emplace_back();
        if (model.incomplete) {
            // dense section end = first block start (single-file) or file0 denseEnd
            const uint64_t dense_end = blocks.empty() ? 0 : blocks[0].off;
            model.dense_section = model.is_v3() ? model.dense_section : std::vector<uint64_t>{ 0, dense_end };
            for (size_t i = 0; i < model.files.size(); ++i) {
                gguf_context* fc = gguf_init_from_file(model.files[i].c_str(), params);
                if (!fc) throw std::runtime_error("cannot open chunk " + model.files[i]);
                kv_u64_arr(fc, "stream_moe.chunk_slices", model.chunk_slices[i]);
                gguf_free(fc);
            }
        }
        const v3_units_t vu = v3_units_of(model);

        // Tensor metadata comes from file 0 (chunk files each carry the full
        // tensor_info table). Classify dense vs expert, map per-expert slices.
        gguf_context* tf = gguf_init_from_file(model.files[0].c_str(), params);
        if (!tf) throw std::runtime_error("cannot open " + model.files[0]);
        const uint64_t data_off = gguf_get_data_offset(tf);

        model.block_srcs.resize(blocks.size());
        for (size_t bi = 0; bi < blocks.size(); ++bi) {
            if (model.incomplete) {
                model.block_srcs[bi] = strip_range_to_segs(model, vu.block_unit(static_cast<uint32_t>(bi)), 0, blocks[bi].size);
            } else {
                model.block_srcs[bi].push_back({ 0, data_off + blocks[bi].off, blocks[bi].size, 0 });
            }
        }

        const int n_t = gguf_get_n_tensors(tf);
        for (int i = 0; i < n_t; ++i) {
            const std::string name(gguf_get_tensor_name(tf, i));
            const uint64_t toff = data_off + static_cast<uint64_t>(gguf_get_tensor_offset(tf, i));
            const uint64_t tsize = gguf_get_tensor_size(tf, i);
            const int32_t ttype = static_cast<int32_t>(gguf_get_tensor_type(tf, i));
            const int64_t* ne = gguf_get_tensor_ne(tf, i);
            const tensor_category_t cat = sm_classify(name, ne, model.n_expert);

            if (cat == tensor_category_t::EXPERT) {
                const uint64_t per_expert = model.n_expert ? tsize / model.n_expert : 0;
                const int32_t layer = sm_layer_of(name);
                const std::string branch = sm_branch_of(name);

                expert_tensor_t et;
                et.name = name; et.type = ttype; et.size = tsize; et.per_expert = per_expert;
                et.branch = branch; et.layer = layer;
                for (int d = 0; d < 4; ++d) et.ne[d] = ne[d];
                uint64_t branch_off = 0;
                if (layer >= 0 && static_cast<size_t>(layer) < layers.size()) {
                    for (const auto& b : layers[layer]) { if (b.tag == branch) { branch_off = b.branch_off; break; } }
                }
                et.branch_off = branch_off;
                et.per_expert_srcs.resize(model.n_expert);
                for (uint32_t e = 0; e < model.n_expert; ++e) {
                    const uint32_t blk_idx = static_cast<uint32_t>(layer) * model.n_expert + e;
                    if (model.incomplete) {
                        et.per_expert_srcs[e] = strip_range_to_segs(model, vu.block_unit(blk_idx), branch_off, per_expert);
                    } else {
                        et.per_expert_srcs[e].push_back({ 0, data_off + blocks[blk_idx].off + branch_off, per_expert, 0 });
                    }
                }
                model.expert.push_back(et);
            } else {
                dense_tensor_t dt;
                dt.name = name; dt.type = ttype; dt.size = tsize; dt.category = cat;
                for (int d = 0; d < 4; ++d) dt.ne[d] = ne[d];
                if (cat == tensor_category_t::DENSE_LAYER || cat == tensor_category_t::EXPERT_META) {
                    dt.layer = sm_layer_of(name);
                }
                if (!model.incomplete) {
                    dt.srcs.push_back({ 0, toff, tsize, 0 });
                } else if (model.is_v3()) {
                    // map the tensor into its v3 chunk unit
                    uint32_t unit = 0; uint64_t base = 0, idx = 0;
                    if (cat == tensor_category_t::DENSE_GLOBAL) {
                        base = model.dense_section.empty() ? 0 : model.dense_section[0];
                    } else if (cat == tensor_category_t::DENSE_LAYER) {
                        if (!find_section(model.dense_layer_sections, static_cast<uint64_t>(dt.layer), idx, base)) {
                            throw std::runtime_error("v3 chunk: no dense-layer section for " + name);
                        }
                        unit = vu.c1_unit(static_cast<uint32_t>(idx));
                    } else { // EXPERT_META
                        if (!find_section(model.expert_meta_sections, static_cast<uint64_t>(dt.layer), idx, base)) {
                            throw std::runtime_error("v3 chunk: no expert-meta section for " + name);
                        }
                        unit = vu.c4_unit(static_cast<uint32_t>(idx));
                    }
                    dt.srcs = strip_range_to_segs(model, unit, (toff - data_off) - base, tsize);
                } else {
                    // v2 chunk: single merged dense section
                    dt.srcs = strip_range_to_segs(model, 0, toff - data_off, tsize);
                }
                model.dense.push_back(dt);
            }
        }
        gguf_free(tf);
    } else {
        // ORIGINAL: per-tensor contiguous across (possibly multiple) shards.
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
            const tensor_category_t cat = sm_classify(name, e.ne, model.n_expert);
            if (cat == tensor_category_t::EXPERT) {
                const uint64_t per_expert = model.n_expert ? e.size / model.n_expert : 0;
                expert_tensor_t et;
                et.name = name; et.type = e.type; et.size = e.size; et.per_expert = per_expert;
                et.branch = sm_branch_of(name); et.layer = sm_layer_of(name);
                for (int d = 0; d < 4; ++d) et.ne[d] = e.ne[d];
                et.per_expert_srcs.resize(model.n_expert);
                for (uint32_t x = 0; x < model.n_expert; ++x) {
                    et.per_expert_srcs[x].push_back({ e.shard, e.off + static_cast<uint64_t>(x) * per_expert, per_expert, 0 });
                }
                model.expert.push_back(et);
            } else {
                dense_tensor_t dt;
                dt.name = name; dt.type = e.type; dt.size = e.size; dt.category = cat;
                for (int d = 0; d < 4; ++d) dt.ne[d] = e.ne[d];
                if (cat != tensor_category_t::DENSE_GLOBAL) dt.layer = sm_layer_of(name);
                dt.srcs.push_back({ e.shard, e.off, e.size, 0 });
                model.dense.push_back(dt);
            }
        }
    }

    // Sort experts by (layer, branch ORDER) like the writer.
    std::sort(model.expert.begin(), model.expert.end(),
              [](const expert_tensor_t& a, const expert_tensor_t& b) {
                  return a.layer != b.layer ? a.layer < b.layer : sm_branch_order(a.branch) < sm_branch_order(b.branch);
              });

    gguf_free(ctx0);
    return model;
}

model_t parse_model_path(const std::string& main_path) {
    return parse_model(std::vector<std::string>{ main_path });
}

} // namespace stream_moe

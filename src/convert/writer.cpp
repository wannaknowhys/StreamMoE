// writer.cpp - C++ converter write side (model_t -> target GGUF).
//
// Targets (docs/STREAMMOE_GGUF_FORMAT.md SS3):
//   V2       single-file expert-blocks (legacy, kept until v2 retires)
//   V3       single-file four category sections (C2/C1/C4/C3)
//   V3_CHUNK N strip files, every section split by the same 4K base/rem rule
//
// Reads bytes from model_t source segments; no staging process, no TCP.

#include "convert/writer.h"
#include "loader/layout_math.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  define SM_FSEEK _fseeki64
#  define SM_FTELL _ftelli64
#else
#  define SM_FSEEK fseeko
#  define SM_FTELL ftello
#endif

namespace stream_moe {

namespace {

constexpr uint64_t ALIGN = 4096;

uint64_t file_size(const std::string& path) {
    std::error_code ec;
    const uint64_t n = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
    if (ec) throw std::runtime_error("cannot stat " + path + ": " + ec.message());
    return n;
}

// ---------- gguf KV copy (skip prefixes) ----------
void set_kv_from_src(gguf_context* dst, const gguf_context* src, int64_t kid) {
    const char* key = gguf_get_key(src, kid);
    switch (gguf_get_kv_type(src, kid)) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8  (dst, key, gguf_get_val_u8  (src, kid)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8  (dst, key, gguf_get_val_i8  (src, kid)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16 (dst, key, gguf_get_val_u16 (src, kid)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16 (dst, key, gguf_get_val_i16 (src, kid)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32 (dst, key, gguf_get_val_u32 (src, kid)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32 (dst, key, gguf_get_val_i32 (src, kid)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64 (dst, key, gguf_get_val_u64 (src, kid)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64 (dst, key, gguf_get_val_i64 (src, kid)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (dst, key, gguf_get_val_f32 (src, kid)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(dst, key, gguf_get_val_bool(src, kid)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str (dst, key, gguf_get_val_str (src, kid)); break;
        case GGUF_TYPE_ARRAY: {
            const int at = gguf_get_arr_type(src, kid);
            const uint64_t n = gguf_get_arr_n(src, kid);
            switch (at) {
                case GGUF_TYPE_UINT8:   gguf_set_arr_data(dst, key, GGUF_TYPE_UINT8,   gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_INT8:    gguf_set_arr_data(dst, key, GGUF_TYPE_INT8,    gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_UINT16:  gguf_set_arr_data(dst, key, GGUF_TYPE_UINT16,  gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_INT16:   gguf_set_arr_data(dst, key, GGUF_TYPE_INT16,   gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_UINT32:  gguf_set_arr_data(dst, key, GGUF_TYPE_UINT32,  gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_INT32:   gguf_set_arr_data(dst, key, GGUF_TYPE_INT32,   gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_UINT64:  gguf_set_arr_data(dst, key, GGUF_TYPE_UINT64,  gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_INT64:   gguf_set_arr_data(dst, key, GGUF_TYPE_INT64,   gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_FLOAT32: gguf_set_arr_data(dst, key, GGUF_TYPE_FLOAT32, gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_BOOL:    gguf_set_arr_data(dst, key, GGUF_TYPE_BOOL,    gguf_get_arr_data(src, kid), n); break;
                case GGUF_TYPE_STRING: {
                    std::vector<const char*> cc(n);
                    for (uint64_t j = 0; j < n; ++j) cc[j] = gguf_get_arr_str(src, kid, j);
                    gguf_set_arr_str(dst, key, cc.data(), static_cast<size_t>(n));
                    break;
                }
                default: break;
            }
            break;
        }
        default: break;
    }
}

void copy_model_kv(gguf_context* dst, const gguf_context* src) {
    static const char* skip[] = { "split.", "stream_moe.", "general.alignment" };
    for (int i = 0; i < gguf_get_n_kv(src); ++i) {
        const std::string k = gguf_get_key(src, i);
        bool skipped = false;
        for (const char* s : skip) if (k.rfind(s, 0) == 0) { skipped = true; break; }
        if (!skipped) set_kv_from_src(dst, src, i);
    }
}

void set_u64_arr(gguf_context* dst, const char* key, const std::vector<uint64_t>& v) {
    gguf_set_arr_data(dst, key, GGUF_TYPE_UINT64, v.data(), v.size());
}
void set_str_arr(gguf_context* dst, const char* key, const std::vector<std::string>& v) {
    std::vector<const char*> cc;
    cc.reserve(v.size());
    for (const auto& s : v) cc.push_back(s.c_str());
    gguf_set_arr_str(dst, key, cc.data(), cc.size());
}

// ---------- tensor ordering ----------
struct item_t {
    bool is_expert = false;
    const dense_tensor_t*  d = nullptr;
    const expert_tensor_t* e = nullptr;
    uint64_t size = 0;
    const char* name = nullptr;
    const int64_t* ne = nullptr;
    int32_t type = 0;
};

std::vector<item_t> order_items(const model_t& model, convert_opts_t::target_t target) {
    std::vector<item_t> items;
    items.reserve(model.dense.size() + model.expert.size());
    auto push_dense = [&](const dense_tensor_t& d) {
        item_t it; it.is_expert = false; it.d = &d; it.size = d.size;
        it.name = d.name.c_str(); it.ne = d.ne; it.type = d.type;
        items.push_back(it);
    };
    if (target == convert_opts_t::target_t::V2) {
        for (const auto& d : model.dense) push_dense(d);
    } else {
        // C2 global, then C1 by layer, then C4 by layer (stable within a group)
        auto rank = [](tensor_category_t c) {
            switch (c) {
                case tensor_category_t::DENSE_GLOBAL: return 0;
                case tensor_category_t::DENSE_LAYER:  return 1;
                case tensor_category_t::EXPERT_META:  return 2;
                default:                              return 3;
            }
        };
        std::vector<size_t> idx(model.dense.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            const auto& da = model.dense[a];
            const auto& db = model.dense[b];
            if (rank(da.category) != rank(db.category)) return rank(da.category) < rank(db.category);
            return da.layer < db.layer;
        });
        for (size_t i : idx) push_dense(model.dense[i]);
    }
    for (const auto& e : model.expert) {
        item_t it; it.is_expert = true; it.e = &e; it.size = e.size;
        it.name = e.name.c_str(); it.ne = e.ne; it.type = e.type;
        items.push_back(it);
    }
    return items;
}

// ---------- v2/v3 layout plan ----------
struct branch_t { size_t expert_index; uint64_t branch_off; uint64_t per_expert; };

struct plan_t {
    std::vector<item_t> items;
    std::vector<uint64_t> off;                 // gguf-relative tensor offset (logical)
    std::vector<uint64_t> dense_global;        // [off, size] (v3)
    std::vector<uint64_t> dense_layer;         // flat [layer, off, size] (v3)
    std::vector<uint64_t> expert_meta;         // flat [layer, off, size] (v3)
    std::vector<uint64_t> expert_sections;     // flat [off, size, nsub]
    std::vector<std::vector<std::string>> bnames;
    std::vector<std::vector<uint64_t>>    bsizes;
    std::vector<uint64_t>                 bcounts; // per-layer branch count
    std::vector<uint64_t> et_branch_off;       // per model.expert index
    std::vector<uint32_t> block_layer, block_expert;
    std::vector<uint64_t> block_off, block_size;
    std::vector<std::vector<size_t>> layer_experts; // model.expert indices per layer
    uint64_t blocks_start = 0;
};

void add_section(std::vector<uint64_t>& flat, uint32_t layer, uint64_t off, uint64_t size) {
    flat.push_back(layer); flat.push_back(off); flat.push_back(size);
}

plan_t build_plan(const model_t& model, convert_opts_t::target_t target) {
    plan_t p;
    p.items = order_items(model, target);

    // logical offsets: gguf pads each tensor to alignment
    p.off.resize(p.items.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < p.items.size(); ++i) {
        p.off[i] = cur;
        cur = sm_align_up(cur + p.items[i].size, ALIGN);
    }

    // per-layer expert tensors (model.expert is sorted by layer, branch order)
    p.layer_experts.resize(model.n_layer);
    for (size_t i = 0; i < model.expert.size(); ++i) {
        const int32_t l = model.expert[i].layer;
        if (l >= 0 && l < static_cast<int32_t>(model.n_layer)) p.layer_experts[l].push_back(i);
    }
    p.et_branch_off.assign(model.expert.size(), 0);

    // branch layout per layer + expert blocks
    uint64_t blk_cur = 0;
    for (uint32_t l = 0; l < model.n_layer; ++l) {
        auto& ex = p.layer_experts[l];
        std::vector<branch_t> branches;
        uint64_t boff = 0;
        for (const auto& gi : ex) {
            branch_t b; b.expert_index = gi; b.branch_off = boff; b.per_expert = model.expert[gi].per_expert;
            p.et_branch_off[gi] = boff;
            boff = sm_align_up(boff + model.expert[gi].per_expert, ALIGN);
            branches.push_back(b);
        }
        const uint64_t bsize = boff;
        if (!ex.empty()) {
            std::vector<std::string> bn; std::vector<uint64_t> bz;
            for (const auto& gi : ex) { bn.push_back(model.expert[gi].name); bz.push_back(model.expert[gi].per_expert); }
            p.bnames.push_back(bn); p.bsizes.push_back(bz); p.bcounts.push_back(static_cast<uint64_t>(ex.size()));
        } else {
            p.bnames.push_back({}); p.bsizes.push_back({}); p.bcounts.push_back(0);
        }
        for (uint32_t e = 0; e < model.n_expert; ++e) {
            p.block_layer.push_back(l);
            p.block_expert.push_back(e);
            p.block_off.push_back(blk_cur);
            p.block_size.push_back(bsize);
            p.expert_sections.push_back(blk_cur);
            p.expert_sections.push_back(bsize);
            p.expert_sections.push_back(static_cast<uint64_t>(ex.size()));
            blk_cur += bsize;
        }
    }

    // v3 category sections + block start
    uint64_t real_end = 0;
    for (size_t i = 0; i < p.items.size(); ++i)
        if (!p.items[i].is_expert) real_end = std::max(real_end, p.off[i] + p.items[i].size);
    p.blocks_start = sm_align_up(real_end, ALIGN);
    // rebase blocks to start after the dense sections
    for (auto& o : p.block_off) o += p.blocks_start;
    for (size_t i = 0; i + 3 <= p.expert_sections.size(); i += 3) p.expert_sections[i] += p.blocks_start;

    if (target != convert_opts_t::target_t::V2) {
        // C2 global
        uint64_t g_first = UINT64_MAX, g_last = 0;
        for (size_t i = 0; i < p.items.size(); ++i) {
            const auto& it = p.items[i];
            if (it.is_expert || it.d->category != tensor_category_t::DENSE_GLOBAL) continue;
            g_first = std::min(g_first, p.off[i]);
            g_last = std::max(g_last, p.off[i] + it.size);
        }
        if (g_first != UINT64_MAX) { p.dense_global.push_back(g_first); p.dense_global.push_back(sm_align_up(g_last, ALIGN) - g_first); }
        // C1 / C4 per layer
        for (uint32_t l = 0; l < model.n_layer; ++l) {
            for (int pass = 0; pass < 2; ++pass) {
                const tensor_category_t cat = pass == 0 ? tensor_category_t::DENSE_LAYER : tensor_category_t::EXPERT_META;
                uint64_t first = UINT64_MAX, last = 0;
                for (size_t i = 0; i < p.items.size(); ++i) {
                    const auto& it = p.items[i];
                    if (it.is_expert || it.d->category != cat || it.d->layer != static_cast<int32_t>(l)) continue;
                    first = std::min(first, p.off[i]);
                    last = std::max(last, p.off[i] + it.size);
                }
                if (first != UINT64_MAX) {
                    add_section(pass == 0 ? p.dense_layer : p.expert_meta, l, first, sm_align_up(last, ALIGN) - first);
                }
            }
        }
    } else {
        // v2: one merged dense section [0, blocks_start)
        p.dense_global.push_back(0);
        p.dense_global.push_back(p.blocks_start);
    }
    return p;
}

// ---------- header write ----------
// ggml has no public "set alignment" in upstream. Seed a context from a minimal
// in-memory GGUF that already carries general.alignment, so gguf_init reads it
// into ctx->alignment and gguf_add_tensor lays tensors out 4K-aligned. This
// keeps the converter free of the vendored STREAM_MOE_GGUF_ALIGN patch.
gguf_context* gguf_init_with_alignment(uint64_t alignment) {
    static const std::vector<uint8_t> seed = [&] {
        std::vector<uint8_t> b;
        auto put = [&](const void* p, size_t n) { b.insert(b.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); };
        auto put_u32 = [&](uint32_t v) { put(&v, 4); };
        auto put_u64 = [&](uint64_t v) { put(&v, 8); };
        auto put_str = [&](const char* s) { put_u64(std::strlen(s)); put(s, std::strlen(s)); };
        put("GGUF", 4);
        put_u32(3);            // version
        put_u64(0);            // n_tensors
        put_u64(1);            // n_kv
        put_str("general.alignment");
        put_u32(4);            // GGUF_TYPE_UINT32
        put_u32(static_cast<uint32_t>(alignment));
        return b;
    }();
    gguf_init_params prm = { /*no_alloc=*/ true, nullptr };
    gguf_context* ctx = gguf_init_from_buffer(seed.data(), seed.size(), prm);
    if (!ctx) throw std::runtime_error("gguf_init_with_alignment failed");
    return ctx;
}

gguf_context* make_header(const model_t& model, const plan_t& p, bool is_v3,
                          const std::vector<uint64_t>* chunk_slices,
                          int chunk_no, int chunk_total) {
    gguf_init_params prm = { true, nullptr };
    gguf_context* src = gguf_init_from_file(model.files[0].c_str(), prm);
    if (!src) throw std::runtime_error("cannot open source " + model.files[0]);

    gguf_context* dst = gguf_init_with_alignment(ALIGN);
    copy_model_kv(dst, src);
    gguf_free(src);
    // Re-append general.alignment after the copied model KV so the header KV
    // order matches the legacy writer (ctx->alignment is already 4096).
    gguf_set_val_u32(dst, "general.alignment", static_cast<uint32_t>(ALIGN));

    ggml_init_params gp = { 16 * 1024 * 1024, nullptr, true };
    ggml_context* gctx = ggml_init(gp);
    if (!gctx) throw std::runtime_error("ggml_init failed");
    for (const auto& it : p.items) {
        int nd = 1; int64_t ne[4] = { 1, 1, 1, 1 };
        for (int d = 0; d < 4; ++d) ne[d] = it.ne[d];
        for (int d = 3; d >= 1; --d) if (ne[d] != 1) { nd = d + 1; break; }
        ggml_tensor* gt = ggml_new_tensor(gctx, static_cast<ggml_type>(it.type), nd, ne);
        if (!gt) throw std::runtime_error(std::string("ggml_new_tensor failed: ") + it.name);
        ggml_set_name(gt, it.name);
        gguf_add_tensor(dst, gt);
    }
    ggml_free(gctx);

    if (is_v3) {
        gguf_set_val_str(dst, "stream_moe.layout", "v3");
        set_u64_arr(dst, "stream_moe.dense_global_section", p.dense_global);
        set_u64_arr(dst, "stream_moe.dense_layer_sections", p.dense_layer);
        set_u64_arr(dst, "stream_moe.expert_meta_sections", p.expert_meta);
    } else {
        gguf_set_val_str(dst, "stream_moe.layout", "expert-blocks-v2");
        set_u64_arr(dst, "stream_moe.dense_section", p.dense_global);
    }
    set_u64_arr(dst, "stream_moe.expert_sections", p.expert_sections);
    {
        std::vector<std::string> bnames;
        std::vector<uint64_t> bsizes, bcounts;
        for (uint32_t l = 0; l < model.n_layer; ++l) {
            bcounts.push_back(p.bcounts.empty() ? 0 : p.bcounts[l]);
            for (size_t k = 0; k < p.bnames[l].size(); ++k) { bnames.push_back(p.bnames[l][k]); bsizes.push_back(p.bsizes[l][k]); }
        }
        set_str_arr(dst, "stream_moe.expert_branch_names", bnames);
        set_u64_arr(dst, "stream_moe.expert_branch_sizes", bsizes);
        set_u64_arr(dst, "stream_moe.expert_branch_counts", bcounts);
    }
    gguf_set_val_u32(dst, "stream_moe.branch_align", 1);
    if (chunk_slices) {
        gguf_set_val_u32(dst, "stream_moe.chunk_no", static_cast<uint32_t>(chunk_no));
        gguf_set_val_u32(dst, "stream_moe.chunk_total", static_cast<uint32_t>(chunk_total));
        gguf_set_val_u32(dst, "stream_moe.incomplete", 1);
        set_u64_arr(dst, "stream_moe.chunk_slices", *chunk_slices);
    }
    return dst;
}

void write_header_to(const model_t& model, const plan_t& p, bool is_v3, const std::string& out,
                     const std::vector<uint64_t>* chunk_slices, int chunk_no, int chunk_total) {
    gguf_context* dst = make_header(model, p, is_v3, chunk_slices, chunk_no, chunk_total);
    if (!gguf_write_to_file(dst, out.c_str(), /*only_meta=*/ true)) {
        gguf_free(dst);
        throw std::runtime_error("gguf_write_to_file failed: " + out);
    }
    gguf_free(dst);
}

// ---------- byte copy / fill ----------
struct copy_op_t { uint32_t file; uint64_t src_off, len, dst_rel; };

void do_copy(const std::string& out, uint64_t data_offset,
             const std::vector<std::string>& files, const std::vector<copy_op_t>& ops) {
    std::FILE* of = std::fopen(out.c_str(), "r+b");
    if (!of) throw std::runtime_error("cannot reopen " + out);
    std::map<uint32_t, std::FILE*> sf;
    std::vector<char> buf(16 * 1024 * 1024);
    try {
        for (const auto& op : ops) {
            auto it = sf.find(op.file);
            if (it == sf.end()) {
                std::FILE* f = std::fopen(files[op.file].c_str(), "rb");
                if (!f) throw std::runtime_error("cannot open src " + files[op.file]);
                it = sf.emplace(op.file, f).first;
            }
            if (SM_FSEEK(it->second, static_cast<long long>(op.src_off), SEEK_SET) != 0)
                throw std::runtime_error("seek src");
            if (SM_FSEEK(of, static_cast<long long>(data_offset + op.dst_rel), SEEK_SET) != 0)
                throw std::runtime_error("seek dst");
            uint64_t left = op.len;
            while (left) {
                const size_t n = left < buf.size() ? static_cast<size_t>(left) : buf.size();
                if (std::fread(buf.data(), 1, n, it->second) != n)
                    throw std::runtime_error("short read from " + files[op.file]);
                if (std::fwrite(buf.data(), 1, n, of) != n) throw std::runtime_error("short write to " + out);
                left -= n;
            }
        }
        std::fclose(of);
        for (auto& kv : sf) std::fclose(kv.second);
    } catch (...) {
        std::fclose(of);
        for (auto& kv : sf) if (kv.second) std::fclose(kv.second);
        throw;
    }
}

void do_fill(const std::string& out, uint64_t data_offset,
             const std::vector<std::pair<uint64_t, uint64_t>>& ops) {
    if (ops.empty()) return;
    std::FILE* of = std::fopen(out.c_str(), "r+b");
    if (!of) throw std::runtime_error("cannot reopen " + out);
    static const char zeros[ALIGN] = { 0 };
    try {
        for (const auto& op : ops) {
            if (op.second == 0) continue;
            if (SM_FSEEK(of, static_cast<long long>(data_offset + op.first), SEEK_SET) != 0)
                throw std::runtime_error("seek fill");
            uint64_t left = op.second;
            while (left) {
                const size_t n = left < sizeof(zeros) ? static_cast<size_t>(left) : sizeof(zeros);
                if (std::fwrite(zeros, 1, n, of) != n) throw std::runtime_error("short fill");
                left -= n;
            }
        }
        std::fclose(of);
    } catch (...) { std::fclose(of); throw; }
}

// ---------- v2 / v3 single-file ----------
void write_single(const model_t& model, const std::string& out, convert_opts_t::target_t target) {
    const plan_t p = build_plan(model, target);
    write_header_to(model, p, target != convert_opts_t::target_t::V2, out, nullptr, 0, 0);
    const uint64_t data_offset = sm_align_up(file_size(out), ALIGN);

    std::vector<copy_op_t> ops;
    for (size_t i = 0; i < p.items.size(); ++i) {
        const auto& it = p.items[i];
        if (it.is_expert) continue;
        for (const auto& s : it.d->srcs) ops.push_back({ s.file, s.off, s.len, p.off[i] + s.in_off });
    }
    for (size_t bi = 0; bi < p.block_off.size(); ++bi) {
        const uint32_t l = p.block_layer[bi];
        const uint32_t e = p.block_expert[bi];
        for (const auto gi : p.layer_experts[l]) {
            const auto& et = model.expert[gi];
            const uint64_t dst = p.block_off[bi] + p.et_branch_off[gi];
            for (const auto& s : et.per_expert_srcs[e]) ops.push_back({ s.file, s.off, s.len, dst + s.in_off });
        }
    }
    do_copy(out, data_offset, model.files, ops);

    // fill inter-branch pads and block tails (DIO windows never read stale bytes)
    std::vector<std::pair<uint64_t, uint64_t>> fills;
    for (size_t bi = 0; bi < p.block_off.size(); ++bi) {
        const uint32_t l = p.block_layer[bi];
        uint64_t cursor = 0;
        for (const auto gi : p.layer_experts[l]) {
            const uint64_t boff = p.et_branch_off[gi];
            if (boff > cursor) fills.emplace_back(p.block_off[bi] + cursor, boff - cursor);
            cursor = boff + model.expert[gi].per_expert;
        }
        if (cursor < p.block_size[bi]) fills.emplace_back(p.block_off[bi] + cursor, p.block_size[bi] - cursor);
    }
    do_fill(out, data_offset, fills);
}

// ---------- v3 chunk ----------
struct content_item_t { uint64_t log_off, len; const std::vector<src_seg_t>* srcs; };
struct unit_t { uint64_t off, size; std::vector<content_item_t> content; };

std::vector<unit_t> build_units(const model_t& model, const plan_t& p) {
    std::vector<unit_t> units;
    // global
    {
        unit_t u; u.off = p.dense_global[0]; u.size = p.dense_global[1];
        for (size_t i = 0; i < p.items.size(); ++i) {
            const auto& it = p.items[i];
            if (it.is_expert || it.d->category != tensor_category_t::DENSE_GLOBAL) continue;
            u.content.push_back({ p.off[i] - u.off, it.size, &it.d->srcs });
        }
        units.push_back(u);
    }
    // C1 / C4 per layer (only sections present)
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<uint64_t>& secs = pass == 0 ? p.dense_layer : p.expert_meta;
        const tensor_category_t cat = pass == 0 ? tensor_category_t::DENSE_LAYER : tensor_category_t::EXPERT_META;
        for (size_t k = 0; k + 3 <= secs.size(); k += 3) {
            unit_t u; u.off = secs[k + 1]; u.size = secs[k + 2];
            const int32_t layer = static_cast<int32_t>(secs[k]);
            for (size_t i = 0; i < p.items.size(); ++i) {
                const auto& it = p.items[i];
                if (it.is_expert || it.d->category != cat || it.d->layer != layer) continue;
                u.content.push_back({ p.off[i] - u.off, it.size, &it.d->srcs });
            }
            units.push_back(u);
        }
    }
    // expert blocks
    for (size_t bi = 0; bi < p.block_off.size(); ++bi) {
        unit_t u; u.off = p.block_off[bi]; u.size = p.block_size[bi];
        const uint32_t l = p.block_layer[bi];
        const uint32_t e = p.block_expert[bi];
        for (const auto gi : p.layer_experts[l]) {
            const auto& et = model.expert[gi];
            u.content.push_back({ p.et_branch_off[gi], et.per_expert, &et.per_expert_srcs[e] });
        }
        units.push_back(u);
    }
    return units;
}

// map logical [start, start+len) of a unit to copy ops at dst_base
void content_to_ops(const std::vector<content_item_t>& content, uint64_t start, uint64_t len,
                    uint64_t dst_base, std::vector<copy_op_t>& ops) {
    for (const auto& item : content) {
        const uint64_t c_start = item.log_off, c_end = item.log_off + item.len;
        if (c_end <= start || c_start >= start + len) continue;
        const uint64_t s = std::max(start, c_start), e = std::min(start + len, c_end);
        const uint64_t rel_start = s - c_start, rel_end = e - c_start;
        for (const auto& src : *item.srcs) {
            const uint64_t ss_start = src.in_off, ss_end = src.in_off + src.len;
            if (ss_end <= rel_start || ss_start >= rel_end) continue;
            const uint64_t a = std::max(rel_start, ss_start), b = std::min(rel_end, ss_end);
            ops.push_back({ src.file, src.off + (a - ss_start), b - a, dst_base + (s - start) + (a - rel_start) });
        }
    }
}

std::string zero_pad(long num, int width) {
    std::string s = std::to_string(num);
    if (static_cast<int>(s.size()) < width) s = std::string(static_cast<size_t>(width) - s.size(), '0') + s;
    return s;
}

// Chunk-name digit width: mirror the source filename's trailing number width
// (e.g. "-00005" -> 5); default 5 (GGUF-style) when the source has none. The
// number grows naturally past the width (printf-style).
int infer_chunk_pad_width(const std::string& input_path) {
    const std::string stem = std::filesystem::path(input_path).stem().string();
    size_t e = stem.size();
    while (e > 0 && std::isdigit(static_cast<unsigned char>(stem[e - 1]))) --e;
    if (e < stem.size()) return static_cast<int>(stem.size() - e);
    return 5;
}

void write_v3_chunk(const model_t& model, const std::string& outBase, int N,
                    const std::vector<int>& ratio, int pad) {
    const plan_t p = build_plan(model, convert_opts_t::target_t::V3);
    const std::vector<unit_t> units = build_units(model, p);

    // per-file per-unit strip counts
    std::vector<std::vector<int>> slices(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) slices[static_cast<size_t>(i)].resize(units.size());
    for (size_t u = 0; u < units.size(); ++u) {
        if (units[u].size % ALIGN != 0) throw std::runtime_error("v3chunk: unit size not 4K aligned");
        const std::vector<int> sp = sm_split_blocks(units[u].size / ALIGN, N, ratio);
        for (int i = 0; i < N; ++i) slices[static_cast<size_t>(i)][u] = sp[static_cast<size_t>(i)];
    }

    for (int i = 0; i < N; ++i) {
        const std::string out = outBase + "-" + zero_pad(i + 1, pad) + ".gguf";
        const std::vector<uint64_t> cs(slices[static_cast<size_t>(i)].begin(), slices[static_cast<size_t>(i)].end());
        write_header_to(model, p, true, out, &cs, i, N);
        const uint64_t data_offset = sm_align_up(file_size(out), ALIGN);

        // per-file cumulative blocks before each unit
        std::vector<uint64_t> cum(units.size() + 1, 0);
        for (size_t u = 0; u < units.size(); ++u) cum[u + 1] = cum[u] + static_cast<uint64_t>(slices[static_cast<size_t>(i)][u]);

        std::vector<copy_op_t> ops;
        std::vector<std::pair<uint64_t, uint64_t>> fills;
        for (size_t u = 0; u < units.size(); ++u) {
            const uint64_t fileBase = cum[u] * ALIGN;
            const uint64_t stripLen = static_cast<uint64_t>(slices[static_cast<size_t>(i)][u]) * ALIGN;
            if (stripLen == 0) continue;
            // logical start of this file's strip inside the unit (unit-relative,
            // matching content_item_t::log_off)
            uint64_t bStart = 0;
            for (int f = 0; f < i; ++f) bStart += static_cast<uint64_t>(slices[static_cast<size_t>(f)][u]);
            const uint64_t logStart = bStart * ALIGN;

            const size_t ops_before = ops.size();
            content_to_ops(units[u].content, logStart, stripLen, fileBase, ops);
            // fill complement of covered data within the strip (strip-relative)
            std::vector<std::pair<uint64_t, uint64_t>> covered;
            for (size_t k = ops_before; k < ops.size(); ++k) {
                const uint64_t d = ops[k].dst_rel - fileBase;
                covered.emplace_back(d, d + ops[k].len);
            }
            std::sort(covered.begin(), covered.end());
            uint64_t cursor = 0;
            for (const auto& c : covered) {
                if (c.first > cursor) fills.emplace_back(fileBase + cursor, c.first - cursor);
                cursor = std::max(cursor, c.second);
            }
            if (cursor < stripLen) fills.emplace_back(fileBase + cursor, stripLen - cursor);
        }
        do_copy(out, data_offset, model.files, ops);
        do_fill(out, data_offset, fills);
    }
}

} // namespace

void convert_model(const model_t& model, const convert_opts_t& opts, const std::string& out) {
    if (opts.target == convert_opts_t::target_t::V3_CHUNK) {
        if (opts.chunks < 1) throw std::runtime_error("chunks must be >= 1");
        const std::filesystem::path op(out);
        if (!op.parent_path().empty()) std::filesystem::create_directories(op.parent_path());
        const int pad = infer_chunk_pad_width(model.files.empty() ? std::string() : model.files[0]);
        write_v3_chunk(model, out, opts.chunks, opts.ratio, pad);
    } else {
        write_single(model, out, opts.target);
    }
}

} // namespace stream_moe

// stream_moe_convertd.cpp - C/C++ wrapper around the ggml gguf library.
// JSON-lines protocol over stdin/stdout. Spawned by tools/stream_moe_convert.js.
// Commands (one JSON per line on stdin):
//   {"cmd":"open","path":"..."}       -> {"ok":true,"meta":{...}}  (GGUF metadata summary)
//   {"cmd":"convert","format":"v1","in":"p1;p2","out":"out.gguf"}   -> v1 sections writer
//   {"cmd":"chunk",...}               -> v2 RAID0 chunking (TODO)
//
// v1 (sections-v1): GGUF superset. alignment=4096, dense (non-_exps, source
// order) section then expert (_exps, source order) section, each tensor 4K
// aligned, stream_moe.* KV. Uses gguf_set_alignment() (vendored ggml extension)
// so gguf_add_tensor / gguf_write_to_file lay out 4K-aligned offsets; tensor
// DATA is streamed manually (metadata-only write, then seek+copy per tensor).
//
// Build (Windows, clang-cl + vendored ggml):
//   clang-cl /std:c++17 tools/stream_moe_convertd.cpp /EHsc /MT ^
//       "/IF:/Dev/StreamMoE/third_party/llama.cpp/ggml/include" ^
//       "/IF:/Dev/StreamMoE/third_party/llama.cpp/ggml/src" ^
//       F:/Dev/StreamMoE/build/main/llama-build/ggml/src/ggml-base.lib ^
//       F:/Dev/LLVM/lib/libomp.lib /Fe:temp/stream_moe_convertd.exe
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

// ---------- minimal JSON (read: flat object of string/num; write: we build manually) ----------
static std::string jstr(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += (char) c;
    }
    return out + "\"";
}

static std::string find_str(const std::string & json, const std::string & key) {
    const std::string pat = "\"" + key + "\":\"";
    size_t p = json.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    std::string v;
    while (p < json.size() && json[p] != '"') {
        if (json[p] == '\\') {
            if (p + 1 < json.size() && json[p + 1] == '\\') { v += '\\'; p += 2; }
            else { v += json[p]; p++; }
        }
        else { v += json[p]; p++; }
    }
    return v;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) { if (c == sep) { out.push_back(cur); cur.clear(); } else cur.push_back(c); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// ---------- gguf metadata -> JSON ----------
static void emit_meta(const gguf_context * ctx, std::string & out) {
    out += "\"n_kv\":";
    out += std::to_string(gguf_get_n_kv(ctx));
    out += ",\"alignment\":";
    out += std::to_string(gguf_get_alignment(ctx));
    out += ",\"n_tensors\":";
    out += std::to_string(gguf_get_n_tensors(ctx));
    out += ",\"kv\":[";
    for (int i = 0; i < gguf_get_n_kv(ctx); ++i) {
        if (i) out += ",";
        out += "{";
        out += "\"k\":" + jstr(gguf_get_key(ctx, i));
        const int t = gguf_get_kv_type(ctx, i);
        out += ",\"t\":" + std::to_string(t);
        switch (t) {
            case GGUF_TYPE_UINT32: out += ",\"v\":" + std::to_string(gguf_get_val_u32(ctx, i)); break;
            case GGUF_TYPE_INT32:  out += ",\"v\":" + std::to_string(gguf_get_val_i32(ctx, i)); break;
            case GGUF_TYPE_UINT64: out += ",\"v\":" + std::to_string(gguf_get_val_u64(ctx, i)); break;
            case GGUF_TYPE_INT64:  out += ",\"v\":" + std::to_string(gguf_get_val_i64(ctx, i)); break;
            case GGUF_TYPE_FLOAT32:out += ",\"v\":" + std::to_string(gguf_get_val_f32(ctx, i)); break;
            case GGUF_TYPE_BOOL:   out += ",\"v\":" + std::to_string(gguf_get_val_bool(ctx, i)); break;
            case GGUF_TYPE_STRING: out += ",\"v\":" + jstr(gguf_get_val_str(ctx, i)); break;
            default: out += ",\"v\":null"; break;
        }
        out += "}";
    }
    out += "],\"tensors\":[";
    for (int i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        if (i) out += ",";
        out += "{\"name\":" + jstr(gguf_get_tensor_name(ctx, i));
        out += ",\"offset\":" + std::to_string(gguf_get_tensor_offset(ctx, i));
        out += ",\"size\":" + std::to_string(gguf_get_tensor_size(ctx, i));
        out += ",\"type\":" + std::to_string(gguf_get_tensor_type(ctx, i));
        out += "}";
    }
    out += "]";
}

static void cmd_open(const std::string & path, std::string & out) {
    gguf_init_params params = { true, nullptr }; // no_alloc: metadata only
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        out = "{\"ok\":false,\"error\":\"gguf_init_from_file failed\"}";
        return;
    }
    out = "{\"ok\":true,\"meta\":{";
    emit_meta(ctx, out);
    out += "}}";
    gguf_free(ctx);
}

// ---------- v1 convert (gguf API + streamed tensor data) ----------
static const size_t ALIGN = 4096;
static size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

static void set_kv_from_src(gguf_context * dst, const gguf_context * src, int64_t kid) {
    const char * key = gguf_get_key(src, kid);
    switch (gguf_get_kv_type(src, kid)) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8  (dst, key, gguf_get_val_u8  (src, kid)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8  (dst, key, gguf_get_val_i8  (src, kid)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16 (dst, key, gguf_get_val_u16 (src, kid)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16 (dst, key, gguf_get_val_i16 (src, kid)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32 (dst, key, gguf_get_val_u32 (src, kid)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32 (dst, key, gguf_get_val_i32 (src, kid)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (dst, key, gguf_get_val_f32 (src, kid)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(dst, key, gguf_get_val_bool(src, kid)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str (dst, key, gguf_get_val_str (src, kid)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64 (dst, key, gguf_get_val_u64 (src, kid)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64 (dst, key, gguf_get_val_i64 (src, kid)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (dst, key, gguf_get_val_f64 (src, kid)); break;
        case GGUF_TYPE_ARRAY: {
            enum gguf_type et = gguf_get_arr_type(src, kid);
            const size_t n = gguf_get_arr_n(src, kid);
            if (et == GGUF_TYPE_STRING) {
                std::vector<const char *> strs(n);
                for (size_t i = 0; i < n; i++) strs[i] = gguf_get_arr_str(src, kid, i);
                gguf_set_arr_str(dst, key, strs.data(), n);
            } else {
                gguf_set_arr_data(dst, key, et, gguf_get_arr_data(src, kid), n);
            }
            break;
        }
        default: throw std::runtime_error("unsupported KV type in source metadata");
    }
}

static int tensor_ndims(const int64_t * ne) {
    int nd = 1;
    for (int d = 0; d < 4; d++) if (ne[d] > 1) nd = d + 1;
    return nd;
}

// ---------- v2 (expert-blocks-v2): dense normal; each expert one compact block ----------
// Per-expert block = concatenation of that expert's branch slices, branch order
// gate_up, gate, up, down, scale. expert_sections (u64[] flattened [off,size,nsub]
// per block, offsets relative to data start). Not upstream-readable.
static void cmd_convert_v2(const std::string & json, std::string & out) {
    try {
        const std::string in = find_str(json, "in");
        const std::string outp = find_str(json, "out");
        std::vector<std::string> shards = split(in, ';');
        if (shards.empty() || outp.empty()) { out = "{\"ok\":false,\"error\":\"need in;out\"}"; return; }

        std::vector<gguf_context *> srcs;
        for (auto & p : shards) {
            gguf_init_params prm = { true, nullptr };
            gguf_context * c = gguf_init_from_file(p.c_str(), prm);
            if (!c) { for (auto * x : srcs) gguf_free(x); out = "{\"ok\":false,\"error\":\"cannot open " + p + "\"}"; return; }
            srcs.push_back(c);
        }

        struct st { std::string name; std::vector<int64_t> ne; ggml_type type; size_t shard; int idx; };
        std::vector<st> dense, expert;
        for (size_t si = 0; si < srcs.size(); si++) {
            const int nt = gguf_get_n_tensors(srcs[si]);
            for (int i = 0; i < nt; i++) {
                st t;
                t.name = gguf_get_tensor_name(srcs[si], i);
                const int64_t * ne = gguf_get_tensor_ne(srcs[si], i);
                for (int d = 0; d < 4; d++) t.ne.push_back(ne[d]);
                t.type  = gguf_get_tensor_type(srcs[si], i);
                t.shard = si;
                t.idx   = i;
                // expert blocks pack the routed GEMM weights only; per-expert
                // scale tensors (*_exps.scale) stay contiguous so the graph reads
                // real data (they are not MUL_MAT_ID weights).
                const bool is_expert_tensor = t.name.find("_exps") != std::string::npos && t.name.find(".scale") == std::string::npos;
                (is_expert_tensor ? expert : dense).push_back(t);
            }
        }

        // n_expert from KV (<arch>.expert_count)
        int64_t n_expert = 0;
        for (auto * s : srcs) {
            for (int i = 0; i < gguf_get_n_kv(s); i++) {
                const std::string k = gguf_get_key(s, i);
                if (k.find("expert_count") != std::string::npos) {
                    if (gguf_get_kv_type(s, i) == GGUF_TYPE_INT32) { n_expert = gguf_get_val_i32(s, i); break; }
                    if (gguf_get_kv_type(s, i) == GGUF_TYPE_UINT32) { n_expert = (int64_t) gguf_get_val_u32(s, i); break; }
                }
            }
            if (n_expert) break;
        }
        if (!n_expert) { out = "{\"ok\":false,\"error\":\"cannot find expert_count KV\"}"; return; }

        // target gguf
        gguf_context * dst = gguf_init_empty();
        gguf_set_alignment(dst, ALIGN);
        for (int i = 0; i < gguf_get_n_kv(srcs[0]); i++) {
            const std::string k = gguf_get_key(srcs[0], i);
            if (k.rfind("split.", 0) == 0) continue;
            if (k == "general.alignment") continue;
            set_kv_from_src(dst, srcs[0], i);
        }
        gguf_set_val_u32(dst, "general.alignment", (uint32_t) ALIGN);
        gguf_set_val_str(dst, "stream_moe.layout", "expert-blocks-v2");

        // add dense tensors + expert tensor metadata (expert data lives in the
        // compact blocks below; tensor_info exists so llama.cpp graph builds,
        // route B never reads tensor data). Offsets for expert tensors occupy a
        // gap that the block section reuses conceptually.
        ggml_init_params gprm = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(gprm);
        if (!gctx) throw std::runtime_error("ggml_init failed");
        auto add_one = [&](const st & t) {
            ggml_tensor * gt = ggml_new_tensor(gctx, t.type, tensor_ndims(t.ne.data()), t.ne.data());
            ggml_set_name(gt, t.name.c_str());
            gguf_add_tensor(dst, gt);
        };
        for (auto & t : dense)  add_one(t);
        for (auto & t : expert) add_one(t);
        ggml_free(gctx);

        // dense section end (data-relative) over ALL tensors (dense + expert metadata gap)
        size_t denseEnd = 0;
        const size_t nAdd = dense.size() + expert.size();
        for (size_t i = 0; i < nAdd; i++) {
            const size_t e = gguf_get_tensor_offset(dst, i) + gguf_get_tensor_size(dst, i);
            if (e > denseEnd) denseEnd = e;
        }
        denseEnd = align_up(denseEnd, ALIGN);

        // group expert tensors by layer + branch tag; per-expert slice size
        struct branch_t { std::string tag; st t; size_t per_expert; };
        std::map<int, std::vector<branch_t>> byLayer;
        for (auto & t : expert) {
            int layer = -1;
            const size_t p = t.name.find("blk.");
            if (p != std::string::npos) layer = std::atoi(t.name.c_str() + p + 4);
            std::string tag;
            if (t.name.find("gate_up") != std::string::npos) tag = "gate_up";
            else if (t.name.find("ffn_gate_exps") != std::string::npos) tag = "gate";
            else if (t.name.find("ffn_up_exps") != std::string::npos) tag = "up";
            else if (t.name.find("down_exps") != std::string::npos) tag = t.name.find(".scale") != std::string::npos ? "scale" : "down";
            else tag = t.name;
            branch_t b;
            b.tag = tag;
            b.t = t;
            b.per_expert = gguf_get_tensor_size(srcs[t.shard], t.idx) / (size_t) n_expert;
            byLayer[layer].push_back(b);
        }
        const std::vector<std::string> order = { "gate_up", "gate", "up", "down", "scale" };

        // per-expert blocks (4K-aligned), offsets relative to data start
        struct block_t { size_t off; size_t size; size_t nsub; };
        std::vector<block_t> blocks;
        size_t cur = denseEnd;
        for (auto & [layer, branches] : byLayer) {
            size_t raw = 0, nsub = 0;
            for (auto & ob : order)
                for (auto & b : branches) if (b.tag == ob) { raw += b.per_expert; nsub++; break; }
            for (int64_t e = 0; e < n_expert; e++) {
                blocks.push_back({ cur, align_up(raw, ALIGN), nsub });
                cur = align_up(cur + raw, ALIGN);
            }
        }
        const size_t nBlocks = blocks.size();
        std::vector<uint64_t> sec(nBlocks * 3);
        for (size_t i = 0; i < nBlocks; i++) { sec[i * 3] = blocks[i].off; sec[i * 3 + 1] = blocks[i].size; sec[i * 3 + 2] = blocks[i].nsub; }
        gguf_set_arr_data(dst, "stream_moe.expert_sections", GGUF_TYPE_UINT64, sec.data(), sec.size());
        uint64_t dsec[2] = { 0, denseEnd };
        gguf_set_arr_data(dst, "stream_moe.dense_section", GGUF_TYPE_UINT64, dsec, 2);

        // per-layer per-branch expert sizes + FULL tensor names (block-internal
        // layout + delegate branch_layout matching against the graph weight name)
        std::vector<std::string> bnames;
        std::vector<uint64_t> bsizes;
        for (auto & [layer, branches] : byLayer) {
            for (auto & ob : order) {
                for (auto & b : branches) if (b.tag == ob) {
                    bnames.push_back(b.t.name); // e.g. "blk.0.ffn_gate_up_exps.weight"
                    bsizes.push_back((uint64_t) b.per_expert);
                    break;
                }
            }
        }
        std::vector<const char *> bnames_c;
        for (auto & n : bnames) bnames_c.push_back(n.c_str());
        gguf_set_arr_str(dst, "stream_moe.expert_branch_names", bnames_c.data(), bnames_c.size());
        gguf_set_arr_data(dst, "stream_moe.expert_branch_sizes", GGUF_TYPE_UINT64, bsizes.data(), bsizes.size());

        if (!gguf_write_to_file(dst, outp.c_str(), /*only_meta=*/ true)) throw std::runtime_error("write meta failed");
        FILE * outF = std::fopen(outp.c_str(), "r+b");
        if (!outF) throw std::runtime_error("reopen output");
        _fseeki64(outF, 0, SEEK_END);
        const size_t dataStart = align_up((size_t) _ftelli64(outF), ALIGN);

        std::vector<FILE *> srcF;
        for (auto & p : shards) { FILE * f = std::fopen(p.c_str(), "rb"); if (!f) throw std::runtime_error("open " + p); srcF.push_back(f); }
        std::vector<char> buf(64 * 1024 * 1024);
        auto copy_from = [&](size_t shard, size_t pos, size_t size, size_t dstPos) {
            FILE * sf = srcF[shard];
            _fseeki64(sf, pos, SEEK_SET);
            _fseeki64(outF, dstPos, SEEK_SET);
            size_t left = size;
            while (left) {
                const size_t n = left < buf.size() ? left : buf.size();
                if (std::fread(buf.data(), 1, n, sf) != n) throw std::runtime_error("read");
                if (std::fwrite(buf.data(), 1, n, outF) != n) throw std::runtime_error("write");
                left -= n;
            }
        };

        // dense data
        for (size_t i = 0; i < dense.size(); i++) {
            auto & t = dense[i];
            const size_t srcPos = gguf_get_data_offset(srcs[t.shard]) + gguf_get_tensor_offset(srcs[t.shard], t.idx);
            const size_t size   = gguf_get_tensor_size(srcs[t.shard], t.idx);
            copy_from(t.shard, srcPos, size, dataStart + gguf_get_tensor_offset(dst, i));
        }
        // expert blocks
        size_t bi = 0;
        for (auto & [layer, branches] : byLayer) {
            for (int64_t e = 0; e < n_expert; e++, bi++) {
                size_t dstPos = dataStart + blocks[bi].off;
                size_t written = 0;
                for (auto & ob : order) {
                    for (auto & b : branches) if (b.tag == ob) {
                        const size_t srcPos = gguf_get_data_offset(srcs[b.t.shard]) + gguf_get_tensor_offset(srcs[b.t.shard], b.t.idx) + (size_t) e * b.per_expert;
                        copy_from(b.t.shard, srcPos, b.per_expert, dstPos + written);
                        written += b.per_expert;
                        break;
                    }
                }
            }
        }
        std::fclose(outF);
        for (auto * f : srcF) std::fclose(f);
        for (auto * c : srcs) gguf_free(c);
        gguf_free(dst);

        out = "{\"ok\":true,\"dense\":" + std::to_string(dense.size()) +
              ",\"expert_blocks\":" + std::to_string(nBlocks) +
              ",\"n_expert\":" + std::to_string(n_expert) + "}";
    } catch (const std::exception & e) {
        out = "{\"ok\":false,\"error\":\"" + std::string(e.what()) + "\"}";
    }
}

static void cmd_convert(const std::string & json, std::string & out) {
    try {
        const std::string format = find_str(json, "format");
        const std::string in     = find_str(json, "in");
        const std::string outp   = find_str(json, "out");
        if (format == "v2") { cmd_convert_v2(json, out); return; }
        if (format != "v1") { out = "{\"ok\":false,\"error\":\"only --format v1/v2\"}"; return; }
        std::vector<std::string> shards = split(in, ';');
        if (shards.empty() || outp.empty()) { out = "{\"ok\":false,\"error\":\"need in;out\"}"; return; }

        // read sources (metadata only)
        std::vector<gguf_context *> srcs;
        for (auto & p : shards) {
            gguf_init_params prm = { true, nullptr };
            gguf_context * c = gguf_init_from_file(p.c_str(), prm);
            if (!c) { for (auto * x : srcs) gguf_free(x); out = "{\"ok\":false,\"error\":\"cannot open " + p + "\"}"; return; }
            srcs.push_back(c);
        }

        // merged tensors in source order, classified dense/expert
        struct st { std::string name; int64_t ne[4]; ggml_type type; size_t shard; int idx; bool expert; };
        std::vector<st> all;
        for (size_t si = 0; si < srcs.size(); si++) {
            const int nt = gguf_get_n_tensors(srcs[si]);
            for (int i = 0; i < nt; i++) {
                st t;
                t.name = gguf_get_tensor_name(srcs[si], i);
                const int64_t * ne = gguf_get_tensor_ne(srcs[si], i);
                for (int d = 0; d < 4; d++) t.ne[d] = ne[d];
                t.type   = gguf_get_tensor_type(srcs[si], i);
                t.shard  = si;
                t.idx    = i;
                t.expert = t.name.find("_exps") != std::string::npos;
                all.push_back(t);
            }
        }

        // target gguf
        gguf_context * dst = gguf_init_empty();
        gguf_set_alignment(dst, ALIGN);
        for (int i = 0; i < gguf_get_n_kv(srcs[0]); i++) {
            const std::string k = gguf_get_key(srcs[0], i);
            if (k.rfind("split.", 0) == 0) continue;
            if (k == "general.alignment") continue;
            set_kv_from_src(dst, srcs[0], i);
        }
        gguf_set_val_u32(dst, "general.alignment", (uint32_t) ALIGN);
        gguf_set_val_str(dst, "stream_moe.layout", "sections-v1");

        // add tensors via ggml_tensor (dense first, then expert)
        ggml_init_params gprm = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(gprm);
        if (!gctx) throw std::runtime_error("ggml_init failed");
        auto add_one = [&](const st & t) {
            ggml_tensor * gt = ggml_new_tensor(gctx, t.type, tensor_ndims(t.ne), t.ne);
            ggml_set_name(gt, t.name.c_str());
            gguf_add_tensor(dst, gt);
        };
        size_t nD = 0, nE = 0;
        for (auto & t : all) if (!t.expert) { add_one(t); nD++; }
        for (auto & t : all) if ( t.expert) { add_one(t); nE++; }

        // dense/expert section offsets (absolute file positions)
        const size_t nd_total = all.size();
        const size_t dense_start = nD ? gguf_get_tensor_offset(dst, 0) : 0;
        const size_t dense_end   = nD ? gguf_get_tensor_offset(dst, nD - 1) + gguf_get_tensor_size(dst, nD - 1) : 0;
        const size_t expert_start = nE ? gguf_get_tensor_offset(dst, nD) : 0;
        const size_t expert_end   = nE ? gguf_get_tensor_offset(dst, nD + nE - 1) + gguf_get_tensor_size(dst, nD + nE - 1) : 0;
        uint64_t ds[2] = { dense_start,  dense_end   - dense_start  };
        uint64_t es[2] = { expert_start, expert_end  - expert_start };
        gguf_set_arr_data(dst, "stream_moe.dense_section",  GGUF_TYPE_UINT64, ds, 2);
        gguf_set_arr_data(dst, "stream_moe.expert_section", GGUF_TYPE_UINT64, es, 2);

        // write metadata only
        if (!gguf_write_to_file(dst, outp.c_str(), /*only_meta=*/ true)) throw std::runtime_error("gguf_write_to_file failed");
        ggml_free(gctx);

        // data start = align_up(metadata file size, 4096)
        FILE * outF = std::fopen(outp.c_str(), "r+b");
        if (!outF) throw std::runtime_error("cannot reopen output for data");
        _fseeki64(outF, 0, SEEK_END);
        const size_t metaEnd = (size_t) _ftelli64(outF);
        const size_t dataStart = (metaEnd + ALIGN - 1) & ~(ALIGN - 1);

        // stream tensor data in ADD order (dense first, then expert) so dst offset[idx] matches
        std::vector<st> ordered;
        ordered.reserve(all.size());
        for (auto & t : all) if (!t.expert) ordered.push_back(t);
        for (auto & t : all) if ( t.expert) ordered.push_back(t);
        std::vector<FILE *> srcF;
        for (auto & p : shards) {
            FILE * f = std::fopen(p.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open source " + p);
            srcF.push_back(f);
        }
        std::vector<char> buf(64 * 1024 * 1024);
        int idx = 0;
        for (auto & t : ordered) {
            const size_t srcPos = gguf_get_data_offset(srcs[t.shard]) + gguf_get_tensor_offset(srcs[t.shard], t.idx);
            const size_t size   = gguf_get_tensor_size(srcs[t.shard], t.idx);
            const size_t dstPos = dataStart + gguf_get_tensor_offset(dst, idx);
            FILE * sf = srcF[t.shard];
            if (_fseeki64(sf, srcPos, SEEK_SET) != 0) throw std::runtime_error("seek src failed");
            if (_fseeki64(outF, dstPos, SEEK_SET) != 0) throw std::runtime_error("seek dst failed");
            size_t left = size;
            while (left) {
                const size_t n = left < buf.size() ? left : buf.size();
                if (std::fread(buf.data(), 1, n, sf) != n) throw std::runtime_error("read tensor data failed");
                if (std::fwrite(buf.data(), 1, n, outF) != n) throw std::runtime_error("write tensor data failed");
                left -= n;
            }
            idx++;
        }
        std::fclose(outF);
        for (auto * f : srcF) std::fclose(f);
        for (auto * c : srcs) gguf_free(c);
        gguf_free(dst);

        out = "{\"ok\":true,\"tensors\":" + std::to_string(nd_total) +
              ",\"dense\":" + std::to_string(nD) +
              ",\"expert\":" + std::to_string(nE) + "}";
    } catch (const std::exception & e) {
        out = "{\"ok\":false,\"error\":\"" + std::string(e.what()) + "\"}";
    }
}

static void cmd_chunk(const std::string & json, std::string & out) {
    out = "{\"ok\":false,\"error\":\"chunk not implemented yet\"}";
    (void) json;
}

int main() {
    std::string line, buf;
    while (true) {
        buf.clear();
        int c;
        while ((c = std::fgetc(stdin)) != EOF && c != '\n') buf += (char) c;
        if (c == EOF && buf.empty()) break;
        std::string out;
        const std::string cmd = find_str(buf, "cmd");
        if (cmd == "open") cmd_open(find_str(buf, "path"), out);
        else if (cmd == "convert") cmd_convert(buf, out);
        else if (cmd == "chunk") cmd_chunk(buf, out);
        else out = "{\"ok\":false,\"error\":\"unknown cmd\"}";
        std::fprintf(stdout, "%s\n", out.c_str());
        std::fflush(stdout);
    }
    return 0;
}

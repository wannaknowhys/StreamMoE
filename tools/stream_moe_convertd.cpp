// stream_moe_convertd.cpp - dumb physical GGUF service.
//
// Layout logic lives entirely on the JS side (tools/stream_moe_layout.js);
// this binary only reads GGUF metadata, writes GGUF headers and moves bytes.
// JSON-lines protocol over a raw TCP connection (127.0.0.1, one command per
// line, one response per line, single client).
//
// Start: convertd <port=0>   (0 = ephemeral; actual port printed to stderr as "PORT n")
//
// Commands:
//   open       {"cmd":"open","in":["p1","p2",...]}
//              -> {"ok":true,"files":[{"path","meta":{kv,tensors,data_offset,...}},...]}
//   write_meta {"cmd":"write_meta","out":O,"in":["p1",...],
//                  "skip_kv":["split.","stream_moe."],           // prefixes
//                  "set_kv":{"stream_moe.layout":"expert-blocks-v2",...},
//                  "tensors":[{"name","ne":[...],"type":int},...],
//                  "alignment":4096}
//              -> {"ok":true,"dataOffset":N,"offsets":[tensor offsets],...}
//   copy       {"cmd":"copy","src":["p1",...],"dst":O,"ops":[[srcIdx,srcOff,len,dstOff],...]}
//              -> {"ok":true,"bytes":N}   (dstOff relative to dataOffset)
//   fill       {"cmd":"fill","dst":O,"ops":[[dstOff,len],...]}
//              -> {"ok":true,"bytes":N}   (write zero runs)
//   close      {"cmd":"close","dst":O}
//              -> {"ok":true,"size":N}
//
// Build (Windows, clang-cl + vendored ggml):
//   clang-cl /std:c++17 tools/stream_moe_convertd.cpp /EHsc /MT ^
//       "/IF:/Dev/StreamMoE/third_party/llama.cpp/ggml/include" ^
//       "/IF:/Dev/StreamMoE/third_party/llama.cpp/ggml/src" ^
//       F:/Dev/StreamMoE/build/main/llama-build/ggml/src/ggml-base.lib ^
//       F:/Dev/LLVM/lib/libomp.lib ws2_32.lib /Fe:temp/stream_moe_convertd.exe
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

static const size_t ALIGN = 4096;
static size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

// ---------- minimal JSON ----------
struct JV {
    enum T { NUL, STR, NUM, BOOL, ARR, OBJ } t = NUL;
    std::string str;
    double num = 0;
    bool b = false;
    std::vector<JV> arr;
    std::vector<std::pair<std::string, JV>> obj;
};

static void skip_ws(const char *& p) { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++; }

static std::string parse_jstr(const char *& p) {
    if (*p != '"') return "";
    p++;
    std::string out;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                default: out += *p; break;
            }
        } else out += *p;
        p++;
    }
    if (*p == '"') p++;
    return out;
}

static JV parse_value(const char *& p);

static JV parse_value(const char *& p) {
    skip_ws(p);
    JV v;
    if (*p == '"') { v.t = JV::STR; v.str = parse_jstr(p); }
    else if (*p == '[') {
        v.t = JV::ARR; p++;
        skip_ws(p);
        if (*p == ']') { p++; return v; }
        while (true) { v.arr.push_back(parse_value(p)); skip_ws(p); if (*p == ',') { p++; continue; } break; }
        if (*p == ']') p++;
    } else if (*p == '{') {
        v.t = JV::OBJ; p++;
        skip_ws(p);
        if (*p == '}') { p++; return v; }
        while (true) {
            skip_ws(p);
            std::string k = parse_jstr(p);
            skip_ws(p);
            if (*p == ':') p++;
            v.obj.push_back({ k, parse_value(p) });
            skip_ws(p);
            if (*p == ',') { p++; continue; }
            break;
        }
        if (*p == '}') p++;
    } else if (*p == 't') { v.t = JV::BOOL; v.b = true; p += 4; }
    else if (*p == 'f') { v.t = JV::BOOL; v.b = false; p += 5; }
    else if (*p == 'n') { v.t = JV::NUL; p += 4; }
    else { v.t = JV::NUM; char * end = nullptr; v.num = std::strtod(p, &end); p = end; }
    return v;
}

static JV parse_json(const std::string & s) { const char * p = s.c_str(); return parse_value(p); }

static const JV * obj_get(const JV & o, const char * k) {
    for (auto & kv : o.obj) if (kv.first == k) return &kv.second;
    return nullptr;
}
static std::string obj_str(const JV & o, const char * k, const std::string & d = "") {
    const JV * v = obj_get(o, k); return (v && v->t == JV::STR) ? v->str : d;
}
static double obj_num(const JV & o, const char * k, double d = 0) {
    const JV * v = obj_get(o, k); return (v && v->t == JV::NUM) ? v->num : d;
}
static uint64_t obj_u64(const JV & o, const char * k, uint64_t d = 0) { return (uint64_t) obj_num(o, k, (double) d); }
static const JV * obj_arr(const JV & o, const char * k) {
    const JV * v = obj_get(o, k); return (v && v->t == JV::ARR) ? v : nullptr;
}

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

// ---------- gguf metadata -> JSON ----------
static void emit_meta(const gguf_context * ctx, std::string & out) {
    out += "\"n_kv\":";
    out += std::to_string(gguf_get_n_kv(ctx));
    out += ",\"alignment\":";
    out += std::to_string(gguf_get_alignment(ctx));
    out += ",\"n_tensors\":";
    out += std::to_string(gguf_get_n_tensors(ctx));
    out += ",\"data_offset\":";
    out += std::to_string(gguf_get_data_offset(ctx));
    out += ",\"kv\":[";
    for (int i = 0; i < gguf_get_n_kv(ctx); ++i) {
        if (i) out += ",";
        out += "{";
        out += "\"k\":" + jstr(gguf_get_key(ctx, i));
        const int t = gguf_get_kv_type(ctx, i);
        out += ",\"t\":" + std::to_string(t);
        switch (t) {
            case GGUF_TYPE_UINT8:  out += ",\"v\":" + std::to_string(gguf_get_val_u8 (ctx, i)); break;
            case GGUF_TYPE_INT8:   out += ",\"v\":" + std::to_string(gguf_get_val_i8 (ctx, i)); break;
            case GGUF_TYPE_UINT16: out += ",\"v\":" + std::to_string(gguf_get_val_u16(ctx, i)); break;
            case GGUF_TYPE_INT16:  out += ",\"v\":" + std::to_string(gguf_get_val_i16(ctx, i)); break;
            case GGUF_TYPE_UINT32: out += ",\"v\":" + std::to_string(gguf_get_val_u32(ctx, i)); break;
            case GGUF_TYPE_INT32:  out += ",\"v\":" + std::to_string(gguf_get_val_i32(ctx, i)); break;
            case GGUF_TYPE_UINT64: out += ",\"v\":" + std::to_string(gguf_get_val_u64(ctx, i)); break;
            case GGUF_TYPE_INT64:  out += ",\"v\":" + std::to_string(gguf_get_val_i64(ctx, i)); break;
            case GGUF_TYPE_FLOAT32:out += ",\"v\":" + std::to_string(gguf_get_val_f32(ctx, i)); break;
            case GGUF_TYPE_BOOL:   out += ",\"v\":" + std::to_string(gguf_get_val_bool(ctx, i)); break;
            case GGUF_TYPE_STRING: out += ",\"v\":" + jstr(gguf_get_val_str(ctx, i)); break;
            case GGUF_TYPE_ARRAY: {
                const int at = gguf_get_arr_type(ctx, i);
                const uint64_t n = gguf_get_arr_n(ctx, i);
                out += ",\"arr\":[";
                for (uint64_t j = 0; j < n; j++) {
                    if (j) out += ",";
                    switch (at) {
                        case GGUF_TYPE_UINT8:  out += std::to_string(((const uint8_t *) gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_INT8:   out += std::to_string(((const int8_t *)  gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_UINT16: out += std::to_string(((const uint16_t*)gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_INT16:  out += std::to_string(((const int16_t*) gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_UINT32: out += std::to_string(((const uint32_t*)gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_INT32:  out += std::to_string(((const int32_t*) gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_UINT64: out += std::to_string(((const uint64_t*)gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_INT64:  out += std::to_string(((const int64_t*) gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_FLOAT32:out += std::to_string(((const float*)    gguf_get_arr_data(ctx, i))[j]); break;
                        case GGUF_TYPE_STRING: out += jstr(gguf_get_arr_str(ctx, i, j)); break;
                        default: out += "0"; break;
                    }
                }
                out += "]";
                break;
            }
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
        const int64_t * tne = gguf_get_tensor_ne(ctx, i);
        out += ",\"ne\":[" + std::to_string(tne[0]) + "," + std::to_string(tne[1]) + "," + std::to_string(tne[2]) + "," + std::to_string(tne[3]) + "]";
        out += "}";
    }
    out += "]";
}

// ---------- KV copy from a source gguf ----------
static void set_kv_from_src(gguf_context * dst, const gguf_context * src, int64_t kid) {
    const char * key = gguf_get_key(src, kid);
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
                    std::vector<const char *> cc(n);
                    for (uint64_t j = 0; j < n; j++) cc[j] = gguf_get_arr_str(src, kid, j);
                    gguf_set_arr_str(dst, key, cc.data(), (size_t) n);
                    break;
                }
                default: break;
            }
            break;
        }
        default: break;
    }
}

// ---------- open ----------
static void cmd_open(const JV & j, std::string & out) {
    std::string files = "[";
    int n = 0;
    const JV * ins = obj_arr(j, "in");
    if (ins) for (auto & v : ins->arr) {
        if (v.t != JV::STR) continue;
        const std::string & p = v.str;
        gguf_init_params prm = { true, nullptr };
        gguf_context * c = gguf_init_from_file(p.c_str(), prm);
        if (!c) { out = "{\"ok\":false,\"error\":\"open failed: " + p + "\"}"; return; }
        std::string m;
        emit_meta(c, m);
        gguf_free(c);
        if (n) files += ",";
        files += "{\"path\":" + jstr(p) + ",\"meta\":{" + m + "}}";
        n++;
    }
    files += "]";
    out = "{\"ok\":true,\"files\":" + files + "}";
}

// ---------- open outputs (persistent across commands) ----------
static std::map<std::string, FILE *> g_dst;
static std::map<std::string, uint64_t> g_dstData;

// ---------- write_meta ----------
static void cmd_write_meta(const JV & j, std::string & out) {
    try {
        const std::string outp = obj_str(j, "out");
        if (outp.empty()) throw std::runtime_error("write_meta: need out");
        std::vector<std::string> ins, skip;
        if (const JV * a = obj_arr(j, "in")) for (auto & v : a->arr) if (v.t == JV::STR) ins.push_back(v.str);
        if (const JV * a = obj_arr(j, "skip_kv")) for (auto & v : a->arr) if (v.t == JV::STR) skip.push_back(v.str);
        const uint64_t alignment = obj_u64(j, "alignment", ALIGN);
        if (ins.empty()) throw std::runtime_error("write_meta: need in[]");

        gguf_context * src = gguf_init_from_file(ins[0].c_str(), gguf_init_params{ true, nullptr });
        if (!src) throw std::runtime_error("write_meta: cannot open source");
        gguf_context * dst = gguf_init_empty();
        gguf_set_alignment(dst, alignment);
        auto skip_match = [&](const std::string & k) {
            for (auto & s : skip) if (k.compare(0, s.size(), s) == 0) return true;
            return false;
        };
        for (int i = 0; i < gguf_get_n_kv(src); i++) {
            const std::string k = gguf_get_key(src, i);
            if (skip_match(k)) continue;
            set_kv_from_src(dst, src, i);
        }
        gguf_free(src);

        if (const JV * sk = obj_get(j, "set_kv")) if (sk->t == JV::OBJ) {
            for (auto & kv : sk->obj) {
                const std::string & k = kv.first;
                const JV & v = kv.second;
                if (v.t == JV::STR) gguf_set_val_str(dst, k.c_str(), v.str.c_str());
                else if (v.t == JV::BOOL) gguf_set_val_bool(dst, k.c_str(), v.b);
                else if (v.t == JV::NUM) gguf_set_val_u32(dst, k.c_str(), (uint32_t) v.num);
                else if (v.t == JV::ARR && !v.arr.empty()) {
                    if (v.arr[0].t == JV::STR) {
                        std::vector<const char *> cc;
                        for (auto & e : v.arr) cc.push_back(e.str.c_str());
                        gguf_set_arr_str(dst, k.c_str(), cc.data(), cc.size());
                    } else {
                        std::vector<uint64_t> dd;
                        for (auto & e : v.arr) dd.push_back((uint64_t) e.num);
                        gguf_set_arr_data(dst, k.c_str(), GGUF_TYPE_UINT64, dd.data(), dd.size());
                    }
                }
            }
        }

        ggml_init_params gprm = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(gprm);
        if (!gctx) throw std::runtime_error("write_meta: ggml_init");
        if (const JV * tarr = obj_arr(j, "tensors")) {
            for (auto & t : tarr->arr) {
                const std::string name = obj_str(t, "name");
                const int64_t ty = (int64_t) obj_num(t, "type", 0);
                int64_t ne[4] = { 1, 1, 1, 1 };
                int nd = 0;
                if (const JV * n = obj_arr(t, "ne")) {
                    for (auto & e : n->arr) if (e.t == JV::NUM && nd < 4) ne[nd++] = (int64_t) e.num;
                }
                ggml_tensor * gt = ggml_new_tensor(gctx, (ggml_type) ty, nd, ne);
                if (!gt) throw std::runtime_error("write_meta: ggml_new_tensor failed");
                ggml_set_name(gt, name.c_str());
                gguf_add_tensor(dst, gt);
            }
        }
        ggml_free(gctx);

        if (!gguf_write_to_file(dst, outp.c_str(), /*only_meta=*/ true)) throw std::runtime_error("write_meta: gguf_write_to_file");
        FILE * f = std::fopen(outp.c_str(), "r+b");
        if (!f) throw std::runtime_error("write_meta: reopen output");
        _fseeki64(f, 0, SEEK_END);
        const uint64_t dataOffset = align_up((uint64_t) _ftelli64(f), alignment);
        g_dst[outp] = f;
        g_dstData[outp] = dataOffset;

        const int nt = gguf_get_n_tensors(dst);
        std::string offs = "[";
        for (int i = 0; i < nt; i++) {
            if (i) offs += ",";
            offs += std::to_string(gguf_get_tensor_offset(dst, i));
        }
        offs += "]";
        gguf_free(dst);
        out = "{\"ok\":true,\"dataOffset\":" + std::to_string(dataOffset) +
              ",\"offsets\":" + offs + ",\"n_tensors\":" + std::to_string(nt) + "}";
    } catch (const std::exception & e) {
        out = "{\"ok\":false,\"error\":\"" + std::string(e.what()) + "\"}";
    }
}

// ---------- copy ----------
static void cmd_copy(const JV & j, std::string & out) {
    try {
        std::vector<std::string> srcs;
        if (const JV * a = obj_arr(j, "src")) for (auto & v : a->arr) if (v.t == JV::STR) srcs.push_back(v.str);
        const std::string dst = obj_str(j, "dst");
        const JV * ops = obj_arr(j, "ops");
        if (srcs.empty() || dst.empty() || !ops) throw std::runtime_error("copy: need src[] + dst + ops");
        auto it = g_dst.find(dst);
        if (it == g_dst.end()) throw std::runtime_error("copy: dst not open (write_meta first)");
        FILE * of = it->second;
        const uint64_t dataOff = g_dstData[dst];
        std::vector<FILE *> sf;
        for (auto & p : srcs) {
            FILE * f = std::fopen(p.c_str(), "rb");
            if (!f) { for (auto * x : sf) std::fclose(x); throw std::runtime_error("copy: open src " + p); }
            sf.push_back(f);
        }
        std::vector<char> buf(16 * 1024 * 1024);
        uint64_t total = 0;
        for (auto & op : ops->arr) {
            if (op.t != JV::ARR || op.arr.size() < 4) continue;
            const int si = (int) op.arr[0].num;
            const uint64_t so  = (uint64_t) op.arr[1].num;
            const uint64_t len = (uint64_t) op.arr[2].num;
            const uint64_t d   = (uint64_t) op.arr[3].num;
            if (si < 0 || si >= (int) sf.size()) throw std::runtime_error("copy: bad src idx");
            if (_fseeki64(sf[si], (long long) so, SEEK_SET) != 0) throw std::runtime_error("copy: seek src");
            if (_fseeki64(of, (long long) (dataOff + d), SEEK_SET) != 0) throw std::runtime_error("copy: seek dst");
            uint64_t left = len;
            while (left) {
                const size_t n = left < buf.size() ? (size_t) left : buf.size();
                if (std::fread(buf.data(), 1, n, sf[si]) != n) {
                    throw std::runtime_error("copy: read op srcIdx=" + std::to_string(si) + " srcOff=" + std::to_string(so) + " len=" + std::to_string(len) + " at=" + std::to_string(len - left) + " (file size " + std::to_string(_ftelli64(sf[si])) + "?)");
                }
                if (std::fwrite(buf.data(), 1, n, of) != n) throw std::runtime_error("copy: write");
                left -= n;
                total += n;
            }
        }
        for (auto * x : sf) std::fclose(x);
        out = "{\"ok\":true,\"bytes\":" + std::to_string(total) + "}";
    } catch (const std::exception & e) {
        out = "{\"ok\":false,\"error\":\"" + std::string(e.what()) + "\"}";
    }
}

// ---------- fill ----------
static void cmd_fill(const JV & j, std::string & out) {
    try {
        const std::string dst = obj_str(j, "dst");
        const JV * ops = obj_arr(j, "ops");
        if (dst.empty() || !ops) throw std::runtime_error("fill: need dst + ops");
        auto it = g_dst.find(dst);
        if (it == g_dst.end()) throw std::runtime_error("fill: dst not open");
        FILE * of = it->second;
        const uint64_t dataOff = g_dstData[dst];
        static const char zeros[ALIGN] = { 0 };
        uint64_t total = 0;
        for (auto & op : ops->arr) {
            if (op.t != JV::ARR || op.arr.size() < 2) continue;
            const uint64_t d   = (uint64_t) op.arr[0].num;
            const uint64_t len = (uint64_t) op.arr[1].num;
            if (_fseeki64(of, (long long) (dataOff + d), SEEK_SET) != 0) throw std::runtime_error("fill: seek");
            uint64_t left = len;
            while (left) {
                const size_t n = left < sizeof(zeros) ? (size_t) left : sizeof(zeros);
                if (std::fwrite(zeros, 1, n, of) != n) throw std::runtime_error("fill: write");
                left -= n;
                total += n;
            }
        }
        out = "{\"ok\":true,\"bytes\":" + std::to_string(total) + "}";
    } catch (const std::exception & e) {
        out = "{\"ok\":false,\"error\":\"" + std::string(e.what()) + "\"}";
    }
}

// ---------- close ----------
static void cmd_close(const JV & j, std::string & out) {
    const std::string dst = obj_str(j, "dst");
    auto it = g_dst.find(dst);
    if (it == g_dst.end()) { out = "{\"ok\":false,\"error\":\"close: not open\"}"; return; }
    _fseeki64(it->second, 0, SEEK_END);
    const uint64_t size = (uint64_t) _ftelli64(it->second);
    std::fclose(it->second);
    g_dst.erase(it);
    g_dstData.erase(dst);
    out = "{\"ok\":true,\"size\":" + std::to_string(size) + "}";
}

// ---------- main (TCP server, JSON-lines) ----------
static void dispatch(const std::string & line, std::string & out) {
    try {
        const JV j = parse_json(line);
        const std::string cmd = obj_str(j, "cmd");
        if (cmd == "open") cmd_open(j, out);
        else if (cmd == "write_meta") cmd_write_meta(j, out);
        else if (cmd == "copy") cmd_copy(j, out);
        else if (cmd == "fill") cmd_fill(j, out);
        else if (cmd == "close") cmd_close(j, out);
        else out = "{\"ok\":false,\"error\":\"unknown cmd\"}";
    } catch (const std::exception & e) {
        out = "{\"ok\":false,\"error\":\"" + std::string(e.what()) + "\"}";
    }
}

static void net_send(SOCKET c, const std::string & s) {
    size_t off = 0;
    while (off < s.size()) {
        const int n = send(c, s.data() + off, (int) (s.size() - off), 0);
        if (n <= 0) return;
        off += (size_t) n;
    }
}

int main(int argc, char ** argv) {
    int port = argc > 1 ? atoi(argv[1]) : 0;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { std::fprintf(stderr, "WSAStartup failed\n"); return 1; }
    SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) { std::fprintf(stderr, "socket failed\n"); return 1; }
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *) &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((u_short) port);
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) != 0) { std::fprintf(stderr, "bind failed (port %d)\n", port); return 1; }
    if (port == 0) {
        sockaddr_in got{};
        int glen = sizeof(got);
        if (getsockname(srv, (sockaddr *) &got, &glen) == 0) port = ntohs(got.sin_port);
    }
    std::fprintf(stderr, "PORT %d\n", port);
    std::fflush(stderr);
    if (listen(srv, 1) != 0) { std::fprintf(stderr, "listen failed\n"); return 1; }
    SOCKET c = accept(srv, nullptr, nullptr);
    if (c == INVALID_SOCKET) { std::fprintf(stderr, "accept failed\n"); return 1; }

    std::string buf;
    char tmp[65536];
    for (;;) {
        const int n = recv(c, tmp, sizeof(tmp), 0);
        if (n <= 0) break; // client closed
        buf.append(tmp, (size_t) n);
        size_t idx;
        while ((idx = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, idx);
            buf.erase(0, idx + 1);
            if (line.empty()) continue;
            std::string out;
            dispatch(line, out);
            net_send(c, out + "\n");
        }
    }
    closesocket(c);
    closesocket(srv);
    WSACleanup();
    return 0;
}

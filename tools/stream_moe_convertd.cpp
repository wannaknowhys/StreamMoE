// stream_moe_convertd.cpp - C/C++ wrapper around the ggml gguf library.
// JSON-lines protocol over stdin/stdout. Spawned by tools/stream_moe_convert.js.
// Commands (one JSON per line on stdin):
//   {"cmd":"open","path":"..."}       -> {"ok":true,"meta":{...}}  (GGUF metadata summary)
//   {"cmd":"convert","format":"v1|v2","in":["..."],"out":"...","plan":{...}}  -> progress/ok
//   {"cmd":"chunk",...}               -> v2 RAID0 chunking (TODO)
// Build (Windows, clang-cl + vendored ggml):
//   clang-cl /std:c++17 tools/stream_moe_convertd.cpp /EHsc -Ithird_party/llama.cpp/ggml/include ^
//       build/main/llama-build/ggml/src/ggml-base.lib /Fe:temp/stream_moe_convertd.exe
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

// ---------- minimal JSON (read: flat object of string/num; write: we build manually) ----------
static std::string jstr(const std::string & s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out + "\"";
}

// find a top-level "key":"value" pair in a flat JSON object (strings only)
static std::string find_str(const std::string & json, const std::string & key) {
    const std::string pat = "\"" + key + "\":\"";
    size_t p = json.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    std::string v;
    while (p < json.size() && json[p] != '"') {
        if (json[p] == '\\') {
            if (p + 1 < json.size() && json[p + 1] == '\\') { v += '\\'; p += 2; }
            else { v += json[p]; p++; } // keep other backslashes literal (paths etc.)
        }
        else { v += json[p]; p++; }
    }
    return v;
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
    gguf_init_params params = { true, nullptr }; // no_alloc: metadata only, skip tensor data
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

static void cmd_convert(const std::string & json, std::string & out) {
    // TODO: full v1/v2 writer using gguf_set_val_* / gguf_add_tensor / gguf_write_to_file.
    out = "{\"ok\":false,\"error\":\"convert not implemented yet\"}";
    (void) json;
}

static void cmd_chunk(const std::string & json, std::string & out) {
    // TODO: v2 RAID0 chunking.
    out = "{\"ok\":false,\"error\":\"chunk not implemented yet\"}";
    (void) json;
}

int main() {
    std::string line;
    std::string buf;
    while (true) {
        // read a line (JSON-lines protocol)
        buf.clear();
        int c;
        while ((c = std::fgetc(stdin)) != EOF && c != '\n') buf += (char) c;
        if (c == EOF && buf.empty()) break;

        std::string out;
        const std::string cmd = find_str(buf, "cmd");
        if (cmd == "open") {
            cmd_open(find_str(buf, "path"), out);
        } else if (cmd == "convert") {
            cmd_convert(buf, out);
        } else if (cmd == "chunk") {
            cmd_chunk(buf, out);
        } else {
            out = "{\"ok\":false,\"error\":\"unknown cmd\"}";
        }
        std::fprintf(stdout, "%s\n", out.c_str());
        std::fflush(stdout);
    }
    return 0;
}

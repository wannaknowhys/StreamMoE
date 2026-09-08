// stream_moe_convert - C++ converter CLI (replaces tools/stream_moe_convert.js).
//
// Any source -> v2 / v3 / v3chunk:
//   original multi-shard / original / v2 / v2chunk / v3 / v3chunk
// all parse into model_t (src/loader/model_builder.cpp); the writer
// (src/convert/writer.cpp) emits the target.
//
// usage:
//   stream_moe_convert -m <model.gguf> -o <out.gguf> [--format v2|v3]
//   stream_moe_convert -m <model.gguf> -o <base> --format v3chunk --chunks 5 [--ratio 8:9:9:7:9]
//     (v3chunk writes <base>-00001.gguf, <base>-00002.gguf, ...; digit width
//      mirrors the source filename's trailing number, else 5, and grows)
//   stream_moe_convert -m "c1.gguf;c2.gguf;..." -o <out.gguf> [--format v2|v3]

#include "convert/writer.h"
#include "loader/model_builder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stream_moe;

static std::vector<std::string> split_semicolon(const std::string& s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find(';', p);
        const std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!tok.empty()) out.push_back(tok);
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

static std::vector<int> parse_ratio(const std::string& s) {
    std::vector<int> out;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find(':', p);
        const std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    convert_opts_t opts;
    std::string model, out;
    std::string format = "v3";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        try {
            if (a == "-m" || a == "--model") model = next();
            else if (a == "-o" || a == "--output") out = next();
            else if (a == "--format") format = next();
            else if (a == "--chunks") opts.chunks = std::atoi(next().c_str());
            else if (a == "--ratio") opts.ratio = parse_ratio(next());
            else throw std::runtime_error("unknown arg: " + a);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[convert] %s\n", e.what());
            return 2;
        }
    }
    if (model.empty() || out.empty()) {
        std::fprintf(stderr,
            "usage: stream_moe_convert -m <model> -o <out> [--format v2|v3|v3chunk] [--chunks N] [--ratio a:b:c]\n");
        return 2;
    }
    if (format == "v2") opts.target = convert_opts_t::target_t::V2;
    else if (format == "v3") opts.target = convert_opts_t::target_t::V3;
    else if (format == "v3chunk") opts.target = convert_opts_t::target_t::V3_CHUNK;
    else { std::fprintf(stderr, "[convert] --format must be v2|v3|v3chunk\n"); return 2; }

    try {
        const std::vector<std::string> paths = split_semicolon(model);
        model_t m = paths.size() > 1 ? parse_model(paths) : parse_model_path(model);
        std::printf("[parse] source layout=%d shards=%zu layers=%u experts=%u dense=%zu expertTensors=%zu\n",
                    static_cast<int>(m.layout), m.files.size(), m.n_layer, m.n_expert,
                    m.dense.size(), m.expert.size());
        convert_model(m, opts, out);
        std::printf("[%s] done -> %s\n", format.c_str(), out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[convert] %s\n", e.what());
        return 1;
    }
    return 0;
}

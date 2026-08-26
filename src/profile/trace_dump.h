#pragma once

// Temporary per-layer tensor trace for baseline-vs-expert-backend comparison.
// Guarded by STREAM_MOE_TEMP (see docs/PROJECT_STRUCTURE.md §10). Short-term
// diagnostic only; not part of the normal build.

// Hooks llama_context_params.cb_eval: the sched fires cb_eval(node, ask=false)
// AFTER the batch containing the node has been computed and synchronized, so
// node->data holds the computed output. We dump:
//   - ffn_norm / ffn_out / attn_* outputs (per-layer X values)
//   - MUL_MAT_ID nodes' src[2] (selected expert ids = routing result)
// Record format (binary, append-only):
//   u32 magic("SMT1") | u32 name_len | name | i32 ne[4] | u32 elem | u64 nbytes | nbytes data

#ifdef STREAM_MOE_TEMP
#include "ggml.h"
#include <cstdio>
#include <cstring>

namespace stream_moe {

inline bool trace_dump_cb(struct ggml_tensor* t, bool ask, void* ud) {
    FILE* f = static_cast<FILE*>(ud);
    if (ask) {
        // isolate target nodes so their batch is synchronized before the dump
        return t->op == GGML_OP_MUL_MAT_ID ||
               std::strstr(t->name, "ffn_norm") != nullptr ||
               std::strstr(t->name, "ffn_out")  != nullptr ||
               std::strstr(t->name, "attn")     != nullptr;
    }

    auto dump = [&](const char* tag, struct ggml_tensor* x) {
        if (!x || !x->data) return;
        const uint32_t magic = 0x31544D53; // "SMT1"
        const uint32_t name_len = static_cast<uint32_t>(std::strlen(tag));
        const int32_t ne[4] = { static_cast<int32_t>(x->ne[0]), static_cast<int32_t>(x->ne[1]),
                                static_cast<int32_t>(x->ne[2]), static_cast<int32_t>(x->ne[3]) };
        const uint32_t elem = static_cast<uint32_t>(ggml_element_size(x));
        const uint64_t nbytes = static_cast<uint64_t>(ggml_nbytes(x));
        fwrite(&magic, 4, 1, f);
        fwrite(&name_len, 4, 1, f);
        fwrite(tag, 1, name_len, f);
        fwrite(ne, 4, 4, f);
        fwrite(&elem, 4, 1, f);
        fwrite(&nbytes, 8, 1, f);
        fwrite(x->data, 1, nbytes, f);
    };

    dump(t->name, t);                              // computed node output (X values)
    if (t->op == GGML_OP_MUL_MAT_ID && t->src[2]) {
        dump(t->src[2]->name, t->src[2]);          // selected expert ids (routing)
    }
    std::fflush(f);
    return true;
}

} // namespace stream_moe
#endif // STREAM_MOE_TEMP

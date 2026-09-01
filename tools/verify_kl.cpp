// verify_kl.cpp - per-token KL divergence between two prefill exports
// (PREFEXP2 embd section) using the GGUF model's LM head (output.weight) via ggml.
//
// Usage: verify_kl <model.gguf> <ref.bin> <cand.bin> [--thresh <T>]
//   ref  = file1 = REFERENCE (baseline). Main metric: KL(ref || cand).
//   cand = file2. Side metric (informational): KL(cand || ref).
//   Tokens are the contiguous sequence pos = 0 .. max(ref_last, cand_last).
//   Missing per pos: in ref only -> "file 2 missing"; in cand only ->
//   "file 1 missing"; in neither -> "both file missing".
//
// embd dtype is detected dynamically (PREFEXP2 per-section GGML_TYPE marker,
// 0=f32 / 1=f16); output.weight type comes from the GGUF. ggml_mul_mat is the
// type-matching operator: it dispatches over quantized/f16/f32 weights itself.
//
// Build (link vendored ggml static libs):
//   clang++.exe -std=c++17 -O2 tools\verify_kl.cpp -I third_party\llama.cpp\ggml\include ^
//     build\main\llama-build\ggml\src\ggml.lib build\main\llama-build\ggml\src\ggml-base.lib ^
//     build\main\llama-build\ggml\src\ggml-cpu.lib -o temp\verify_kl.exe
// See tools/verify_kl.md for full usage.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct row_t {
    uint32_t pos;
    std::vector<float> d;
};

static uint32_t rd_u32(FILE * f) {
    uint32_t v = 0;
    if (fread(&v, 4, 1, f) != 1) v = 0;
    return v;
}

// Parse the PREFEXP1/PREFEXP2 embd section (LM head input rows) only.
// Returns rows (pos + embd, always f32), sets dim and dtype (0=f32, 1=f16).
static bool read_embd(const char * path, std::vector<row_t> & out, uint32_t & dim, int & dtype) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[verify_kl] cannot open %s\n", path);
        return false;
    }
    char magic[9] = {};
    if (fread(magic, 1, 8, f) != 8 ||
        (strncmp(magic, "PREFEXP2", 8) != 0 && strncmp(magic, "PREFEXP1", 8) != 0)) {
        fprintf(stderr, "[verify_kl] bad magic in %s: %.8s\n", path, magic);
        fclose(f);
        return false;
    }
    const bool v2 = strncmp(magic, "PREFEXP2", 8) == 0;
    const uint32_t nRows = rd_u32(f);
    const uint32_t nDim  = rd_u32(f);
    int dt = 0;
    if (v2) dt = (int) rd_u32(f); // 0=f32, 1=f16
    dtype = dt;
    dim   = nDim;

    const size_t esz = dt ? 2 : 4;
    std::vector<uint8_t> tmp((size_t) nDim * esz);
    out.reserve(nRows);
    for (uint32_t i = 0; i < nRows; i++) {
        const uint32_t pos = rd_u32(f);
        if (fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) break;
        row_t r;
        r.pos = pos;
        r.d.resize(nDim);
        if (dt) {
            for (uint32_t j = 0; j < nDim; j++) {
                ggml_fp16_t h;
                memcpy(&h, tmp.data() + (size_t) j * 2, 2);
                r.d[j] = ggml_fp16_to_fp32(h);
            }
        } else {
            memcpy(r.d.data(), tmp.data(), (size_t) nDim * 4);
        }
        out.push_back(std::move(r));
    }
    fclose(f);
    return true;
}

static void softmax_row(const float * logits, size_t n, std::vector<double> & p) {
    float m = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double sum = 0.0;
    p.resize(n);
    for (size_t i = 0; i < n; i++) {
        p[i] = std::exp((double) logits[i] - (double) m);
        sum += p[i];
    }
    for (size_t i = 0; i < n; i++) p[i] /= sum;
}

// KL(P || Q) = sum_i P[i] * log(P[i] / Q[i]); P==0 terms contribute 0;
// P>0 with Q==0 -> infinity.
static double kl_div(const std::vector<double> & P, const std::vector<double> & Q) {
    double s = 0.0;
    for (size_t i = 0; i < P.size(); i++) {
        if (P[i] > 0.0) {
            if (Q[i] <= 0.0) return std::numeric_limits<double>::infinity();
            s += P[i] * std::log(P[i] / Q[i]);
        }
    }
    return s;
}

static void usage(const char * p) {
    fprintf(stderr,
        "usage: %s <model.gguf> <ref.bin> <cand.bin> [--thresh <T>]\n"
        "  ref  = file1 = REFERENCE (baseline), main metric KL(ref || cand)\n"
        "  cand = file2 (informational KL(cand || ref) also printed)\n"
        "  --thresh T : mark RESULT FAIL when any token KL(ref||cand) > T\n"
        "  missing: ref-only pos -> 'file 2 missing'; cand-only -> 'file 1 missing';\n"
        "           neither -> 'both file missing'\n",
        p);
}

int main(int argc, char ** argv) {
    const char * model = argv[1];
    const char * refp  = argv[2];
    const char * candp = argv[3];
    double thresh = 0.0;
    bool listTensors = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list-tensors")) { listTensors = true; continue; }
        if (!strcmp(argv[i], "--thresh") && i + 1 < argc) thresh = atof(argv[++i]);
    }

    // ---- open GGUF, locate output.weight ----
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(model, gp);
    if (!g) { fprintf(stderr, "[verify_kl] failed to open GGUF %s\n", model); return 1; }

    if (listTensors) {
        const int64_t nt = gguf_get_n_tensors(g);
        fprintf(stderr, "[verify_kl] %lld tensors:\n", (long long) nt);
        for (int64_t i = 0; i < nt; i++) {
            const int64_t * tn = gguf_get_tensor_ne(g, i);
            const size_t tsz = gguf_get_tensor_size(g, i);
            fprintf(stderr, "  %-40s %-6s ne=[%lld x %lld x %lld x %lld] %zu bytes\n",
                    gguf_get_tensor_name(g, i), ggml_type_name(gguf_get_tensor_type(g, i)),
                    (long long) tn[0], (long long) tn[1], (long long) tn[2], (long long) tn[3], tsz);
        }
        gguf_free(g);
        return 0;
    }

    if (argc < 4) { usage(argv[0]); return 2; }
    int64_t tid = gguf_find_tensor(g, "output.weight");
    const char * head_name = "output.weight";
    if (tid < 0) {
        // tied-embeddings models (e.g. gemma) have no output.weight - the LM head
        // is token_embd.weight itself, already laid out as [n_embd, n_vocab].
        tid = gguf_find_tensor(g, "token_embd.weight");
        head_name = "token_embd.weight (tied)";
        if (tid < 0) {
            const int64_t nt = gguf_get_n_tensors(g);
            for (int64_t i = 0; i < nt; i++) {
                if (strstr(gguf_get_tensor_name(g, i), "output")) { tid = i; break; }
            }
        }
    }
    if (tid < 0) { fprintf(stderr, "[verify_kl] no LM head tensor in %s\n", model); gguf_free(g); return 1; }

    const enum ggml_type wtype = gguf_get_tensor_type(g, tid);
    const int64_t * ne = gguf_get_tensor_ne(g, tid);
    const int64_t n_embd = ne[0], n_vocab = ne[1];
    if (ne[2] != 1 || ne[3] != 1) {
        fprintf(stderr, "[verify_kl] LM head %s is %lld-D; only 2D [n_embd, n_vocab] supported\n",
                head_name, (long long) (ne[2] * ne[3]));
        gguf_free(g);
        return 1;
    }
    fprintf(stderr, "[verify_kl] LM head %s type=%s ne=[%lld x %lld]\n",
            head_name, ggml_type_name(wtype), (long long) n_embd, (long long) n_vocab);

    // read weight bytes (no_alloc: gguf only gives offsets)
    FILE * mf = fopen(model, "rb");
    if (!mf) { fprintf(stderr, "[verify_kl] cannot open %s\n", model); gguf_free(g); return 1; }
    const size_t woff   = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    const size_t wbytes = gguf_get_tensor_size(g, tid);
    std::vector<uint8_t> wbuf(wbytes);
    if (fseek(mf, (long) woff, SEEK_SET) != 0 || fread(wbuf.data(), 1, wbytes, mf) != wbytes) {
        fprintf(stderr, "[verify_kl] failed to read output.weight (%zu bytes @ %zu)\n", wbytes, woff);
        fclose(mf); gguf_free(g); return 1;
    }
    fclose(mf);
    gguf_free(g);

    // ---- exports ----
    std::vector<row_t> A, B;
    uint32_t Adim = 0, Bdim = 0;
    int Adt = 0, Bdt = 0;
    if (!read_embd(refp,  A, Adim, Adt)) return 1;
    if (!read_embd(candp, B, Bdim, Bdt)) return 1;
    fprintf(stderr, "[verify_kl] ref  (%s): rows=%zu dim=%u dtype=%s\n", refp, A.size(), Adim, Adt ? "f16" : "f32");
    fprintf(stderr, "[verify_kl] cand (%s): rows=%zu dim=%u dtype=%s\n", candp, B.size(), Bdim, Bdt ? "f16" : "f32");
    if (Adim != (uint32_t) n_embd || Bdim != (uint32_t) n_embd) {
        fprintf(stderr, "[verify_kl] embd dim mismatch: model=%lld ref=%u cand=%u\n",
                (long long) n_embd, Adim, Bdim);
        return 1;
    }

    std::map<uint32_t, size_t> ai, bi;
    for (size_t i = 0; i < A.size(); i++) ai[A[i].pos] = i;
    for (size_t i = 0; i < B.size(); i++) bi[B[i].pos] = i;
    const uint32_t last = std::max(A.empty() ? 0 : A.back().pos, B.empty() ? 0 : B.back().pos);

    // ---- missing stats (contiguous pos 0..last) + shared pos list ----
    std::vector<uint32_t> shared;
    size_t ref_missing = 0, cand_missing = 0, both_missing = 0;
    for (uint32_t pos = 0; pos <= last; pos++) {
        const bool ha = ai.count(pos) != 0, hb = bi.count(pos) != 0;
        if (ha && hb) {
            shared.push_back(pos);
        } else if (ha) {
            printf("token#%u: file 2 missing\n", pos);
            cand_missing++;
        } else if (hb) {
            printf("token#%u: file 1 missing\n", pos);
            ref_missing++;
        } else {
            printf("token#%u: both file missing\n", pos);
            both_missing++;
        }
    }
    const size_t nTok = shared.size();

    // ---- weights in a persistent context (tied token_embd can be ~800 MiB) ----
    struct ggml_init_params wip = { /*.mem_size =*/ ggml_tensor_overhead() + wbytes + (1u << 20),
                                    /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    struct ggml_context * wctx = ggml_init(wip);
    if (!wctx) { fprintf(stderr, "[verify_kl] ggml_init(weights) failed (need ~%.1f MiB)\n", wbytes / (1024.0 * 1024.0)); return 1; }
    struct ggml_tensor * w = ggml_new_tensor_2d(wctx, wtype, n_embd, n_vocab);
    memcpy(w->data, wbuf.data(), wbytes);

    // ---- batched logits + per-token KL (n_vocab ~262k -> batch to bound memory) ----
    const size_t BATCH = 128;
    std::vector<double> P, Q;
    std::vector<float> xA((size_t) n_embd * BATCH), xB((size_t) n_embd * BATCH);
    double sum_k = 0.0, max_k = 0.0;
    uint32_t max_pos = 0;
    bool fail = false;
    for (size_t tb = 0; tb < nTok; tb += BATCH) {
        const size_t nb = std::min(BATCH, nTok - tb);
        for (size_t t = 0; t < nb; t++) {
            const uint32_t pos = shared[tb + t];
            memcpy(&xA[t * (size_t) n_embd], A[ai[pos]].d.data(), (size_t) n_embd * 4);
            memcpy(&xB[t * (size_t) n_embd], B[bi[pos]].d.data(), (size_t) n_embd * 4);
        }
        const size_t bmem = ggml_tensor_overhead() * 4
                          + (size_t) n_embd * nb * 4 * 2
                          + (size_t) n_vocab * nb * 4 * 2
                          + ggml_graph_overhead()
                          + (1u << 20);
        struct ggml_init_params bip = { /*.mem_size =*/ bmem, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
        struct ggml_context * bctx = ggml_init(bip);
        if (!bctx) { fprintf(stderr, "[verify_kl] batch ggml_init failed\n"); ggml_free(wctx); return 1; }
        struct ggml_tensor * x1 = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, n_embd, nb);
        struct ggml_tensor * x2 = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, n_embd, nb);
        memcpy(x1->data, xA.data(), (size_t) n_embd * nb * 4);
        memcpy(x2->data, xB.data(), (size_t) n_embd * nb * 4);
        struct ggml_tensor * m1 = ggml_mul_mat(bctx, w, x1);
        struct ggml_tensor * m2 = ggml_mul_mat(bctx, w, x2);
        struct ggml_cgraph * gf = ggml_new_graph(bctx);
        ggml_build_forward_expand(gf, m1);
        ggml_build_forward_expand(gf, m2);
        if (ggml_graph_compute_with_ctx(bctx, gf, 8) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[verify_kl] graph compute failed\n");
            ggml_free(bctx); ggml_free(wctx);
            return 1;
        }
        const float * L1 = (const float *) m1->data; // [n_vocab, nb] col-major, elem (v,t) at v + t*n_vocab
        const float * L2 = (const float *) m2->data;
        for (size_t t = 0; t < nb; t++) {
            const uint32_t pos = shared[tb + t];
            softmax_row(L1 + t * (size_t) n_vocab, (size_t) n_vocab, P);
            softmax_row(L2 + t * (size_t) n_vocab, (size_t) n_vocab, Q);
            const double k  = kl_div(P, Q);
            const double kr = kl_div(Q, P);
            sum_k += k;
            if (k > max_k) { max_k = k; max_pos = pos; }
            if (thresh > 0.0 && k > thresh) fail = true;
            printf("token#%u: KL(ref||cand)=%.6e  KL(cand||ref)=%.6e\n", pos, k, kr);
        }
        ggml_free(bctx);
    }
    ggml_free(wctx);

    const size_t n_shared = shared.size();
    printf("\n[verify_kl] tokens=%u shared=%zu ref_missing=%zu cand_missing=%zu both_missing=%zu\n",
           last + 1, n_shared, ref_missing, cand_missing, both_missing);
    if (n_shared > 0) {
        printf("[verify_kl] mean_KL(ref||cand)=%.6e  max_KL(ref||cand)=%.6e @token#%u\n",
               sum_k / n_shared, max_k, max_pos);
    }
    if (thresh > 0.0) {
        printf("[verify_kl] RESULT: %s (thresh=%.3e, max_KL=%.6e)\n",
               fail ? "FAIL" : "PASS", thresh, max_k);
    } else {
        printf("[verify_kl] RESULT: %s (no thresh; max_KL(ref||cand)=%.6e)\n",
               max_k == std::numeric_limits<double>::infinity() ? "INF" : "reported", max_k);
    }

    return fail ? 1 : 0;
}

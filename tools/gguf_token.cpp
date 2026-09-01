// gguf_token.cpp - dump GGUF tokenizer vocab facts: BOS/EOS ids, token text+type.
// usage: gguf_token <model.gguf> [--token N]
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void show_token(struct gguf_context * g, int64_t kt, int64_t kty, int id) {
    const char * txt = gguf_get_arr_str(g, kt, id);
    const uint32_t * ty = kty >= 0 ? (const uint32_t *) gguf_get_arr_data(g, kty) : nullptr;
    uint32_t t = ty ? ty[id] : 0;
    printf("  token %d: '%s' type=%u\n", id, txt, t);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: gguf_token <model.gguf> [--token N]\n"); return 2; }
    const char * model = argv[1];
    int want = -1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--token") && i + 1 < argc) want = atoi(argv[++i]);
    }

    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(model, gp);
    if (!g) { fprintf(stderr, "cannot open GGUF %s\n", model); return 1; }

    const int64_t kt  = gguf_find_key(g, "tokenizer.ggml.tokens");
    const int64_t kty = gguf_find_key(g, "tokenizer.ggml.token_type");
    const int64_t kb  = gguf_find_key(g, "tokenizer.ggml.bos_token_id");
    const int64_t ke  = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
    const int64_t km  = gguf_find_key(g, "tokenizer.ggml.model");

    if (kt < 0) { fprintf(stderr, "no tokenizer.ggml.tokens\n"); gguf_free(g); return 1; }
    const size_t ntok = gguf_get_arr_n(g, kt);

    if (km >= 0) printf("model = %s\n", gguf_get_val_str(g, km));
    if (kb >= 0) { const int id = (int) gguf_get_val_u32(g, kb); printf("BOS id = %d", id); show_token(g, kt, kty, id); }
    if (ke >= 0) { const int id = (int) gguf_get_val_u32(g, ke); printf("EOS id = %d", id); show_token(g, kt, kty, id); }
    printf("n_tokens = %zu\n", ntok);

    if (want >= 0 && (size_t) want < ntok) {
        printf("requested:\n");
        show_token(g, kt, kty, want);
    } else {
        printf("first 6:\n");
        for (int i = 0; i < 6 && (size_t) i < ntok; i++) show_token(g, kt, kty, i);
    }
    gguf_free(g);
    return 0;
}

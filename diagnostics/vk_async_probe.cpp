// vk_async_probe: measure RX590 Vulkan async transfer + event primitives.
// Answers: does get_tensor_async/event_record/wait/synchronize work here,
// sync-get vs async-batch-get timing, transfer/compute overlap, submit latency.
// Only backend-agnostic tensor APIs (no raw data memcpy); exit 0 = all checks pass.
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_pattern(std::vector<uint8_t> & v, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = (uint8_t)(s >> 24);
    }
}

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) { std::printf("NO-DEVICE Vulkan0 not found\n"); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { std::printf("NO-INIT backend init failed\n"); return 2; }

    const size_t N_SMALL = 256;              // ids-like
    const size_t N_CUR   = 16 * 1024;        // decode cur-like
    const size_t N_W     = 100 * 1024;       // weights-like
    const size_t N_BIG   = 3 * 1024 * 1024;  // prefill cur-like
    const size_t N_COMP  = 16 * 1024 * 1024; // compute add (floats)

    ggml_init_params ip = { 128 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_small = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_SMALL / 4);
    ggml_tensor * t_cur   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_CUR / 4);
    ggml_tensor * t_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_W / 4);
    ggml_tensor * t_big   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_BIG / 4);
    ggml_tensor * t_a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_COMP / 4);
    ggml_tensor * t_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_COMP / 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { std::printf("NO-ALLOC device buffer failed\n"); return 2; }

    std::vector<uint8_t> h_small(N_SMALL), h_cur(N_CUR), h_w(N_W), h_big(N_BIG);
    std::vector<uint8_t> h_a(N_COMP), h_b(N_COMP);
    std::vector<uint8_t> stage(N_BIG);
    fill_pattern(h_small, 11); fill_pattern(h_cur, 22); fill_pattern(h_w, 23);
    fill_pattern(h_big, 33); fill_pattern(h_a, 44); fill_pattern(h_b, 55);

    // upload all inputs (sync uploads are fine; downloads are the question)
    ggml_backend_tensor_set(t_small, h_small.data(), 0, N_SMALL);
    ggml_backend_tensor_set(t_cur, h_cur.data(), 0, N_CUR);
    ggml_backend_tensor_set(t_w, h_w.data(), 0, N_W);
    ggml_backend_tensor_set(t_big, h_big.data(), 0, N_BIG);
    ggml_backend_tensor_set(t_a, h_a.data(), 0, N_COMP);
    ggml_backend_tensor_set(t_b, h_b.data(), 0, N_COMP);

    // ---- correctness: async get + event, then verify bytes ----
    ggml_backend_event_t ev = ggml_backend_event_new(dev);
    if (!ev) { std::printf("NO-EVENT event_new failed\n"); return 2; }
    std::vector<uint8_t> got_small(N_SMALL), got_cur(N_CUR), got_big(N_BIG);
    ggml_backend_tensor_get_async(be, t_small, got_small.data(), 0, N_SMALL);
    ggml_backend_tensor_get_async(be, t_cur, got_cur.data(), 0, N_CUR);
    ggml_backend_tensor_get_async(be, t_big, got_big.data(), 0, N_BIG);
    ggml_backend_event_record(ev, be);
    ggml_backend_event_synchronize(ev);
    bool ok = memcmp(got_small.data(), h_small.data(), N_SMALL) == 0 &&
              memcmp(got_cur.data(), h_cur.data(), N_CUR) == 0 &&
              memcmp(got_big.data(), h_big.data(), N_BIG) == 0;
    std::printf("integrity: %s\n", ok ? "OK" : "MISMATCH");
    if (!ok) return 1;

    const int N = 200;
    // ---- A: 3x sync get ----
    double t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get(t_small, stage.data(), 0, N_SMALL);
        ggml_backend_tensor_get(t_cur, stage.data(), 0, N_CUR);
        ggml_backend_tensor_get(t_big, stage.data(), 0, N_BIG);
    }
    double sync3 = (now_ms() - t0) / N;

    // ---- B: 3x async get + 1 sync ----
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, stage.data(), 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur, stage.data(), 0, N_CUR);
        ggml_backend_tensor_get_async(be, t_big, stage.data(), 0, N_BIG);
        ggml_backend_synchronize(be);
    }
    double async3 = (now_ms() - t0) / N;

    // ---- C: async get + event_sync ----
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, stage.data(), 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur, stage.data(), 0, N_CUR);
        ggml_backend_tensor_get_async(be, t_big, stage.data(), 0, N_BIG);
        ggml_backend_event_record(ev, be);
        ggml_backend_event_synchronize(ev);
    }
    double event3 = (now_ms() - t0) / N;

    // ---- D: submit latency (tiny graph, 1 ADD on 1MB) ----
    ggml_tensor * s_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024);
    ggml_tensor * s_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024);
    // NOTE: created after buf alloc -> bufferless; bind manually is complex,
    // so measure with a fresh small ctx+buffer instead.
    ggml_init_params ip2 = { 8 * 1024 * 1024, nullptr, true };
    ggml_context * ctx2 = ggml_init(ip2);
    ggml_tensor * q_a = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 256 * 1024);
    ggml_tensor * q_b = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 256 * 1024);
    ggml_tensor * q_c = ggml_add(ctx2, q_a, q_b);
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, be);
    std::vector<uint8_t> h_q(1024 * 1024, 7);
    ggml_backend_tensor_set(q_a, h_q.data(), 0, h_q.size());
    ggml_backend_tensor_set(q_b, h_q.data(), 0, h_q.size());
    ggml_cgraph * g = ggml_new_graph_custom(ctx2, 4, false);
    ggml_build_forward_expand(g, q_c);
    (void) s_a; (void) s_b; (void) buf2;
    t0 = now_ms();
    for (int i = 0; i < N; ++i) ggml_backend_graph_compute(be, g);
    double submit = (now_ms() - t0) / N;

    // ---- E: overlap (compute async + big get async, one sync) vs serial ----
    t0 = now_ms();
    for (int i = 0; i < 50; ++i) {
        ggml_backend_graph_compute(be, g);
        ggml_backend_tensor_get(t_big, stage.data(), 0, N_BIG);
    }
    double serial = (now_ms() - t0) / 50;
    t0 = now_ms();
    for (int i = 0; i < 50; ++i) {
        ggml_backend_graph_compute_async(be, g);
        ggml_backend_tensor_get_async(be, t_big, stage.data(), 0, N_BIG);
        ggml_backend_synchronize(be);
    }
    double overlapped = (now_ms() - t0) / 50;

    std::printf("sync3x[256B+16K+3M]: %.3f ms\n", sync3);
    std::printf("async3x+sync:        %.3f ms\n", async3);
    std::printf("async3x+event_sync:  %.3f ms\n", event3);
    std::printf("submit[1MB add]:     %.3f ms\n", submit);
    std::printf("serial[comp+3Mget]:  %.3f ms\n", serial);
    std::printf("overlap[comp+3Mget]: %.3f ms\n", overlapped);
    // ---- F: tiny batch (layer readback mix): sync 3x vs async+1sync ----
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get(t_small, stage.data(), 0, N_SMALL);
        ggml_backend_tensor_get(t_cur, stage.data(), 0, N_CUR);
        ggml_backend_tensor_get(t_w, stage.data(), 0, N_W);
    }
    double tiny_sync = (now_ms() - t0) / N;
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, stage.data(), 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur, stage.data(), 0, N_CUR);
        ggml_backend_tensor_get_async(be, t_w, stage.data(), 0, N_W);
        ggml_backend_synchronize(be);
    }
    double tiny_async = (now_ms() - t0) / N;
    std::printf("tiny3x[256B+16K+100K]sync:  %.3f ms\n", tiny_sync);
    std::printf("tiny3x[256B+16K+100K]async: %.3f ms\n", tiny_async);

    std::printf("RESULT integrity=%d\n", ok ? 1 : 0);

    ggml_backend_event_free(ev);
    ggml_backend_free(be);
    return ok ? 0 : 1;
}

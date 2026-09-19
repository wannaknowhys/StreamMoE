// vk_async_probe: measure RX590 Vulkan async transfer + event primitives.
// Compares:
//   1. Unpinned memory (std::vector heap): triggers internal ggml_vk_synchronize
//   2. Pinned host memory (HOST_VISIBLE|HOST_COHERENT): true non-blocking copyBuffer batch
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
    fill_pattern(h_small, 11); fill_pattern(h_cur, 22); fill_pattern(h_w, 23);
    fill_pattern(h_big, 33); fill_pattern(h_a, 44); fill_pattern(h_b, 55);

    // Unpinned staging buffers (standard CRT heap)
    std::vector<uint8_t> u_small(N_SMALL), u_cur(N_CUR), u_w(N_W), u_big(N_BIG);

    // Allocate Vulkan Pinned Host Memory
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) { std::printf("NO-HOST-BUFT host buffer type not found\n"); return 2; }
    const size_t TOTAL_PINNED = N_SMALL + N_CUR + N_W + N_BIG + 4096;
    ggml_backend_buffer_t pinned_buf = ggml_backend_buft_alloc_buffer(host_buft, TOTAL_PINNED);
    if (!pinned_buf) { std::printf("NO-PINNED-ALLOC pinned buffer alloc failed\n"); return 2; }
    uint8_t * p_base  = (uint8_t *) ggml_backend_buffer_get_base(pinned_buf);
    uint8_t * p_small = p_base;
    uint8_t * p_cur   = p_small + N_SMALL;
    uint8_t * p_w     = p_cur   + N_CUR;
    uint8_t * p_big   = p_w     + N_W;

    // Upload all inputs
    ggml_backend_tensor_set(t_small, h_small.data(), 0, N_SMALL);
    ggml_backend_tensor_set(t_cur,   h_cur.data(),   0, N_CUR);
    ggml_backend_tensor_set(t_w,     h_w.data(),     0, N_W);
    ggml_backend_tensor_set(t_big,   h_big.data(),   0, N_BIG);
    ggml_backend_tensor_set(t_a,     h_a.data(),     0, N_COMP);
    ggml_backend_tensor_set(t_b,     h_b.data(),     0, N_COMP);

    ggml_backend_event_t ev = ggml_backend_event_new(dev);
    if (!ev) { std::printf("NO-EVENT event_new failed\n"); return 2; }

    // Verify pinned integrity
    std::memset(p_base, 0, TOTAL_PINNED);
    ggml_backend_tensor_get_async(be, t_small, p_small, 0, N_SMALL);
    ggml_backend_tensor_get_async(be, t_cur,   p_cur,   0, N_CUR);
    ggml_backend_tensor_get_async(be, t_big,   p_big,   0, N_BIG);
    ggml_backend_event_record(ev, be);
    ggml_backend_event_synchronize(ev);
    bool ok = (std::memcmp(p_small, h_small.data(), N_SMALL) == 0) &&
              (std::memcmp(p_cur,   h_cur.data(),   N_CUR)   == 0) &&
              (std::memcmp(p_big,   h_big.data(),   N_BIG)   == 0);
    std::printf("pinned async integrity: %s\n", ok ? "OK" : "MISMATCH");
    if (!ok) return 1;

    const int N = 200;

    // =========================================================================
    // Benchmark 1: Large Batch (256B + 16KB + 3MB)
    // =========================================================================
    // 1A. Unpinned 3x sync
    double t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get(t_small, u_small.data(), 0, N_SMALL);
        ggml_backend_tensor_get(t_cur,   u_cur.data(),   0, N_CUR);
        ggml_backend_tensor_get(t_big,   u_big.data(),   0, N_BIG);
    }
    double unpinned_sync3 = (now_ms() - t0) / N;

    // 1B. Unpinned 3x async + 1 sync (the flawed test: degraded into 3x sync)
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, u_small.data(), 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur,   u_cur.data(),   0, N_CUR);
        ggml_backend_tensor_get_async(be, t_big,   u_big.data(),   0, N_BIG);
        ggml_backend_synchronize(be);
    }
    double unpinned_async3 = (now_ms() - t0) / N;

    // 1C. Pinned 3x sync
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get(t_small, p_small, 0, N_SMALL);
        ggml_backend_tensor_get(t_cur,   p_cur,   0, N_CUR);
        ggml_backend_tensor_get(t_big,   p_big,   0, N_BIG);
    }
    double pinned_sync3 = (now_ms() - t0) / N;

    // 1D. Pinned 3x async + 1 sync (GENUINE async batch)
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, p_small, 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur,   p_cur,   0, N_CUR);
        ggml_backend_tensor_get_async(be, t_big,   p_big,   0, N_BIG);
        ggml_backend_synchronize(be);
    }
    double pinned_async3 = (now_ms() - t0) / N;

    // 1E. Pinned 3x async + event sync
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, p_small, 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur,   p_cur,   0, N_CUR);
        ggml_backend_tensor_get_async(be, t_big,   p_big,   0, N_BIG);
        ggml_backend_event_record(ev, be);
        ggml_backend_event_synchronize(ev);
    }
    double pinned_event3 = (now_ms() - t0) / N;

    // =========================================================================
    // Benchmark 2: Layer Readback Mix (Tiny: 256B ids + 16KB cur + 100KB weights)
    // =========================================================================
    // 2A. Unpinned tiny 3x sync
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get(t_small, u_small.data(), 0, N_SMALL);
        ggml_backend_tensor_get(t_cur,   u_cur.data(),   0, N_CUR);
        ggml_backend_tensor_get(t_w,     u_w.data(),     0, N_W);
    }
    double unpinned_tiny_sync = (now_ms() - t0) / N;

    // 2B. Unpinned tiny 3x async (degraded)
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, u_small.data(), 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur,   u_cur.data(),   0, N_CUR);
        ggml_backend_tensor_get_async(be, t_w,     u_w.data(),     0, N_W);
        ggml_backend_synchronize(be);
    }
    double unpinned_tiny_async = (now_ms() - t0) / N;

    // 2C. Pinned tiny 3x sync
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get(t_small, p_small, 0, N_SMALL);
        ggml_backend_tensor_get(t_cur,   p_cur,   0, N_CUR);
        ggml_backend_tensor_get(t_w,     p_w,     0, N_W);
    }
    double pinned_tiny_sync = (now_ms() - t0) / N;

    // 2D. Pinned tiny 3x async + 1 sync (GENUINE tiny batch)
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, p_small, 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur,   p_cur,   0, N_CUR);
        ggml_backend_tensor_get_async(be, t_w,     p_w,     0, N_W);
        ggml_backend_synchronize(be);
    }
    double pinned_tiny_async = (now_ms() - t0) / N;

    // 2E. Pinned tiny 3x async + event sync
    t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        ggml_backend_tensor_get_async(be, t_small, p_small, 0, N_SMALL);
        ggml_backend_tensor_get_async(be, t_cur,   p_cur,   0, N_CUR);
        ggml_backend_tensor_get_async(be, t_w,     p_w,     0, N_W);
        ggml_backend_event_record(ev, be);
        ggml_backend_event_synchronize(ev);
    }
    double pinned_tiny_event = (now_ms() - t0) / N;

    // Output Report
    std::printf("\n================ [VK ASYNC PROBE REPORT] ================\n");
    std::printf("--- Big Batch (256B + 16KB + 3MB) ---\n");
    std::printf("  Unpinned 3x sync:            %.3f ms\n", unpinned_sync3);
    std::printf("  Unpinned 3x async (degraded):%.3f ms  [delta vs sync: %+.1f%%]\n",
                unpinned_async3, (unpinned_async3 - unpinned_sync3) / unpinned_sync3 * 100.0);
    std::printf("  Pinned 3x sync:              %.3f ms\n", pinned_sync3);
    std::printf("  Pinned 3x async + 1 sync:    %.3f ms  <-- GENUINE ASYNC\n", pinned_async3);
    std::printf("  Pinned 3x async + event:     %.3f ms\n", pinned_event3);
    std::printf("  -> Speedup (genuine vs sync): %.2fx (saves %.3f ms)\n",
                unpinned_sync3 / pinned_async3, unpinned_sync3 - pinned_async3);

    std::printf("\n--- Tiny Layer Readback Mix (256B ids + 16KB cur + 100KB w) ---\n");
    std::printf("  Unpinned 3x sync:            %.3f ms\n", unpinned_tiny_sync);
    std::printf("  Unpinned 3x async (degraded):%.3f ms  [delta vs sync: %+.1f%%]\n",
                unpinned_tiny_async, (unpinned_tiny_async - unpinned_tiny_sync) / unpinned_tiny_sync * 100.0);
    std::printf("  Pinned 3x sync:              %.3f ms\n", pinned_tiny_sync);
    std::printf("  Pinned 3x async + 1 sync:    %.3f ms  <-- GENUINE ASYNC\n", pinned_tiny_async);
    std::printf("  Pinned 3x async + event:     %.3f ms\n", pinned_tiny_event);
    std::printf("  -> Speedup (genuine vs sync): %.2fx (saves %.3f ms)\n",
                unpinned_tiny_sync / pinned_tiny_async, unpinned_tiny_sync - pinned_tiny_async);
    std::printf("=========================================================\n\n");

    ggml_backend_event_free(ev);
    ggml_backend_buffer_free(pinned_buf);
    ggml_backend_free(be);
    return 0;
}

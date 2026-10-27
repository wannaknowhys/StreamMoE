// vk_launch_probe: fixed cost of one Vulkan graph submission on this driver.
//
// Route-B executes ~2 device graphs per layer (dense head, MoE+tail). This probe
// measures the same ggml path (ggml_backend_graph_compute = launch + fence wait)
// for graphs of 1..N nodes so the fixed per-submission cost can be separated
// from the actual node work.
//
// Build:  build.bat harness StreamMoE vk_launch_probe
// Run:    build\StreamMoE\bin\vk_launch_probe.exe
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Build a chain of `n_adds` F32 add nodes over [n] elements and time N submits.
static double run_chain(ggml_backend_t be, int n_adds, int n, int iters) {
    ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * cur = a;
    std::vector<ggml_tensor *> nodes;
    for (int i = 0; i < n_adds; ++i) {
        cur = ggml_add(ctx, cur, (i & 1) ? b : a);
        nodes.push_back(cur);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return -1.0; }
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, cur);

    for (int i = 0; i < 10; ++i) ggml_backend_graph_compute(be, g);   // warm up
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g);
    const double dt = (now_ms() - t0) / iters;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return dt;
}

// One MUL_MAT node per graph: how much per-node overhead matters vs real compute.
static double run_mm(ggml_backend_t be, int n, int iters, double * gflop_out) {
    ggml_init_params ip = { 512u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return -1.0; }
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, c);
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, g);
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g);
    const double dt = (now_ms() - t0) / iters;
    if (gflop_out) *gflop_out = 2.0 * n * n * n / 1e9;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return dt;
}

int main(int argc, char ** argv) {
    const char * dev_name = (argc > 1) ? argv[1] : "Vulkan0";
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
    if (!dev) { std::printf("NO-DEVICE %s\n", dev_name); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { std::printf("NO-INIT\n"); return 2; }

    std::printf("device: %s\n", dev_name);
    std::printf("graph submission cost (graph_compute = launch + fence wait)\n");
    std::printf("%-10s %-14s %-14s\n", "nodes", "128 KB add", "1 MB add");
    const int shapes[] = { 1, 2, 4, 8, 16, 38, 76 };
    for (int k : shapes) {
        const double small = run_chain(be, k, 32 * 1024 / 4, 200);
        const double big   = run_chain(be, k, 256 * 1024 / 4, 200);
        std::printf("%-10d %-14.3f %-14.3f\n", k, small, big);
    }

    // Break penalty: the same 38 ops as ONE graph vs 38 separate 1-node graphs.
    {
        ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024 / 4);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024 / 4);
        ggml_cgraph * one = ggml_new_graph_custom(ctx, 8, false);
        ggml_tensor * n0 = ggml_add(ctx, a, b);
        ggml_build_forward_expand(one, n0);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        for (int i = 0; i < 10; ++i) for (int k = 0; k < 38; ++k) ggml_backend_graph_compute(be, one);
        const double t0 = now_ms();
        for (int i = 0; i < 200; ++i) for (int k = 0; k < 38; ++k) ggml_backend_graph_compute(be, one);
        const double split = (now_ms() - t0) / 200;
        std::printf("\n38 ops as 38 one-node graphs: %.3f ms  (%.3f ms/op)\n", split, split / 38);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Queue-depth test: is the per-submit cost a fence latency or real GPU time?
    // Enqueue N submits without waiting, then wait once.
    {
        ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024 / 4);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024 / 4);
        ggml_tensor * cur = a;
        for (int i = 0; i < 38; ++i) cur = ggml_add(ctx, cur, (i & 1) ? b : a);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, cur);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(be, g);
        const int N = 100;
        const double t0 = now_ms();
        for (int i = 0; i < N; ++i) ggml_backend_graph_compute_async(be, g);
        const double t_enq = now_ms() - t0;
        const double t1 = now_ms();
        ggml_backend_synchronize(be);
        const double t_sync = now_ms() - t1;
        std::printf("\nqueue depth: enqueue %d x 38-node graphs = %.1f ms total (%.3f ms/submit enqueue)\n", N, t_enq, t_enq / N);
        std::printf("             one final sync = %.1f ms -> GPU %.3f ms/graph, %.1f us/node\n",
                    t_sync, t_sync / N, 1000.0 * t_sync / N / 38.0);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // Per-node overhead vs compute: one MUL_MAT (f32) per graph.
    std::printf("\none MUL_MAT per graph (per-node overhead vs compute):\n");
    std::printf("%-6s %-12s %-12s %-12s\n", "n", "ms", "GFLOP", "TFLOPS");
    const int mms[] = { 256, 512, 1024, 2048 };
    for (int n : mms) {
        double gf = 0.0;
        const int iters = (n >= 2048) ? 10 : 40;
        const double ms = run_mm(be, n, iters, &gf);
        std::printf("%-6d %-12.3f %-12.3f %-12.3f\n", n, ms, gf, gf / (ms / 1000.0));
    }

    // Async submit + explicit fence wait, for the overlap path route B uses on
    // its device rounds (one graph, 38 nodes, 1 MB).
    {
        ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024 / 4);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256 * 1024 / 4);
        ggml_tensor * cur = a;
        std::vector<ggml_tensor *> nodes;
        for (int i = 0; i < 38; ++i) { cur = ggml_add(ctx, cur, (i & 1) ? b : a); nodes.push_back(cur); }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, cur);
        for (int i = 0; i < 10; ++i) ggml_backend_graph_compute(be, g);
        double t0 = now_ms();
        for (int i = 0; i < 200; ++i) ggml_backend_graph_compute_async(be, g);
        const double async_only = (now_ms() - t0) / 200;
        ggml_backend_synchronize(be);
        t0 = now_ms();
        for (int i = 0; i < 200; ++i) { ggml_backend_graph_compute_async(be, g); ggml_backend_synchronize(be); }
        const double async_sync = (now_ms() - t0) / 200;
        std::printf("\nasync-only (no wait) : %.3f ms/submit\n", async_only);
        std::printf("async + sync        : %.3f ms/submit\n", async_sync);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(be);
    return 0;
}

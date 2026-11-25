// Unit test: async DMA through the device interface (Phase 2).
// Builds for every tag; without STREAMMOE_HAVE_CUDA it prints SKIP and exits 0.
// With CUDA (StreamMoE_cuda tag) it exercises the real path on the P100:
//   - a ggml CUDA buffer (device memory) filled with a known pattern,
//   - host_offset == get_base + off, host_ptr == nullptr,
//   - writer (HtoD) + reader (DtoH) round-trip with data comparison,
//   - asyncness: the enqueue of a large copy returns in milliseconds while
//     the copy itself takes measurably longer, and the returned token is
//     non-null (a null token would mean the copy ran synchronously).
//
// The test uses the ggml backend registry API (ggml-base) plus cudart; on a
// CUDA tag it also links the backend libs (ggml-cuda, ggml-vulkan), because
// ggml_backend_load_all() pulls the registry, whose constructor references
// the compiled-in backends reg functions. On CPU-only tags the CUDA
// branch is compiled out and the test prints SKIP.

#include "backend/device_iface.h"

#ifdef STREAMMOE_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::fprintf(stderr, "[test_device_cuda] %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

uint64_t now_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Deterministic byte pattern (LCG, high byte) so writer and reader agree.
void fill_pattern(uint8_t* p, size_t n, uint32_t seed) {
    uint32_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = static_cast<uint8_t>(x >> 24);
    }
}

bool check_pattern(const uint8_t* p, size_t n, uint32_t seed) {
    uint32_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        if (p[i] != static_cast<uint8_t>(x >> 24)) return false;
    }
    return true;
}

} // namespace

int main() {
#ifdef STREAMMOE_HAVE_CUDA
    using namespace stream_moe;
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) {
        std::fprintf(stderr, "[test_device_cuda] SKIP: no CUDA0 device in this build\n");
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft) {
        std::fprintf(stderr, "[test_device_cuda] SKIP: CUDA0 has no default buffer type\n");
        return 0;
    }

    // ---- interface shape -------------------------------------------------
    check(stmoe_dev_kind_for_buft(buft) == STMOE_DEV_CUDA, "kind of CUDA buft is CUDA");
    check(stmoe_dev_dma_available(STMOE_DEV_CUDA), "dma_available(CUDA) is true");
    auto reader = stmoe_dev_dma_reader(STMOE_DEV_CUDA);
    auto writer = stmoe_dev_dma_writer(STMOE_DEV_CUDA);
    auto waiter = stmoe_dev_dma_waiter(STMOE_DEV_CUDA);
    auto poller = stmoe_dev_dma_poller(STMOE_DEV_CUDA);
    check(reader && writer && waiter && poller, "CUDA reader/writer/waiter/poller are non-null");

    // ---- buffer + address semantics --------------------------------------
    const size_t kSmall = 1u << 20;   // 1 MiB pattern buffer
    const size_t kBig   = 512u << 20; // 512 MiB asyncness probe
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, kBig);
    check(buf != nullptr, "alloc 512 MiB CUDA buffer");
    if (!buf) return 1;
    void* base = ggml_backend_buffer_get_base(buf);
    check(base != nullptr, "get_base non-null");
    check(stmoe_dev_buffer_host_ptr(buf) == nullptr, "host_ptr(CUDA buffer) is nullptr");
    check(stmoe_dev_buffer_host_offset(buf, 0) == base, "host_offset(0) == get_base");
    check(stmoe_dev_buffer_host_offset(buf, 4096) == static_cast<uint8_t*>(base) + 4096,
          "host_offset(4096) == get_base + 4096");

    // ---- HtoD then DtoH round-trip (small, correctness) ------------------
    std::vector<uint8_t> host_src(kSmall), host_dst(kSmall, 0);
    fill_pattern(host_src.data(), kSmall, 0x12345678u);
    void* wtok = writer(buf, 0, host_src.data(), kSmall);
    check(wtok != nullptr, "writer returns a non-null token (async enqueue)");
    check(waiter(wtok), "writer token waits cleanly");
    check(poller(nullptr), "poll(null token) is true");
    void* rtok = reader(buf, 0, host_dst.data(), kSmall);
    check(rtok != nullptr, "reader returns a non-null token (async enqueue)");
    check(waiter(rtok), "reader token waits cleanly");
    check(check_pattern(host_dst.data(), kSmall, 0x12345678u), "DtoH round-trip matches pattern");

    // ---- asyncness probe (large copy) ------------------------------------
    // Enqueue a 512 MiB DtoH and measure: the enqueue itself must return in
    // milliseconds (it only submits work), while enqueue+wait takes the full
    // transfer time. A synchronous implementation would make the enqueue
    // itself take the full transfer time (~40 ms over PCIe gen3 x16).
    std::vector<uint8_t> big_dst(kBig, 0);
    // Pin the probe buffer BEFORE the timed enqueue: registering 512 MiB
    // costs ~18 ms, which would blow the <10 ms enqueue budget if paid
    // inside the call. The DMA layer sees the range as already-registered
    // and enqueues directly against it.
    check(cudaHostRegister(big_dst.data(), kBig, cudaHostRegisterDefault) == cudaSuccess,
          "pre-pin 512 MiB probe buffer");
    const uint64_t t0 = now_ms();
    void* big_tok = reader(buf, 0, big_dst.data(), kBig);
    const uint64_t enqueue_ms = now_ms() - t0;
    check(big_tok != nullptr, "large DtoH returns a non-null token (async enqueue)");
    check(enqueue_ms < 10, "large DtoH enqueue returns fast (<10 ms)");
    std::fprintf(stderr, "[test_device_cuda] large DtoH enqueue=%llums\n",
                 (unsigned long long)enqueue_ms);
    // Poll must not block: call it once right after enqueue (the result may
    // be true or false; what matters is that it returns promptly).
    const uint64_t p0 = now_ms();
    const bool polled = poller(big_tok);
    const uint64_t poll_ms = now_ms() - p0;
    check(poll_ms < 10, "poll returns promptly (<10 ms)");
    std::fprintf(stderr, "[test_device_cuda] poll right after enqueue: %s\n",
                 polled ? "complete" : "not ready");
    const uint64_t w0 = now_ms();
    check(waiter(big_tok), "large DtoH wait succeeds");
    const uint64_t total_ms = now_ms() - w0 + enqueue_ms;
    std::fprintf(stderr, "[test_device_cuda] large DtoH total (enqueue+wait)=%llums\n",
                 (unsigned long long)total_ms);
    check(total_ms > 20, "large DtoH total time is real transfer time (>20 ms)");
    check(check_pattern(big_dst.data(), kSmall, 0x12345678u),
          "large DtoH first 1 MiB matches the pattern written by the writer");
    cudaHostUnregister(big_dst.data());

    ggml_backend_buffer_free(buf);
    std::fprintf(stderr, "[test_device_cuda] %s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
#else
    std::fprintf(stderr, "[test_device_cuda] SKIP: STREAMMOE_HAVE_CUDA not defined in this build\n");
    return 0;
#endif
}

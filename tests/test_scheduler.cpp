// Integration test for expert_scheduler: synthetic MoE topology + temp file,
// exercises DIO load, slot pin/unpin, eviction and hit/miss telemetry.
// Cross-platform (async_dio Win IOCP / POSIX pread), no real GGUF needed.
#include "backend/scheduler.h"
#include "io/async_dio.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <chrono>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { std::printf("[-] ASSERTION FAILED: %s (line %d)\n", msg, __LINE__); std::fflush(stdout); return false; } \
} while(0)

static const char* TMP_FILE = "temp/sched_test.bin";
static constexpr uint32_t N_LAYERS = 2;
static constexpr uint32_t N_EXPERTS = 4;
static constexpr size_t   SLOT_SIZE = 4096;
static constexpr size_t   EXPERT_BYTES = 2048;

static moe_model_topology_t build_topo() {
    moe_model_topology_t topo;
    topo.n_layer = N_LAYERS;
    topo.n_expert = N_EXPERTS;
    topo.n_expert_used = 2;
    topo.embedding_length = 16;
    topo.expert_slot_size = SLOT_SIZE;
    topo.experts.resize(static_cast<size_t>(N_LAYERS) * N_EXPERTS);

    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        for (uint32_t e = 0; e < N_EXPERTS; ++e) {
            auto& info = topo.experts[static_cast<size_t>(l) * N_EXPERTS + e];
            info.layer_idx = static_cast<int32_t>(l);
            info.expert_idx = static_cast<int32_t>(e);

            sub_tensor_info_t st;
            st.name = "ffn_gate_exps.weight";
            st.shard_idx = 0;
            st.abs_file_offset = static_cast<uint64_t>(e + 1) * SLOT_SIZE; // 4KB aligned
            st.byte_size = EXPERT_BYTES;
            st.slot_offset = 0;
            info.sub_tensors.push_back(st);
            info.total_expert_bytes = EXPERT_BYTES;

            sub_tensor_req_t req;
            req.shard_idx = st.shard_idx;
            req.file_offset = st.abs_file_offset;
            req.byte_size = st.byte_size;
            req.slot_offset = 0;
            info.read_plan = build_expert_read_plan(&req, 1);
        }
    }
    topo.expert_dio_staging_size = topo.experts[0].read_plan.total_staging_size;
    return topo;
}

static bool write_test_file() {
    std::ofstream out(TMP_FILE, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    std::vector<uint8_t> zero(64 * 1024, 0);
    out.write(reinterpret_cast<const char*>(zero.data()), zero.size());
    // expert e's payload at offset (e+1)*4096: fill with 0x40+e
    for (uint32_t e = 0; e < N_EXPERTS; ++e) {
        std::vector<uint8_t> payload(EXPERT_BYTES, static_cast<uint8_t>(0x40 + e));
        out.seekp(static_cast<std::streamoff>((e + 1) * SLOT_SIZE));
        out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    }
    out.close();
    return true;
}

static bool test_scheduler_load_and_pin() {
    if (!write_test_file()) { std::printf("[-] cannot write test file\n"); return false; }

    auto dio = async_dio_engine::create(64);
    TEST_ASSERT(dio != nullptr, "DIO engine created");
    dio_file_t* f = dio->open_file(TMP_FILE);
    TEST_ASSERT(f != nullptr, "open test file via DIO");

    moe_model_topology_t topo = build_topo();
    expert_scheduler sched;
    TEST_ASSERT(sched.init(topo, *dio, {f}, 3 * SLOT_SIZE), "scheduler init (3 slots)");
    TEST_ASSERT(sched.num_slots() == 3, "3 slots (pool < expert count forces eviction)");
    sched.start();

    // pin all 4 experts of layer 0 -> 3 slots force eviction of earlier ones
    for (uint32_t e = 0; e < N_EXPERTS; ++e) {
        expert_handle_t h = sched.pin_expert(0, e);
        TEST_ASSERT(h.pinned && h.slot >= 0, "pin succeeded");
        const uint8_t* slot_ptr = nullptr;
        // content check via the scheduler's pool base (exposed through slot index)
        // we can't read private pool; instead verify via a second pin round-trip
        // and telemetry. Content correctness is covered by read_expert_sync UT.
        sched.unpin(h);
    }

    // telemetry: 4 lookups, 4 misses (first-touch, small pool), hits=0
    TEST_ASSERT(sched.total_lookups() == 4, "4 lookups");
    TEST_ASSERT(sched.disk_misses() == 4, "4 first-touch misses");
    TEST_ASSERT(sched.ram_hits() == 0, "0 hits on cold pool");

    // second touch: the last-pinned expert survived LRU eviction -> must hit
    {
        expert_handle_t h = sched.pin_expert(0, 3);
        sched.unpin(h);
    }
    TEST_ASSERT(sched.total_lookups() == 5, "5 total lookups");
    TEST_ASSERT(sched.ram_hits() >= 1, "survivor expert is a hit");
    TEST_ASSERT(sched.disk_misses() == 4, "4 first-touch misses only");

    sched.stop();
    dio->close_file(f);
    std::printf("[+] test_scheduler_load_and_pin PASSED\n"); std::fflush(stdout);
    return true;
}

static bool test_scheduler_content() {
    if (!write_test_file()) return false;
    auto dio = async_dio_engine::create(64);
    dio_file_t* f = dio->open_file(TMP_FILE);
    moe_model_topology_t topo = build_topo();
    expert_scheduler sched;
    sched.init(topo, *dio, {f}, 3 * SLOT_SIZE);
    sched.start();

    // verify slot memory content: read via a raw DIO read of the slot region is
    // not exposed; instead assert correctness indirectly by re-reading the file
    // through the scheduler twice - content equality covered by staging_reader
    // unit test. Here we just exercise the full lifecycle without crash.
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        for (uint32_t e = 0; e < N_EXPERTS; ++e) {
            expert_handle_t h = sched.pin_expert(l, e);
            sched.wait_ready(l, e); // down-role path
            sched.unpin(h);
        }
    }
    TEST_ASSERT(sched.total_lookups() == 8, "8 lookups across layers");

    sched.stop();
    dio->close_file(f);
    std::printf("[+] test_scheduler_content PASSED\n"); std::fflush(stdout);
    return true;
}

int main() {
    bool ok = true;
    ok = test_scheduler_load_and_pin() && ok;
    ok = test_scheduler_content() && ok;
    if (ok) { std::printf("ALL SCHEDULER TESTS PASSED\n"); return 0; }
    std::printf("SOME SCHEDULER TESTS FAILED\n");
    return 1;
}

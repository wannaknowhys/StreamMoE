#include "io/async_dio.h"
#include "io/staging_reader.h"
#include "common/logger.h"
#include "common/types.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_PASS(name) \
    std::cout << "[+] TEST PASSED: " << name << std::endl

// Test 1: Alignment & Aligned Allocator
bool test_alignment_and_allocator() {
    TEST_ASSERT(align_floor(0) == 0, "align_floor(0)");
    TEST_ASSERT(align_floor(4095) == 0, "align_floor(4095)");
    TEST_ASSERT(align_floor(4096) == 4096, "align_floor(4096)");
    TEST_ASSERT(align_floor(4097) == 4096, "align_floor(4097)");

    TEST_ASSERT(align_ceil(0) == 0, "align_ceil(0)");
    TEST_ASSERT(align_ceil(1) == 4096, "align_ceil(1)");
    TEST_ASSERT(align_ceil(4096) == 4096, "align_ceil(4096)");
    TEST_ASSERT(align_ceil(4097) == 8192, "align_ceil(4097)");

    // Test VirtualAlloc 4KB alignment
    size_t sizes[] = { 1, 4096, 65536, 1024 * 1024 * 4 };
    for (size_t sz : sizes) {
        auto buf = make_aligned_buffer(sz);
        TEST_ASSERT(buf.get() != nullptr, "Buffer allocation failed");
        TEST_ASSERT(is_aligned(buf.get(), 4096), "Buffer not 4KB aligned");
        
        // Write pattern
        std::memset(buf.get(), 0xAA, sz);
        TEST_ASSERT(buf.get()[0] == 0xAA && buf.get()[sz - 1] == 0xAA, "Memory write check");
    }

    TEST_PASS("test_alignment_and_allocator");
    return true;
}

// Test 2: Mock File Direct I/O Unbuffered Reading
bool test_mock_file_dio() {
    const std::string mock_file_path = "temp/mock_dio_test.bin";
    const size_t file_size = 2 * 1024 * 1024; // 2MB

    std::vector<uint8_t> ground_truth(file_size);
    std::mt19937 rng(42);
    for (size_t i = 0; i < file_size; ++i) {
        ground_truth[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    // Write ground truth file
    {
        std::ofstream out(mock_file_path, std::ios::binary);
        TEST_ASSERT(out.is_open(), "Failed to open mock file for writing");
        out.write(reinterpret_cast<const char*>(ground_truth.data()), file_size);
    }

    auto engine = async_dio_engine::create(64);
    TEST_ASSERT(engine != nullptr, "Failed to create DIO engine");

    dio_file_t* file = engine->open_file(mock_file_path);
    TEST_ASSERT(file != nullptr, "Failed to open file via DIO");
    TEST_ASSERT(file->file_size == file_size, "File size mismatch");

    // Perform aligned DIO read
    const size_t test_offset = 8192;
    const size_t test_len = 16384;
    auto read_buf = make_aligned_buffer(test_len);

    aio_req_t req;
    req.file        = file;
    req.file_offset = test_offset;
    req.aligned_buf = read_buf.get();
    req.aligned_len = static_cast<uint32_t>(test_len);

    int32_t sub = engine->submit_batch(&req, 1);
    TEST_ASSERT(sub == 1, "submit_batch failed");

    aio_req_t* comp = nullptr;
    uint32_t n = engine->wait_events(&comp, 1, 1, 3000);
    TEST_ASSERT(n == 1 && comp == &req, "wait_events failed");
    TEST_ASSERT(req.error_code == 0, "IO error in request");
    TEST_ASSERT(req.bytes_read == test_len, "Bytes read mismatch");

    TEST_ASSERT(std::memcmp(read_buf.get(), ground_truth.data() + test_offset, test_len) == 0, 
                "DIO read data mismatch with ground truth");

    engine->close_file(file);
    TEST_PASS("test_mock_file_dio");
    return true;
}

// Test 3: Multi-Tensor Staging Reader
bool test_multi_tensor_staging_reader() {
    const std::string mock_file_path = "temp/mock_dio_test.bin";
    
    // Read ground truth back for comparison
    std::ifstream in(mock_file_path, std::ios::binary);
    std::vector<uint8_t> ground_truth((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto engine = async_dio_engine::create(64);
    dio_file_t* file = engine->open_file(mock_file_path);
    TEST_ASSERT(file != nullptr, "Failed to open mock file");

    // Define 3 unaligned sub-tensors inside an expert (e.g. SwiGLU: gate, up, down)
    sub_tensor_req_t tensors[3];
    tensors[0] = { 1234,   54321,  0 };      // gate: offset 1234, size 54321
    tensors[1] = { 78901,  32100,  54321 };  // up: offset 78901, size 32100
    tensors[2] = { 200013, 65432,  86421 };  // down: offset 200013, size 65432

    expert_read_plan_t plan = build_expert_read_plan(tensors, 3);
    TEST_ASSERT(plan.num_tensors == 3, "Plan num_tensors mismatch");
    TEST_ASSERT(plan.total_staging_size > 0, "Staging size must be > 0");
    TEST_ASSERT(plan.total_slot_size >= (86421 + 65432), "Slot size calculation mismatch");

    auto staging_buf = make_aligned_buffer(plan.total_staging_size);
    auto slot_buf    = make_aligned_buffer(plan.total_slot_size);

    bool ok = read_expert_sync(engine.get(), file, plan, staging_buf.get(), slot_buf.get());
    TEST_ASSERT(ok, "read_expert_sync failed");

    // Verify all 3 sub-tensors in the slot
    for (int i = 0; i < 3; ++i) {
        const uint8_t* slot_slice = slot_buf.get() + tensors[i].slot_offset;
        const uint8_t* gt_slice   = ground_truth.data() + tensors[i].file_offset;
        TEST_ASSERT(std::memcmp(slot_slice, gt_slice, tensors[i].byte_size) == 0,
                    "Sub-tensor payload mismatch in reconstructed slot");
    }

    engine->close_file(file);
    TEST_PASS("test_multi_tensor_staging_reader");
    return true;
}

// Test 4: Real GGUF Direct I/O Read (Qwen3-VL and DeepSeek-V4 Shard)
bool test_real_gguf_dio_read() {
    auto engine = async_dio_engine::create(32);

    // 1. Test Qwen3-VL-2B (1.03 GB)
    const std::string qwen_path = "F:/Dev/computer-use/Qwen3-VL-2B-Instruct-Q4_K_M.gguf";
    dio_file_t* qwen_file = engine->open_file(qwen_path);
    if (qwen_file) {
        LOG_INFO("Found Qwen3-VL file, size: " << qwen_file->file_size << " bytes");
        auto buf = make_aligned_buffer(4096);
        aio_req_t req;
        req.file = qwen_file;
        req.file_offset = 0;
        req.aligned_buf = buf.get();
        req.aligned_len = 4096;

        int32_t sub = engine->submit_batch(&req, 1);
        TEST_ASSERT(sub == 1, "submit Qwen header read");
        aio_req_t* comp = nullptr;
        engine->wait_events(&comp, 1, 1, 3000);
        TEST_ASSERT(req.error_code == 0, "Qwen header read error");
        
        // GGUF Magic check ('G', 'G', 'U', 'F')
        TEST_ASSERT(buf.get()[0] == 'G' && buf.get()[1] == 'G' && buf.get()[2] == 'U' && buf.get()[3] == 'F',
                    "Qwen file GGUF magic check failed");
        uint32_t version = *reinterpret_cast<uint32_t*>(buf.get() + 4);
        LOG_INFO("Qwen3-VL GGUF Magic verified! Version: " << version);
        TEST_ASSERT(version == 2 || version == 3, "Unexpected GGUF version");
        engine->close_file(qwen_file);
    } else {
        LOG_WARN("Qwen3-VL test file not found at " << qwen_path << " (skipping)");
    }

    // 2. Test DeepSeek-V4 MoE Shard (Header only, zero-load test)
    const std::string ds_shard1_path = "N:/AI_LLM/DeepSeek-V4-Flash-0731/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf";
    dio_file_t* ds_file = engine->open_file(ds_shard1_path);
    if (ds_file) {
        LOG_INFO("Found DeepSeek-V4 Shard 1, size: " << ds_file->file_size << " bytes");
        auto buf = make_aligned_buffer(4096);
        aio_req_t req;
        req.file = ds_file;
        req.file_offset = 0;
        req.aligned_buf = buf.get();
        req.aligned_len = 4096;

        int32_t sub = engine->submit_batch(&req, 1);
        TEST_ASSERT(sub == 1, "submit DeepSeek header read");
        aio_req_t* comp = nullptr;
        engine->wait_events(&comp, 1, 1, 3000);
        TEST_ASSERT(req.error_code == 0, "DeepSeek header read error");

        // GGUF Magic check
        TEST_ASSERT(buf.get()[0] == 'G' && buf.get()[1] == 'G' && buf.get()[2] == 'U' && buf.get()[3] == 'F',
                    "DeepSeek Shard 1 GGUF magic check failed");
        uint32_t version = *reinterpret_cast<uint32_t*>(buf.get() + 4);
        LOG_INFO("DeepSeek-V4 MoE Shard 1 GGUF Magic verified! Version: " << version);
        TEST_ASSERT(version == 2 || version == 3, "Unexpected GGUF version");
        engine->close_file(ds_file);
    } else {
        LOG_WARN("DeepSeek-V4 Shard 1 test file not found (skipping)");
    }

    TEST_PASS("test_real_gguf_dio_read");
    return true;
}

// Test 5: Concurrent Multi-Batch Overlap
bool test_concurrent_batch_dio() {
    const std::string mock_file_path = "temp/mock_dio_test.bin";
    auto engine = async_dio_engine::create(64);
    dio_file_t* file = engine->open_file(mock_file_path);
    TEST_ASSERT(file != nullptr, "Failed to open mock file");

    const uint32_t num_reqs = 16;
    std::vector<aio_req_t> reqs(num_reqs);
    std::vector<aligned_buffer_ptr> bufs;

    for (uint32_t i = 0; i < num_reqs; ++i) {
        bufs.push_back(make_aligned_buffer(4096));
        reqs[i].file        = file;
        reqs[i].file_offset = i * 8192;
        reqs[i].aligned_buf = bufs[i].get();
        reqs[i].aligned_len = 4096;
        reqs[i].user_data   = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
    }

    int32_t sub = engine->submit_batch(reqs.data(), num_reqs);
    TEST_ASSERT(sub == static_cast<int32_t>(num_reqs), "submit_batch concurrent failed");

    std::vector<aio_req_t*> completed(num_reqs, nullptr);
    uint32_t done = 0;
    while (done < num_reqs) {
        uint32_t c = engine->wait_events(completed.data() + done, num_reqs - done, 1, 5000);
        TEST_ASSERT(c > 0, "wait_events timeout");
        done += c;
    }

    for (uint32_t i = 0; i < num_reqs; ++i) {
        TEST_ASSERT(reqs[i].error_code == 0, "Request error in batch");
        TEST_ASSERT(reqs[i].bytes_read == 4096, "Bytes read mismatch");
    }

    engine->close_file(file);
    TEST_PASS("test_concurrent_batch_dio");
    return true;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Running StreamMoE Phase 1 Unit Tests     " << std::endl;
    std::cout << "===========================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_alignment_and_allocator();
    all_passed &= test_mock_file_dio();
    all_passed &= test_multi_tensor_staging_reader();
    all_passed &= test_real_gguf_dio_read();
    all_passed &= test_concurrent_batch_dio();

    std::cout << "===========================================" << std::endl;
    if (all_passed) {
        std::cout << "  ALL UNIT TESTS PASSED SUCCESSFULLY!       " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 0;
    } else {
        std::cerr << "  SOME UNIT TESTS FAILED!                   " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 1;
    }
}
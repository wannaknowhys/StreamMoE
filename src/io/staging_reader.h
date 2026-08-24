#pragma once

#include "io/async_dio.h"
#include "common/types.h"
#include <vector>

namespace stream_moe {

#define MAX_SUB_TENSORS 8

struct sub_tensor_req_t {
    uint64_t file_offset; // Absolute file offset in GGUF
    uint64_t byte_size;   // Actual tensor size in bytes
    uint64_t slot_offset; // Destination offset inside the expert slot
};

struct expert_read_plan_t {
    uint32_t           num_tensors = 0;
    size_t             total_slot_size = 0;
    size_t             total_staging_size = 0;
    sub_tensor_req_t   tensors[MAX_SUB_TENSORS];
    
    // Aligned metadata computed for DIO
    uint64_t           aligned_offsets[MAX_SUB_TENSORS];
    uint32_t           aligned_lens[MAX_SUB_TENSORS];
    uint32_t           staging_offsets[MAX_SUB_TENSORS];
    uint32_t           slice_shifts[MAX_SUB_TENSORS];
};

// Build an optimized read plan for an expert
expert_read_plan_t build_expert_read_plan(const sub_tensor_req_t* tensors, uint32_t num_tensors);

// Read an entire expert into a compact slot using a staging buffer
bool read_expert_sync(
    async_dio_engine* engine,
    dio_file_t* file,
    const expert_read_plan_t& plan,
    uint8_t* staging_buf,
    uint8_t* dest_slot_ptr
);

} // namespace stream_moe
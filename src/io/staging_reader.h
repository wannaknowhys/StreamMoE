#pragma once

#include "common/types.h"
#include "io/async_dio.h"
#include <cstdint>
#include <vector>

namespace stream_moe {

struct sub_tensor_req_t {
    uint32_t shard_idx;   // Shard file index
    uint64_t file_offset; // Absolute file offset within that shard
    uint64_t byte_size;   // Byte length of this tensor slice
    uint32_t column;      // SoA column index this slice belongs to (2026-09)
    uint64_t col_off;     // offset of this slice's payload inside the column slot
};

struct tensor_slice_read_t {
    uint32_t shard_idx;       // Shard file index
    uint64_t file_read_start; // 4KB sector aligned start in file
    uint32_t file_read_len;   // 4KB sector aligned length in file
    uint32_t column;          // SoA column index (dst = that column's slot slice)
    size_t   staging_offset;  // Offset in temporary staging buffer
    size_t   copy_src_offset; // Offset in staging buffer where valid payload starts
    size_t   copy_dst_offset; // Offset inside the column slot slice
    size_t   copy_byte_len;   // Exact payload byte count to copy
    bool     direct;          // source aligned + len % 4096 == 0: DIO straight into column slot
};

struct expert_read_plan_t {
    uint32_t            num_tensors        = 0;
    size_t              total_staging_size = 0;
    size_t              total_slot_size    = 0;
    tensor_slice_read_t slices[MAX_SUB_TENSORS_PER_EXPERT];
};

// Compute sector-aligned Direct I/O plan for loading an expert with N sub-tensors
expert_read_plan_t build_expert_read_plan(
    const sub_tensor_req_t* reqs,
    uint32_t                num_tensors,
    size_t                  sector_size = 4096
);

// Synchronous wrapper to execute batch Direct I/O and compact into slot buffer
bool read_expert_sync(
    async_dio_engine*               engine,
    const std::vector<dio_file_t*>& shard_files,
    const expert_read_plan_t&       plan,
    uint8_t*                        staging_buffer,
    uint8_t*                        slot_buffer
);

// Single file convenience overload
bool read_expert_sync(
    async_dio_engine*         engine,
    dio_file_t*               file,
    const expert_read_plan_t& plan,
    uint8_t*                  staging_buffer,
    uint8_t*                  slot_buffer
);

} // namespace stream_moe
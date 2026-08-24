#include "io/staging_reader.h"
#include "common/logger.h"
#include <cstring>
#include <cassert>

namespace stream_moe {

expert_read_plan_t build_expert_read_plan(
    const sub_tensor_req_t* reqs,
    uint32_t num_tensors,
    size_t sector_size
) {
    expert_read_plan_t plan;
    if (num_tensors == 0 || num_tensors > MAX_SUB_TENSORS_PER_EXPERT) {
        LOG_ERROR("build_expert_read_plan: invalid num_tensors = " << num_tensors);
        return plan;
    }

    plan.num_tensors = num_tensors;
    size_t cur_staging_offset = 0;
    size_t max_slot_extent = 0;

    for (uint32_t i = 0; i < num_tensors; ++i) {
        const sub_tensor_req_t& req = reqs[i];
        tensor_slice_read_t& slice = plan.slices[i];

        slice.shard_idx = req.shard_idx;

        // Direct I/O sector alignment arithmetic
        uint64_t aligned_start = align_floor(req.file_offset, sector_size);
        uint64_t aligned_end   = align_ceil(req.file_offset + req.byte_size, sector_size);
        uint32_t aligned_len   = static_cast<uint32_t>(aligned_end - aligned_start);

        slice.file_read_start = aligned_start;
        slice.file_read_len   = aligned_len;
        slice.staging_offset  = cur_staging_offset;
        slice.copy_src_offset = cur_staging_offset + (req.file_offset - aligned_start);
        slice.copy_dst_offset = req.slot_offset;
        slice.copy_byte_len   = req.byte_size;

        cur_staging_offset += align_ceil(aligned_len, sector_size);
        size_t slot_extent = req.slot_offset + req.byte_size;
        if (slot_extent > max_slot_extent) {
            max_slot_extent = slot_extent;
        }
    }

    plan.total_staging_size = cur_staging_offset;
    plan.total_slot_size    = align_ceil(max_slot_extent, sector_size);

    return plan;
}

bool read_expert_sync(
    async_dio_engine* engine,
    const std::vector<dio_file_t*>& shard_files,
    const expert_read_plan_t& plan,
    uint8_t* staging_buffer,
    uint8_t* slot_buffer
) {
    if (!engine || shard_files.empty() || !staging_buffer || !slot_buffer || plan.num_tensors == 0) {
        return false;
    }

    aio_req_t reqs[MAX_SUB_TENSORS_PER_EXPERT];
    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        const tensor_slice_read_t& slice = plan.slices[i];
        reqs[i].file         = (slice.shard_idx < shard_files.size()) ? shard_files[slice.shard_idx] : shard_files[0];
        reqs[i].aligned_buf  = staging_buffer + slice.staging_offset;
        reqs[i].file_offset  = slice.file_read_start;
        reqs[i].aligned_len  = slice.file_read_len;
        reqs[i].user_data    = nullptr;
        reqs[i].is_completed = false;
        reqs[i].error_code   = 0;
    }

    if (!engine->submit_batch(reqs, plan.num_tensors)) {
        LOG_ERROR("read_expert_sync: submit_batch failed");
        return false;
    }

    aio_req_t* completed_reqs[MAX_SUB_TENSORS_PER_EXPERT];
    uint32_t completed = 0;
    while (completed < plan.num_tensors) {
        uint32_t n = engine->wait_events(
            completed_reqs + completed,
            plan.num_tensors - completed,
            plan.num_tensors - completed,
            5000
        );
        if (n == 0) {
            LOG_ERROR("read_expert_sync: timeout or incomplete batch, completed " 
                      << completed << " of " << plan.num_tensors);
            return false;
        }
        completed += n;
    }

    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        if (reqs[i].error_code != 0) {
            LOG_ERROR("read_expert_sync: req " << i << " failed with error " << reqs[i].error_code);
            return false;
        }
    }

    // Step 2: Slice payload and copy to compact slot buffer
    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        const tensor_slice_read_t& slice = plan.slices[i];
        std::memcpy(
            slot_buffer + slice.copy_dst_offset,
            staging_buffer + slice.copy_src_offset,
            slice.copy_byte_len
        );
    }

    return true;
}

bool read_expert_sync(
    async_dio_engine* engine,
    dio_file_t* file,
    const expert_read_plan_t& plan,
    uint8_t* staging_buffer,
    uint8_t* slot_buffer
) {
    std::vector<dio_file_t*> files = { file };
    return read_expert_sync(engine, files, plan, staging_buffer, slot_buffer);
}

} // namespace stream_moe
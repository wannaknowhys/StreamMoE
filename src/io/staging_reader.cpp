#include "io/staging_reader.h"
#include "common/logger.h"
#include <cassert>
#include <cstring>

namespace stream_moe {

expert_read_plan_t build_expert_read_plan(const sub_tensor_req_t* tensors, uint32_t num_tensors) {
    expert_read_plan_t plan;
    plan.num_tensors = std::min<uint32_t>(num_tensors, MAX_SUB_TENSORS);

    size_t cur_staging_offset = 0;
    size_t max_slot_end = 0;

    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        const auto& t = tensors[i];
        plan.tensors[i] = t;

        uint64_t start_aligned = align_floor(t.file_offset, DIO_SECTOR_SIZE);
        uint64_t end_aligned   = align_ceil(t.file_offset + t.byte_size, DIO_SECTOR_SIZE);
        uint32_t read_len      = static_cast<uint32_t>(end_aligned - start_aligned);
        uint32_t shift         = static_cast<uint32_t>(t.file_offset - start_aligned);

        plan.aligned_offsets[i] = start_aligned;
        plan.aligned_lens[i]    = read_len;
        plan.staging_offsets[i] = static_cast<uint32_t>(cur_staging_offset);
        plan.slice_shifts[i]    = shift;

        cur_staging_offset += align_ceil(read_len, DIO_SECTOR_SIZE);
        
        size_t slot_end = t.slot_offset + t.byte_size;
        if (slot_end > max_slot_end) {
            max_slot_end = slot_end;
        }
    }

    plan.total_staging_size = cur_staging_offset;
    plan.total_slot_size    = align_ceil(max_slot_end, DIO_SECTOR_SIZE);

    return plan;
}

bool read_expert_sync(
    async_dio_engine* engine,
    dio_file_t* file,
    const expert_read_plan_t& plan,
    uint8_t* staging_buf,
    uint8_t* dest_slot_ptr
) {
    if (!engine || !file || !staging_buf || !dest_slot_ptr || plan.num_tensors == 0) {
        return false;
    }

    aio_req_t reqs[MAX_SUB_TENSORS];
    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        reqs[i].file        = file;
        reqs[i].file_offset = plan.aligned_offsets[i];
        reqs[i].aligned_buf = staging_buf + plan.staging_offsets[i];
        reqs[i].aligned_len = plan.aligned_lens[i];
        reqs[i].user_data   = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
    }

    int32_t submitted = engine->submit_batch(reqs, plan.num_tensors);
    if (submitted != static_cast<int32_t>(plan.num_tensors)) {
        LOG_ERROR("read_expert_sync: submitted " << submitted << " / " << plan.num_tensors);
        return false;
    }

    aio_req_t* completed[MAX_SUB_TENSORS];
    uint32_t n_done = 0;
    while (n_done < plan.num_tensors) {
        uint32_t c = engine->wait_events(completed + n_done, plan.num_tensors - n_done, 1, 5000);
        if (c == 0) {
            LOG_ERROR("read_expert_sync: timeout waiting for completion");
            return false;
        }
        n_done += c;
    }

    // Check all requests for errors
    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        if (reqs[i].error_code != 0) {
            LOG_ERROR("read_expert_sync: req " << i << " failed with error " << reqs[i].error_code);
            return false;
        }
    }

    // Copy exact payloads from staging buffer into compact slot
    for (uint32_t i = 0; i < plan.num_tensors; ++i) {
        const uint8_t* src_payload = staging_buf + plan.staging_offsets[i] + plan.slice_shifts[i];
        uint8_t* dst_payload       = dest_slot_ptr + plan.tensors[i].slot_offset;
        std::memcpy(dst_payload, src_payload, plan.tensors[i].byte_size);
    }

    return true;
}

} // namespace stream_moe
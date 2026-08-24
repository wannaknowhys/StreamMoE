#include "loader/moe_loader.h"
#include "common/logger.h"

#include "gguf.h"

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <regex>
#include <cassert>

namespace stream_moe {

namespace {

int64_t get_kv_int(const gguf_context* ctx, const char* key, int64_t default_val = 0) {
    int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) return default_val;
    
    enum gguf_type type = gguf_get_kv_type(ctx, key_id);
    switch (type) {
        case GGUF_TYPE_UINT8:   return gguf_get_val_u8(ctx, key_id);
        case GGUF_TYPE_INT8:    return gguf_get_val_i8(ctx, key_id);
        case GGUF_TYPE_UINT16:  return gguf_get_val_u16(ctx, key_id);
        case GGUF_TYPE_INT16:   return gguf_get_val_i16(ctx, key_id);
        case GGUF_TYPE_UINT32:  return gguf_get_val_u32(ctx, key_id);
        case GGUF_TYPE_INT32:   return gguf_get_val_i32(ctx, key_id);
        case GGUF_TYPE_UINT64:  return static_cast<int64_t>(gguf_get_val_u64(ctx, key_id));
        case GGUF_TYPE_INT64:   return gguf_get_val_i64(ctx, key_id);
        default: return default_val;
    }
}

std::string get_kv_str(const gguf_context* ctx, const char* key, const std::string& default_val = "") {
    int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) return default_val;
    if (gguf_get_kv_type(ctx, key_id) == GGUF_TYPE_STRING) {
        return gguf_get_val_str(ctx, key_id);
    }
    return default_val;
}

// Check if filename matches split pattern: -00001-of-00005.gguf
std::vector<std::string> discover_shards(const std::string& main_path, const gguf_context* ctx0) {
    std::vector<std::string> shards;
    shards.push_back(main_path);

    int64_t split_count = get_kv_int(ctx0, "split.count", 0);
    
    std::regex split_regex("^(.*-)(\\d{5})-of-(\\d{5})(\\.gguf)$", std::regex::icase);
    std::smatch match;
    if (std::regex_match(main_path, match, split_regex)) {
        std::string prefix = match[1].str();
        int total_shards = std::stoi(match[3].str());
        std::string suffix = match[4].str();

        shards.clear();
        for (int i = 1; i <= total_shards; ++i) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%05d-of-%05d", i, total_shards);
            std::string shard_path = prefix + buf + suffix;
            if (std::filesystem::exists(shard_path)) {
                shards.push_back(shard_path);
            }
        }
    } else if (split_count > 1) {
        // Fallback for numbered shards if named differently
        std::filesystem::path p(main_path);
        std::string stem = p.stem().string();
        std::string ext = p.extension().string();
        std::string dir = p.parent_path().string();
        
        shards.clear();
        for (int i = 1; i <= split_count; ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s-%05d-of-%05d%s", stem.c_str(), i, static_cast<int>(split_count), ext.c_str());
            std::string shard_path = (dir.empty() ? "" : dir + "/") + buf;
            if (std::filesystem::exists(shard_path)) {
                shards.push_back(shard_path);
            }
        }
    }

    if (shards.empty()) {
        shards.push_back(main_path);
    }
    return shards;
}

} // namespace

moe_model_topology_t moe_loader::parse_gguf_topology(const std::string& main_gguf_path) {
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;

    struct gguf_context* ctx0 = gguf_init_from_file(main_gguf_path.c_str(), params);
    if (!ctx0) {
        throw std::runtime_error("Failed to open main GGUF file: " + main_gguf_path);
    }

    moe_model_topology_t topo;
    topo.arch_name = get_kv_str(ctx0, "general.architecture", "llama");

    std::string block_count_key  = topo.arch_name + ".block_count";
    std::string expert_count_key = topo.arch_name + ".expert_count";
    std::string expert_used_key  = topo.arch_name + ".expert_used_count";

    topo.n_layer       = static_cast<uint32_t>(get_kv_int(ctx0, block_count_key.c_str(), 0));
    topo.n_expert      = static_cast<uint32_t>(get_kv_int(ctx0, expert_count_key.c_str(), 0));
    topo.n_expert_used = static_cast<uint32_t>(get_kv_int(ctx0, expert_used_key.c_str(), 0));

    if (topo.n_layer == 0) topo.n_layer = static_cast<uint32_t>(get_kv_int(ctx0, "block_count", 0));
    if (topo.n_expert == 0) topo.n_expert = static_cast<uint32_t>(get_kv_int(ctx0, "expert_count", 0));

    topo.shard_paths = discover_shards(main_gguf_path, ctx0);
    gguf_free(ctx0);

    LOG_INFO("Parsed GGUF Topology: arch=" << topo.arch_name 
             << ", n_layer=" << topo.n_layer 
             << ", n_expert=" << topo.n_expert 
             << ", n_expert_used=" << topo.n_expert_used
             << ", shards=" << topo.shard_paths.size());

    if (topo.n_layer == 0) {
        throw std::runtime_error("Model has 0 layers or missing block_count metadata");
    }

    if (topo.n_expert == 0) {
        LOG_INFO("Model is purely Dense (0 experts).");
        return topo;
    }

    topo.experts.resize(static_cast<size_t>(topo.n_layer) * topo.n_expert);

    // Common MoE tensor suffixes
    const std::vector<std::string> sub_tensor_names = {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_up_exps.weight"
    };

    struct tensor_entry_t {
        std::string   name;
        uint32_t      shard_idx;
        uint64_t      abs_file_offset;
        size_t        total_size;
        enum ggml_type dtype;
        int64_t       ne[4];
    };

    // Collect all tensor metadata across all shards
    std::unordered_map<std::string, tensor_entry_t> global_tensors;

    for (uint32_t s = 0; s < topo.shard_paths.size(); ++s) {
        struct gguf_context* s_ctx = gguf_init_from_file(topo.shard_paths[s].c_str(), params);
        if (!s_ctx) {
            LOG_WARN("Could not open shard " << s << ": " << topo.shard_paths[s]);
            continue;
        }

        uint64_t data_offset = gguf_get_data_offset(s_ctx);
        int64_t n_t = gguf_get_n_tensors(s_ctx);

        for (int64_t i = 0; i < n_t; ++i) {
            const char* tname = gguf_get_tensor_name(s_ctx, i);
            tensor_entry_t entry;
            entry.name = tname;
            entry.shard_idx = s;
            entry.abs_file_offset = data_offset + gguf_get_tensor_offset(s_ctx, i);
            entry.total_size = gguf_get_tensor_size(s_ctx, i);
            entry.dtype = gguf_get_tensor_type(s_ctx, i);
            const int64_t* ne = gguf_get_tensor_ne(s_ctx, i);
            for (int d = 0; d < 4; ++d) entry.ne[d] = ne[d];

            global_tensors[entry.name] = entry;
        }

        gguf_free(s_ctx);
    }

    LOG_INFO("Total unique tensors found across all shards: " << global_tensors.size());

    std::vector<uint32_t> detected_moe_layers;
    size_t expected_expert_size = 0;
    uint32_t expected_num_sub_tensors = 0;
    bool baseline_established = false;

    for (uint32_t l = 0; l < topo.n_layer; ++l) {
        std::string layer_prefix = "blk." + std::to_string(l) + ".";
        
        std::vector<tensor_entry_t> layer_sub_tensors;
        for (const auto& suffix : sub_tensor_names) {
            std::string tname = layer_prefix + suffix;
            auto it = global_tensors.find(tname);
            if (it != global_tensors.end()) {
                layer_sub_tensors.push_back(it->second);
            }
        }

        if (layer_sub_tensors.empty()) {
            continue;
        }

        detected_moe_layers.push_back(l);

        for (uint32_t e = 0; e < topo.n_expert; ++e) {
            size_t expert_flat_idx = static_cast<size_t>(l) * topo.n_expert + e;
            expert_info_t& exp_info = topo.experts[expert_flat_idx];
            exp_info.layer_idx = static_cast<int32_t>(l);
            exp_info.expert_idx = static_cast<int32_t>(e);
            exp_info.total_expert_bytes = 0;
            exp_info.sub_tensors.clear();

            std::vector<sub_tensor_req_t> reqs;
            uint64_t cur_slot_offset = 0;

            for (const auto& tentry : layer_sub_tensors) {
                size_t slice_bytes = tentry.total_size / topo.n_expert;
                uint64_t slice_abs_offset = tentry.abs_file_offset + e * slice_bytes;

                sub_tensor_info_t st;
                st.name = tentry.name;
                st.shard_idx = tentry.shard_idx;
                st.abs_file_offset = slice_abs_offset;
                st.byte_size = slice_bytes;
                st.slot_offset = cur_slot_offset;
                st.ggml_type = static_cast<int32_t>(tentry.dtype);
                for (int d = 0; d < 4; ++d) st.ne[d] = tentry.ne[d];

                exp_info.sub_tensors.push_back(st);
                exp_info.total_expert_bytes += slice_bytes;

                                sub_tensor_req_t req;
                req.shard_idx   = tentry.shard_idx;
                req.file_offset = slice_abs_offset;
                req.byte_size   = slice_bytes;
                req.slot_offset = cur_slot_offset;
                reqs.push_back(req);

                cur_slot_offset += slice_bytes;
            }

            exp_info.read_plan = build_expert_read_plan(reqs.data(), static_cast<uint32_t>(reqs.size()));

            if (!baseline_established) {
                expected_expert_size = exp_info.total_expert_bytes;
                expected_num_sub_tensors = static_cast<uint32_t>(exp_info.sub_tensors.size());
                topo.expert_slot_size = exp_info.read_plan.total_slot_size;
                topo.expert_dio_staging_size = exp_info.read_plan.total_staging_size;
                topo.num_sub_tensors_per_expert = expected_num_sub_tensors;
                baseline_established = true;
            } else {
                if (exp_info.total_expert_bytes != expected_expert_size ||
                    exp_info.sub_tensors.size() != expected_num_sub_tensors) {
                    std::ostringstream err;
                    err << "Heterogeneous MoE model detected! Layer " << l << " Expert " << e 
                        << " has size " << exp_info.total_expert_bytes << " bytes, expected " << expected_expert_size;
                    throw std::runtime_error(err.str());
                }
            }
        }
    }

    topo.moe_layers = detected_moe_layers;

    LOG_INFO("Homogeneous MoE Validation PASSED! " << topo.moe_layers.size() << " MoE layers, "
             << topo.n_expert << " experts/layer, slot_size=" << (topo.expert_slot_size / 1024) << " KB, "
             << "staging_size=" << (topo.expert_dio_staging_size / 1024) << " KB");

    return topo;
}

} // namespace stream_moe
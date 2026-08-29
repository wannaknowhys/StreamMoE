#include "loader/moe_loader.h"
#include "common/logger.h"

#include "gguf.h"

#include <algorithm>
#include <cstring>
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

        if (split_count > 0 && static_cast<int>(split_count) != total_shards) {
            throw std::runtime_error(
                "Shard count mismatch: filename says " + std::to_string(total_shards)
                + " but GGUF metadata split.count=" + std::to_string(split_count));
        }

        shards.clear();
        std::vector<std::string> missing;
        for (int i = 1; i <= total_shards; ++i) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%05d-of-%05d", i, total_shards);
            std::string shard_path = prefix + buf + suffix;
            if (std::filesystem::exists(shard_path)) {
                shards.push_back(shard_path);
            } else {
                missing.push_back(shard_path);
            }
        }
        if (!missing.empty()) {
            // A partial shard set silently produces a broken model - fail hard
            throw std::runtime_error(
                "Incomplete GGUF shard set: missing " + std::to_string(missing.size())
                + " of " + std::to_string(total_shards) + " shards, first missing: " + missing.front());
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
            } else {
                throw std::runtime_error("Incomplete GGUF shard set: missing shard " + std::to_string(i)
                                         + "/" + std::to_string(split_count) + ": " + shard_path);
            }
        }
    }

    if (shards.empty()) {
        shards.push_back(main_path);
    }
    return shards;
}

// v2 (expert-blocks-v2): experts are compact blocks described by stream_moe.* KV
// (expert_sections [off,size,nsub] per block, branch_names + per-layer
// branch_sizes for the block-internal layout). Builds the same
// expert_info_t / read_plan shape as v1 so scheduler + delegate are untouched.
static void build_v2_experts(struct gguf_context* ctx0, moe_model_topology_t& topo) {
    auto get_arr_u64 = [&](const char* key, std::vector<uint64_t>& out) {
        int64_t id = gguf_find_key(ctx0, key);
        if (id < 0 || gguf_get_kv_type(ctx0, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx0, id) != GGUF_TYPE_UINT64) {
            throw std::runtime_error(std::string("v2: missing/invalid ") + key);
        }
        const uint64_t* d = static_cast<const uint64_t*>(gguf_get_arr_data(ctx0, id));
        out.assign(d, d + gguf_get_arr_n(ctx0, id));
    };
    std::vector<uint64_t> sections, bsizes;
    get_arr_u64("stream_moe.expert_sections", sections);
    get_arr_u64("stream_moe.expert_branch_sizes", bsizes);
    const int64_t bnid = gguf_find_key(ctx0, "stream_moe.expert_branch_names");
    if (bnid < 0) throw std::runtime_error("v2: missing expert_branch_names");
    // branch_names are flattened per-layer full tensor names, sliced by the
    // per-layer branch counts (non-uniform MoE layers supported).
    const size_t n_branch_total = gguf_get_arr_n(ctx0, bnid);
    std::vector<uint64_t> bcounts;
    get_arr_u64("stream_moe.expert_branch_counts", bcounts);
    if (bcounts.size() != static_cast<size_t>(topo.n_layer)) {
        throw std::runtime_error("v2: expert_branch_counts length mismatch");
    }
    std::vector<std::string> bnames;
    bnames.reserve(n_branch_total);
    for (size_t i = 0; i < n_branch_total; ++i) bnames.push_back(gguf_get_arr_str(ctx0, bnid, i));

    const size_t n_blocks = sections.size() / 3;
    if (n_blocks != static_cast<size_t>(topo.n_layer) * topo.n_expert) {
        throw std::runtime_error("v2: expert_sections count mismatch");
    }
    const uint64_t data_start = gguf_get_data_offset(ctx0);

    topo.experts.resize(static_cast<size_t>(topo.n_layer) * topo.n_expert);

    size_t branch_off = 0;
    for (uint32_t l = 0; l < topo.n_layer; ++l) {
        const size_t n_branch = static_cast<size_t>(bcounts[l]);
        const size_t layer_bsize = sections[(static_cast<size_t>(l) * topo.n_expert) * 3 + 1];
        for (uint32_t e = 0; e < topo.n_expert; ++e) {
            const size_t flat = static_cast<size_t>(l) * topo.n_expert + e;
            const uint64_t boff  = sections[flat * 3];
            const uint64_t bsize = sections[flat * 3 + 1];
            expert_info_t& exp = topo.experts[flat];
            exp.layer_idx = static_cast<int32_t>(l);
            exp.expert_idx = static_cast<int32_t>(e);
            exp.total_expert_bytes = 0;
            exp.sub_tensors.clear();
            std::vector<sub_tensor_req_t> reqs;
            uint64_t cur = 0;
            for (size_t j = 0; j < n_branch; ++j) {
                const uint64_t bsz = bsizes[branch_off + j];
                sub_tensor_info_t st;
                st.name = bnames[branch_off + j]; // full tensor name
                st.shard_idx = 0; // v2 is a single file
                st.abs_file_offset = data_start + boff + cur;
                st.byte_size = bsz;
                st.slot_offset = cur;
                st.ggml_type = 0; // F32 placeholder; staging is keyed on byte size
                st.ne[0] = 0;
                exp.sub_tensors.push_back(st);
                sub_tensor_req_t req;
                req.shard_idx = 0;
                req.file_offset = data_start + boff + cur;
                req.byte_size = bsz;
                req.slot_offset = cur;
                reqs.push_back(req);
                cur += bsz;
                exp.total_expert_bytes += bsz;
            }
            exp.read_plan = build_expert_read_plan(reqs.data(), static_cast<uint32_t>(reqs.size()));
        }
        branch_off += n_branch;
        // sub-pool group by layer block size
        auto fit = std::find_if(topo.groups.begin(), topo.groups.end(),
                                [&](const auto& g) { return g.expert_size == layer_bsize; });
        uint32_t gidx;
        if (fit != topo.groups.end()) {
            gidx = fit->idx;
        } else {
            gidx = static_cast<uint32_t>(topo.groups.size());
            moe_model_topology_t::expert_group_t g;
            g.idx = gidx;
            g.expert_size = layer_bsize;
            topo.groups.push_back(g);
        }
        if (topo.groups[gidx].layers.empty() || topo.groups[gidx].layers.back() != l) {
            topo.groups[gidx].layers.push_back(l);
        }
    }
    topo.moe_layers.clear();
    for (uint32_t l = 0; l < topo.n_layer; ++l) topo.moe_layers.push_back(l);
    for (auto& g : topo.groups) {
        g.total_bytes = static_cast<uint64_t>(g.layers.size()) * topo.n_expert * g.expert_size;
    }
    if (!topo.experts.empty()) {
        topo.expert_slot_size = topo.experts[0].read_plan.total_slot_size;
        topo.expert_dio_staging_size = topo.experts[0].read_plan.total_staging_size;
        topo.num_sub_tensors_per_expert = static_cast<uint32_t>(topo.experts[0].sub_tensors.size());
    }
    LOG_INFO("v2 expert-blocks: layers=" << topo.n_layer << " experts/layer=" << topo.n_expert
             << " groups=" << topo.groups.size());
    for (const auto& g : topo.groups) {
        LOG_INFO("  group " << g.idx << ": layers=" << g.layers.size()
                 << " block_size=" << (g.expert_size / 1024) << "KB total=" << (g.total_bytes / (1024ull * 1024ull)) << "MB");
    }
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

    // Dynamically extract Context and Attention dimensions
    std::string ctx_len_key   = topo.arch_name + ".context_length";
    std::string embd_key      = topo.arch_name + ".embedding_length";
    std::string head_key      = topo.arch_name + ".attention.head_count";
    std::string head_kv_key   = topo.arch_name + ".attention.head_count_kv";
    std::string head_dim_key  = topo.arch_name + ".attention.key_length";

        topo.max_context_length = static_cast<uint32_t>(get_kv_int(ctx0, ctx_len_key.c_str(), 4096));
    topo.embedding_length   = static_cast<uint32_t>(get_kv_int(ctx0, embd_key.c_str(), 2048));
    topo.head_count         = static_cast<uint32_t>(get_kv_int(ctx0, head_key.c_str(), 32));
    topo.head_count_kv      = static_cast<uint32_t>(get_kv_int(ctx0, head_kv_key.c_str(), topo.head_count));
    topo.head_dim           = static_cast<uint32_t>(get_kv_int(ctx0, head_dim_key.c_str(), topo.head_count > 0 ? (topo.embedding_length / topo.head_count) : 64));

    // Multi-Head Latent Attention (MLA) detection (DeepSeek V2/V3/V4, Kimi-K3, etc.)
    std::string kv_lora_key   = topo.arch_name + ".attention.kv_lora_rank";
    std::string q_lora_key    = topo.arch_name + ".attention.q_lora_rank";
    std::string rope_dim_key  = topo.arch_name + ".rope.dimension_count";
    std::string key_len_key   = topo.arch_name + ".attention.key_length";

    topo.kv_lora_rank = static_cast<uint32_t>(get_kv_int(ctx0, kv_lora_key.c_str(), 0));
    topo.q_lora_rank  = static_cast<uint32_t>(get_kv_int(ctx0, q_lora_key.c_str(), 0));
    topo.qk_rope_dim  = static_cast<uint32_t>(get_kv_int(ctx0, rope_dim_key.c_str(), 64));

    // If key_length is 512 and head_count_kv == 1 (e.g. DeepSeek-V4), it uses MLA latent representation
    uint32_t key_len = static_cast<uint32_t>(get_kv_int(ctx0, key_len_key.c_str(), 0));
    if (topo.kv_lora_rank > 0 || (topo.head_count_kv == 1 && key_len == 512) || topo.arch_name.find("deepseek") != std::string::npos) {
        topo.is_mla = true;
        if (topo.kv_lora_rank == 0) {
            topo.kv_lora_rank = (key_len > 0) ? key_len : 512;
        }
    }    if (topo.n_layer == 0) topo.n_layer = static_cast<uint32_t>(get_kv_int(ctx0, "block_count", 0));
    if (topo.n_expert == 0) topo.n_expert = static_cast<uint32_t>(get_kv_int(ctx0, "expert_count", 0));topo.shard_paths = discover_shards(main_gguf_path, ctx0);

    // v2 (expert-blocks-v2): experts are compact blocks described by stream_moe.* KV.
    // Build the topology from KV and bypass the v1 per-tensor-slice path.
    {
        const std::string layout = get_kv_str(ctx0, "stream_moe.layout", "");
        if (layout == "expert-blocks-v2") {
            topo.layout = gguf_layout_t::V2_EXPERT_BLOCKS;
            build_v2_experts(ctx0, topo);
            gguf_free(ctx0);
            return topo;
        }
        if (layout == "sections-v1") topo.layout = gguf_layout_t::V1_SECTIONS;
    }

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
            // Fail hard: continuing with a partial shard set silently corrupts the model
            throw std::runtime_error("Failed to open GGUF shard: " + topo.shard_paths[s]);
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

    // Dense vs expert accounting at parse time: `_exps` tensors are the routed
    // experts (pooled); everything else stays on llama.cpp defaults (mmap).
    for (const auto& [name, e] : global_tensors) {
        const bool is_expert = std::strstr(name.c_str(), "_exps") != nullptr;
        if (is_expert) {
            topo.expert_total_bytes += e.total_size;
        } else {
            topo.dense_tensor_names.push_back(name);
            topo.dense_total_bytes += e.total_size;
        }
    }
    LOG_INFO("Weight breakdown: dense=" << (topo.dense_total_bytes / (1024.0 * 1024.0 * 1024.0))
             << " GB, expert=" << (topo.expert_total_bytes / (1024.0 * 1024.0 * 1024.0))
             << " GB (" << topo.dense_tensor_names.size() << " dense tensors)");

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
                // Validate the homogeneous equal-slice assumption before using it:
                // 1) tensor must divide evenly across experts
                if (topo.n_expert == 0 || tentry.total_size % topo.n_expert != 0) {
                    std::ostringstream err;
                    err << "Expert tensor '" << tentry.name << "' size " << tentry.total_size
                        << " is not divisible by expert count " << topo.n_expert;
                    throw std::runtime_error(err.str());
                }
                // 2) 3D exps tensors declare n_expert as ne[2] - cross-check against metadata
                if (tentry.ne[2] > 0 && static_cast<uint32_t>(tentry.ne[2]) != topo.n_expert) {
                    std::ostringstream err;
                    err << "Expert tensor '" << tentry.name << "' declares ne[2]=" << tentry.ne[2]
                        << " but metadata expert_count=" << topo.n_expert
                        << " - per-expert slicing offset math would be invalid";
                    throw std::runtime_error(err.str());
                }
                // 3) slice must be a whole multiple of the quantization block size
                uint32_t blck = static_cast<uint32_t>(ggml_blck_size(tentry.dtype));
                if (blck == 0) blck = 1;
                size_t slice_bytes_probe = tentry.total_size / topo.n_expert;
                if (slice_bytes_probe % blck != 0) {
                    std::ostringstream err;
                    err << "Expert slice of '" << tentry.name << "' (" << slice_bytes_probe
                        << " bytes) is not aligned to quantization block size " << blck;
                    throw std::runtime_error(err.str());
                }

                size_t slice_bytes = slice_bytes_probe;
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
            }

            // Per-expert group assignment (docs/MULTI_SUBPOOL.md): experts with
            // the same per-expert byte size share a sub-pool. Heterogeneous
            // layers simply land in their own group instead of being excluded.
            {
                auto fit = std::find_if(topo.groups.begin(), topo.groups.end(),
                                        [&](const auto& g) { return g.expert_size == exp_info.total_expert_bytes; });
                uint32_t gidx;
                if (fit != topo.groups.end()) {
                    gidx = fit->idx;
                } else {
                    gidx = static_cast<uint32_t>(topo.groups.size());
                    moe_model_topology_t::expert_group_t g;
                    g.idx = gidx;
                    g.expert_size = exp_info.total_expert_bytes;
                    topo.groups.push_back(g);
                }
                if (topo.groups[gidx].layers.empty() || topo.groups[gidx].layers.back() != static_cast<uint32_t>(l)) {
                    topo.groups[gidx].layers.push_back(static_cast<uint32_t>(l));
                }
            }
        }
    }

    topo.moe_layers = detected_moe_layers;

    // Finalize group byte accounting.
    for (auto& g : topo.groups) {
        g.total_bytes = static_cast<uint64_t>(g.layers.size()) * topo.n_expert * g.expert_size;
    }

    LOG_INFO("Expert groups: " << topo.groups.size() << " ("
             << topo.moe_layers.size() << " MoE layers, " << topo.n_expert
             << " experts/layer, baseline slot_size=" << (topo.expert_slot_size / 1024) << " KB, "
             << "staging_size=" << (topo.expert_dio_staging_size / 1024) << " KB)");
    for (const auto& g : topo.groups) {
        LOG_INFO("  group " << g.idx << ": layers=" << g.layers.size()
                 << " expert_size=" << (g.expert_size / 1024) << "KB total=" << (g.total_bytes / (1024ull*1024ull)) << "MB");
    }

    return topo;
}

} // namespace stream_moe
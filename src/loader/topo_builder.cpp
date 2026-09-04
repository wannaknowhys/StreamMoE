#include "loader/topo_builder.h"
#include "common/logger.h"

#include "gguf.h"

#include <algorithm>
#include <cstring>

namespace stream_moe {

namespace {

int64_t kv_int(const gguf_context* ctx, const char* key, int64_t def = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, id));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default: return def;
    }
}

std::string kv_str(const gguf_context* ctx, const char* key, const std::string& def = "") {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(ctx, id);
}

} // namespace

moe_model_topology_t build_topology(const model_t& m, const std::string& main_gguf_path) {
    moe_model_topology_t topo;
    topo.arch_name = m.arch;
    topo.n_layer = m.n_layer;
    topo.n_expert = m.n_expert;
    topo.n_expert_used = m.n_expert_used;
    topo.shard_paths = m.files;

    switch (m.layout) {
        case model_layout_t::V1_SECTIONS:      topo.layout = gguf_layout_t::V1_SECTIONS; break;
        case model_layout_t::V2_EXPERT_BLOCKS:
        case model_layout_t::V2_CHUNK:         topo.layout = gguf_layout_t::V2_EXPERT_BLOCKS; break;
        default:                               topo.layout = gguf_layout_t::ORIGINAL; break;
    }
    topo.incomplete = m.incomplete; // v2 chunk: dense tensors need route_b takeover

    // Attention / MLA metadata is not carried by model_t - read it from the
    // source GGUF KV directly (same defaults as moe_loader before).
    {
        gguf_init_params params = { true, nullptr };
        gguf_context* ctx0 = gguf_init_from_file(main_gguf_path.c_str(), params);
        if (!ctx0) throw std::runtime_error("cannot open " + main_gguf_path);
        const std::string arch = topo.arch_name;
        topo.max_context_length = static_cast<uint32_t>(kv_int(ctx0, (arch + ".context_length").c_str(), 4096));
        topo.embedding_length   = static_cast<uint32_t>(kv_int(ctx0, (arch + ".embedding_length").c_str(), 2048));
        topo.head_count         = static_cast<uint32_t>(kv_int(ctx0, (arch + ".attention.head_count").c_str(), 32));
        topo.head_count_kv      = static_cast<uint32_t>(kv_int(ctx0, (arch + ".attention.head_count_kv").c_str(), topo.head_count));
        topo.head_dim           = static_cast<uint32_t>(kv_int(ctx0, (arch + ".attention.key_length").c_str(),
                                                          topo.head_count > 0 ? (topo.embedding_length / topo.head_count) : 64));
        topo.kv_lora_rank = static_cast<uint32_t>(kv_int(ctx0, (arch + ".attention.kv_lora_rank").c_str(), 0));
        topo.q_lora_rank  = static_cast<uint32_t>(kv_int(ctx0, (arch + ".attention.q_lora_rank").c_str(), 0));
        topo.qk_rope_dim  = static_cast<uint32_t>(kv_int(ctx0, (arch + ".rope.dimension_count").c_str(), 64));
        const uint32_t key_len = static_cast<uint32_t>(kv_int(ctx0, (arch + ".attention.key_length").c_str(), 0));
        if (topo.kv_lora_rank > 0 || (topo.head_count_kv == 1 && key_len == 512) ||
            arch.find("deepseek") != std::string::npos) {
            topo.is_mla = true;
            if (topo.kv_lora_rank == 0) topo.kv_lora_rank = key_len > 0 ? key_len : 512;
        }
        gguf_free(ctx0);
    }

    for (const auto& d : m.dense) {
        topo.dense_tensor_names.push_back(d.name);
        topo.dense_total_bytes += d.size;
    }
    for (const auto& et : m.expert) topo.expert_total_bytes += et.size;

    LOG_INFO("Parsed Model: arch=" << topo.arch_name << " layout=" << (int) topo.layout
             << " n_layer=" << topo.n_layer << " n_expert=" << topo.n_expert
             << " shards=" << topo.shard_paths.size());
    LOG_INFO("Weight breakdown: dense=" << (topo.dense_total_bytes / (1024.0 * 1024.0 * 1024.0))
             << " GB, expert=" << (topo.expert_total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB");

    if (topo.n_layer == 0) throw std::runtime_error("model has 0 layers");
    if (topo.n_expert == 0) {
        LOG_INFO("model is purely dense (0 experts)");
        return topo;
    }

    topo.experts.resize((size_t) topo.n_layer * topo.n_expert);

    for (uint32_t l = 0; l < topo.n_layer; ++l) {
        std::vector<const expert_tensor_t*> branches;
        for (const auto& et : m.expert) if (et.layer == (int32_t) l) branches.push_back(&et);
        if (branches.empty()) continue;
        topo.moe_layers.push_back(l);

        // sub-pool group key: an expert's resident bytes = sum of its compact
        // per-branch slices (SoA: pool stores per-column compact data, not the
        // whole v2 block with its inter-branch 4K padding).
        uint64_t layer_expert_size = 0;
        for (const auto* b : branches) layer_expert_size += b->per_expert;

        for (uint32_t e = 0; e < topo.n_expert; ++e) {
            auto& exp = topo.experts[(size_t) l * topo.n_expert + e];
            exp.layer_idx = (int32_t) l;
            exp.expert_idx = (int32_t) e;
            exp.sub_tensors.clear();
            std::vector<sub_tensor_req_t> reqs;
            // SoA column ordinal == branch position within the layer's branch
            // list (ORDER); homogeneous groups make it equal the group column
            // index. Branches push sub_tensors in the same ORDER.
            uint32_t col_ord = 0;
            for (const auto* b : branches) {
                uint64_t col_start = 0;   // offset of this branch's slice inside its column (0)
                for (const auto& s : b->per_expert_srcs[e]) {
                    sub_tensor_info_t st;
                    st.name = b->name;
                    st.shard_idx = s.file;
                    st.abs_file_offset = s.off;
                    st.byte_size = s.len;
                    st.slot_offset = col_start;   // offset inside the column slice
                    st.ggml_type = b->type;
                    for (int d = 0; d < 4; ++d) st.ne[d] = b->ne[d];
                    exp.sub_tensors.push_back(st);
                    sub_tensor_req_t req;
                    req.shard_idx = s.file;
                    req.file_offset = s.off;
                    req.byte_size = s.len;
                    req.column = col_ord;
                    req.col_off = col_start;
                    reqs.push_back(req);
                    col_start += s.len;
                }
                ++col_ord;
            }
            exp.total_expert_bytes = 0;
            for (const auto& b : branches) exp.total_expert_bytes += b->per_expert;

            exp.read_plan = build_expert_read_plan(reqs.data(), static_cast<uint32_t>(reqs.size()));
        }

        // sub-pool group by layer expert size (docs/MULTI_SUBPOOL.md)
        auto fit = std::find_if(topo.groups.begin(), topo.groups.end(),
                                [&](const auto& g) { return g.expert_size == layer_expert_size; });
        uint32_t gidx;
        if (fit != topo.groups.end()) {
            gidx = fit->idx;
        } else {
            gidx = static_cast<uint32_t>(topo.groups.size());
            moe_model_topology_t::expert_group_t g;
            g.idx = gidx;
            g.expert_size = layer_expert_size;
            topo.groups.push_back(g);
        }
        if (topo.groups[gidx].layers.empty() || topo.groups[gidx].layers.back() != l) {
            topo.groups[gidx].layers.push_back(l);
        }
    }

    for (auto& g : topo.groups) {
        g.total_bytes = static_cast<uint64_t>(g.layers.size()) * topo.n_expert * g.expert_size;
    }

    // SoA columns per group (docs/MULTI_SUBPOOL.md §1a): each expert tensor that
    // the group's layers share becomes one column (stride = its perExpert).
    // All experts of a layer have identical sub_tensor layout, so inspect the
    // first expert of the group's first layer. Columns are (tag,type,perExpert)
    // merged across the group's layers where the layout matches.
    for (auto& g : topo.groups) {
        if (g.layers.empty()) continue;
        const expert_info_t& probe = topo.experts[(size_t) g.layers[0] * topo.n_expert];
        auto& cols = g.columns;
        cols.clear();
        for (const auto& st : probe.sub_tensors) {
            const std::string tag = [&]() -> std::string {
                if (st.name.find("gate_up") != std::string::npos) return "gate_up";
                if (st.name.find("ffn_gate_exps") != std::string::npos) return "gate";
                if (st.name.find("ffn_up_exps") != std::string::npos) return "up";
                if (st.name.find("down_exps") != std::string::npos) return "down";
                return "?";
            }();
            // merge with an existing column when the slice layout matches
            auto it = std::find_if(cols.begin(), cols.end(), [&](const auto& c) {
                return c.tag == tag && c.ggml_type == st.ggml_type && c.per_expert == st.byte_size;
            });
            if (it != cols.end()) continue;
            moe_model_topology_t::expert_group_t::column_t c;
            c.col_index = static_cast<uint32_t>(cols.size());
            c.name = st.name;
            c.tag = tag;
            c.ggml_type = st.ggml_type;
            for (int d = 0; d < 4; ++d) c.ne[d] = st.ne[d];
            c.per_expert = st.byte_size;
            cols.push_back(c);
        }
    }

    LOG_INFO("Expert groups: " << topo.groups.size() << " ("
             << topo.moe_layers.size() << " MoE layers, " << topo.n_expert << " experts/layer)");
    for (const auto& g : topo.groups) {
        LOG_INFO("  group " << g.idx << ": layers=" << g.layers.size()
                 << " expert_size=" << (g.expert_size / 1024) << "KB total="
                 << (g.total_bytes / (1024ull * 1024ull)) << "MB"
                 << " columns=" << g.columns.size());
        for (const auto& c : g.columns) {
            LOG_INFO("    col " << c.col_index << " " << c.tag << " type=" << c.ggml_type
                     << " perExpert=" << (c.per_expert / 1024) << "KB"
                     << (c.per_expert_4k() ? " [direct]" : " [staging]"));
        }
    }

    return topo;
}

} // namespace stream_moe

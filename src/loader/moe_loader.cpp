#include "loader/moe_loader.h"
#include "loader/model_builder.h"
#include "loader/topo_builder.h"
#include "common/logger.h"

namespace stream_moe {

moe_model_topology_t moe_loader::parse_gguf_topology(const std::string& main_gguf_path) {
    // Single source of truth: parse_model (layout knowledge in model_builder /
    // model.h, mirroring stream_moe_layout.js) -> build_topology (read policy:
    // v2 whole-block straight DIO, v1 4K-aligned slices, original staging).
    const model_t m = parse_model_path(main_gguf_path);
    return build_topology(m, main_gguf_path);
}

} // namespace stream_moe

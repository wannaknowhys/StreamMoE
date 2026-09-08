#pragma once

#include "loader/model.h"

#include <string>
#include <vector>

namespace stream_moe {

// Parse GGUF source file(s) into a uniform model_t (read direction, shared
// with the C++ converter). Layout knowledge for original / v2 / v2chunk / v3
// lives here; everything downstream (DIO plan, slot placement, staging
// decision) consumes only model_t.
//
//   parse_model_path(main)       - single main path; original shards
//                                  (-00001-of-N) auto-discovered
//   parse_model(paths)           - explicit list (chunk strip files)
model_t parse_model(const std::vector<std::string>& paths);
model_t parse_model_path(const std::string& main_path);

} // namespace stream_moe

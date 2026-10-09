#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "../ggml/src/ggml-impl.h"
#ifdef STREAM_MOE_PREFILL_EXPORT
#include "tsc_timer.h"
#endif
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <filesystem>
#ifdef STREAM_MOE_PREFILL_EXPORT
// STREAM_MOE_TEMP: graph-eval-callback export (defined near the export section)
#ifdef STREAM_MOE_TEMP
namespace stream_moe { void route_b_dump_node_bin(int layer, const char *, const char *, int, int64_t, int64_t, const void *, size_t); void route_b_begin_ubatch(); }
#endif
static bool moe_export_eval_cb(ggml_tensor * t, bool ask, void * ud);
#endif

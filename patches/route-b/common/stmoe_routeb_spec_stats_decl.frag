#include <cstdio>
#include <vector>

// StreamMoE: speculative decoding statistics (per draft position).
// Filled by the speculative engine during process()/accept(); callers (CLI /
// server) print these after a session.
size_t                 common_speculative_n_gen_tokens(const common_speculative * spec);
size_t                 common_speculative_n_acc_tokens(const common_speculative * spec);
size_t                 common_speculative_n_acc_drafts(const common_speculative * spec);
const std::vector<size_t> & common_speculative_n_acc_tokens_per_pos(const common_speculative * spec);
void                   common_speculative_print_stats(const common_speculative * spec, FILE * out);

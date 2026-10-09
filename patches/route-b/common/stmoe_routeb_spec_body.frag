size_t common_speculative_n_gen_tokens(const common_speculative * spec) {
    size_t t = 0;
    for (const auto & impl : spec->impls) t += impl->n_gen_tokens;
    return t;
}

size_t common_speculative_n_acc_tokens(const common_speculative * spec) {
    size_t t = 0;
    for (const auto & impl : spec->impls) t += impl->n_acc_tokens;
    return t;
}

size_t common_speculative_n_acc_drafts(const common_speculative * spec) {
    size_t t = 0;
    for (const auto & impl : spec->impls) t += impl->n_acc_drafts;
    return t;
}

const std::vector<size_t> & common_speculative_n_acc_tokens_per_pos(const common_speculative * spec) {
    static thread_local std::vector<size_t> merged;
    merged.clear();
    size_t n = 0;
    for (const auto & impl : spec->impls) n = std::max(n, impl->n_acc_tokens_per_pos.size());
    merged.resize(n, 0);
    for (const auto & impl : spec->impls) {
        for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
            merged[i] += impl->n_acc_tokens_per_pos[i];
        }
    }
    return merged;
}

void common_speculative_print_stats(const common_speculative * spec, FILE * out) {
    const size_t n_gen    = common_speculative_n_gen_tokens(spec);
    const size_t n_acc    = common_speculative_n_acc_tokens(spec);
    const size_t n_drafts = common_speculative_n_acc_drafts(spec);
    const auto & per_pos  = common_speculative_n_acc_tokens_per_pos(spec);
    std::fprintf(out, "[spec] draft tokens = %zu, accepted = %zu (%.2f%%), draft runs = %zu\n",
                 n_gen, n_acc, n_gen ? 100.0 * n_acc / n_gen : 0.0, n_drafts);
    if (!per_pos.empty()) {
        std::fprintf(out, "[spec] accepted per pos = [");
        for (size_t i = 0; i < per_pos.size(); ++i) {
            std::fprintf(out, "%s%zu", i ? ", " : "", per_pos[i]);
        }
        std::fprintf(out, "]\n");
    }
}

#include "pool/expert_stats.h"
#include "common/logger.h"

#include <fstream>
#include <filesystem>
#include <cmath>

namespace stream_moe {

expert_stats_tracker::~expert_stats_tracker() {
    flush();
}

bool expert_stats_tracker::init(const std::string& file_path, uint32_t n_layer, uint32_t n_expert, uint32_t sync_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_path_      = file_path;
    n_layer_        = n_layer;
    n_expert_       = n_expert;
    sync_threshold_ = sync_threshold;
    tokens_since_last_sync_ = 0;
    dirty_          = false;

    size_t total_elements = static_cast<size_t>(n_layer_) * n_expert_;
    global_counts_.assign(total_elements, 0);
    adaptive_scores_.assign(total_elements, 0.0);

    if (total_elements == 0) {
        LOG_ERROR("expert_stats_tracker::init: n_layer or n_expert is 0");
        return false;
    }

    // Try loading existing file
    bool loaded = false;
    if (!file_path_.empty() && std::filesystem::exists(file_path_)) {
        std::ifstream in(file_path_, std::ios::binary);
        if (in.is_open()) {
            uint32_t magic = 0;
            uint32_t file_n_layer = 0;
            uint32_t file_n_expert = 0;

            in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            in.read(reinterpret_cast<char*>(&file_n_layer), sizeof(file_n_layer));
            in.read(reinterpret_cast<char*>(&file_n_expert), sizeof(file_n_expert));

            if (magic == EST1_MAGIC && file_n_layer == n_layer_ && file_n_expert == n_expert_) {
                in.read(reinterpret_cast<char*>(global_counts_.data()), total_elements * sizeof(uint64_t));
                if (in.good() || in.gcount() == static_cast<std::streamsize>(total_elements * sizeof(uint64_t))) {
                    loaded = true;
                    LOG_INFO("Loaded historical expert stats from " << file_path_ << " (" << n_layer_ << " layers, " << n_expert_ << " experts)");
                }
            } else {
                LOG_WARN("Existing EST1 stats mismatch topology (file: " << file_n_layer << "x" << file_n_expert 
                         << ", model: " << n_layer_ << "x" << n_expert_ << "). Reinitializing stats.");
            }
            in.close();
        }
    }

    // If file did not exist or mismatched, create/save initial zero file
    if (!loaded) {
        LOG_INFO("Initializing new expert stats file at " << file_path_);
        dirty_ = true;
        
        // Make parent directory if needed
        if (!file_path_.empty()) {
            std::filesystem::path p(file_path_);
            if (p.has_parent_path()) {
                std::error_code ec;
                std::filesystem::create_directories(p.parent_path(), ec);
            }
        }
        
        // Write initial header & zero counts
        std::ofstream out(file_path_, std::ios::binary);
        if (out.is_open()) {
            uint32_t magic = EST1_MAGIC;
            out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            out.write(reinterpret_cast<const char*>(&n_layer_), sizeof(n_layer_));
            out.write(reinterpret_cast<const char*>(&n_expert_), sizeof(n_expert_));
            out.write(reinterpret_cast<const char*>(global_counts_.data()), total_elements * sizeof(uint64_t));
            out.close();
            dirty_ = false;
        }
    }

    // Initialize adaptive scores with normalized global counts
    uint64_t max_count = 1;
    for (uint64_t cnt : global_counts_) {
        if (cnt > max_count) max_count = cnt;
    }
    for (size_t i = 0; i < total_elements; ++i) {
        adaptive_scores_[i] = static_cast<double>(global_counts_[i]) / static_cast<double>(max_count);
    }

    return true;
}

void expert_stats_tracker::record_access(uint32_t layer_idx, uint32_t expert_idx) {
    if (layer_idx >= n_layer_ || expert_idx >= n_expert_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    size_t idx = index(layer_idx, expert_idx);
    global_counts_[idx]++;
    adaptive_scores_[idx] += 1.0;
    dirty_ = true;
}

void expert_stats_tracker::apply_decay(double factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (double& score : adaptive_scores_) {
        score *= factor;
    }
}

void expert_stats_tracker::notify_tokens_generated(uint32_t num_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    tokens_since_last_sync_ += num_tokens;
    total_tokens_           += num_tokens;

    // Decay adaptive scores per generated token batch
    double decay_factor = std::pow(0.999, static_cast<double>(num_tokens));
    for (double& score : adaptive_scores_) {
        score *= decay_factor;
    }

    // Check 8192-token persistence threshold
    if (tokens_since_last_sync_ >= sync_threshold_ && dirty_) {
        if (!file_path_.empty()) {
            std::ofstream out(file_path_, std::ios::binary);
            if (out.is_open()) {
                uint32_t magic = EST1_MAGIC;
                out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                out.write(reinterpret_cast<const char*>(&n_layer_), sizeof(n_layer_));
                out.write(reinterpret_cast<const char*>(&n_expert_), sizeof(n_expert_));
                out.write(reinterpret_cast<const char*>(global_counts_.data()), global_counts_.size() * sizeof(uint64_t));
                out.close();
                dirty_ = false;
                tokens_since_last_sync_ = 0;
                LOG_INFO("Periodic expert stats synced to disk (threshold " << sync_threshold_ << " tokens)");
            }
        }
    }
}

bool expert_stats_tracker::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dirty_ || file_path_.empty() || global_counts_.empty()) {
        return true;
    }

    std::ofstream out(file_path_, std::ios::binary);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open " << file_path_ << " for writing stats");
        return false;
    }

    uint32_t magic = EST1_MAGIC;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&n_layer_), sizeof(n_layer_));
    out.write(reinterpret_cast<const char*>(&n_expert_), sizeof(n_expert_));
    out.write(reinterpret_cast<const char*>(global_counts_.data()), global_counts_.size() * sizeof(uint64_t));
    out.close();
    dirty_ = false;
    tokens_since_last_sync_ = 0;
    LOG_INFO("Expert stats successfully flushed to " << file_path_);
    return true;
}

double expert_stats_tracker::get_adaptive_frequency(uint32_t layer_idx, uint32_t expert_idx) const {
    if (layer_idx >= n_layer_ || expert_idx >= n_expert_) return 0.0;
    std::lock_guard<std::mutex> lock(mutex_);
    return adaptive_scores_[index(layer_idx, expert_idx)];
}

uint64_t expert_stats_tracker::get_global_count(uint32_t layer_idx, uint32_t expert_idx) const {
    if (layer_idx >= n_layer_ || expert_idx >= n_expert_) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return global_counts_[index(layer_idx, expert_idx)];
}

} // namespace stream_moe
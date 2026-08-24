CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -fopenmp -D_CRT_SECURE_NO_WARNINGS -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src -I third_party/llama.cpp/include -Wall -Wextra -Wno-unused-parameter

BUILD_DIR = build
TEMP_DIR = temp

.PHONY: all test clean

all: test

test: $(TEMP_DIR)/test_async_dio $(TEMP_DIR)/test_expert_pool $(TEMP_DIR)/test_moe_loader
	./$(TEMP_DIR)/test_async_dio
	./$(TEMP_DIR)/test_expert_pool
	./$(TEMP_DIR)/test_moe_loader

$(TEMP_DIR)/test_async_dio: tests/test_async_dio.cpp src/io/staging_reader.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp
	@mkdir -p $(TEMP_DIR)
	$(CXX) $(CXXFLAGS) -D_FILE_OFFSET_BITS=64 $^ -o $@

$(TEMP_DIR)/test_expert_pool: tests/test_expert_pool.cpp src/pool/expert_stats.cpp src/pool/expert_pool.cpp
	@mkdir -p $(TEMP_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEMP_DIR)/test_moe_loader: tests/test_moe_loader.cpp third_party/llama.cpp/ggml/src/gguf.cpp src/io/staging_reader.cpp src/loader/moe_loader.cpp
	@mkdir -p $(TEMP_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(TEMP_DIR)
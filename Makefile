CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -fopenmp -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/include -Wall -Wextra -Wno-unused-parameter

SRCS_COMMON = src/io/staging_reader.cpp
SRCS_WIN = src/io/async_dio_win.cpp
SRCS_POSIX = src/io/async_dio_posix.cpp

BUILD_DIR = build
TEMP_DIR = temp

.PHONY: all test clean

all: test

test: $(TEMP_DIR)/test_async_dio
	./$(TEMP_DIR)/test_async_dio

$(TEMP_DIR)/test_async_dio: tests/test_async_dio.cpp $(SRCS_COMMON) $(SRCS_POSIX)
	@mkdir -p $(TEMP_DIR)
	$(CXX) $(CXXFLAGS) -D_FILE_OFFSET_BITS=64 $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(TEMP_DIR)
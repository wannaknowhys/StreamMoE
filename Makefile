# StreamMoE Linux build (mirrors build.bat layout)
#   make llamalibs TAG=main   - build vendored libllama into build/$(TAG)/llama-build
#   make                       - build stream_moe + stream_moe_server
#   make test                  - build + run unit tests
#   make clean                 - remove build/

TAG      ?= main
OUT      := build/$(TAG)
LLAMA_BUILD := $(OUT)/llama-build
BIN      := $(OUT)/bin
OBJ      := $(OUT)/obj
CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -fopenmp -D_CRT_SECURE_NO_WARNINGS \
            -I src -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src \
            -I third_party/llama.cpp/include -I third_party/llama.cpp/vendor \
            -Wall -Wextra -Wno-unused-parameter
LLAMA_LIBS := $(LLAMA_BUILD)/src/libllama.a $(LLAMA_BUILD)/ggml/src/libggml.a \
              $(LLAMA_BUILD)/ggml/src/libggml-base.a $(LLAMA_BUILD)/ggml/src/libggml-cpu.a

.PHONY: all llamalibs test clean

all: $(BIN)/stream_moe $(BIN)/stream_moe_server

$(LLAMA_BUILD)/src/libllama.a:
	@mkdir -p $(LLAMA_BUILD)
	cmake -S third_party/llama.cpp -B $(LLAMA_BUILD) -G Ninja \
	    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
	    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
	    -DLLAMA_CURL=OFF -DGGML_OPENMP=ON -DGGML_NATIVE=ON
	cmake --build $(LLAMA_BUILD) --target llama llama-common-base -j

llamalibs: $(LLAMA_BUILD)/src/libllama.a

$(BIN)/stream_moe: $(LLAMA_BUILD)/src/libllama.a src/main.cpp src/engine/llama_engine.cpp src/profile/profiler.cpp src/backend/moe_backend.cpp src/backend/scheduler.cpp src/io/staging_reader.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp src/pool/expert_stats.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) src/backend/moe_backend.cpp src/backend/scheduler.cpp src/io/staging_reader.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp src/pool/expert_stats.cpp src/engine/llama_engine.cpp src/main.cpp src/profile/profiler.cpp \
	    $(LLAMA_LIBS) -lopenmp -o $@

$(BIN)/stream_moe_server: $(LLAMA_BUILD)/src/libllama.a src/server_main.cpp src/engine/llama_engine.cpp src/server/http_server.cpp src/loader/moe_loader.cpp src/io/staging_reader.cpp src/profile/profiler.cpp src/backend/moe_backend.cpp src/backend/scheduler.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp src/pool/expert_stats.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) src/backend/moe_backend.cpp src/backend/scheduler.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp src/pool/expert_stats.cpp src/engine/llama_engine.cpp src/server/http_server.cpp src/loader/moe_loader.cpp \
	    src/io/staging_reader.cpp src/profile/profiler.cpp src/server_main.cpp \
	    $(LLAMA_LIBS) -lopenmp -lpthread -o $@

test: $(OBJ)/test_async_dio $(OBJ)/test_moe_loader $(OBJ)/test_profiler $(OBJ)/test_scheduler $(OBJ)/test_slot
	./$(OBJ)/test_async_dio
	./$(OBJ)/test_moe_loader
	./$(OBJ)/test_profiler
	./$(OBJ)/test_scheduler
	./$(OBJ)/test_slot

$(OBJ)/test_async_dio: tests/test_async_dio.cpp src/io/staging_reader.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp
	@mkdir -p $(OBJ)
	$(CXX) $(CXXFLAGS) -D_FILE_OFFSET_BITS=64 $^ -o $@

$(OBJ)/test_moe_loader: $(LLAMA_BUILD)/src/libllama.a tests/test_moe_loader.cpp src/io/staging_reader.cpp src/loader/moe_loader.cpp
	@mkdir -p $(OBJ)
	$(CXX) $(CXXFLAGS) $(LLAMA_LIBS) -lopenmp $^ -o $@

$(OBJ)/test_profiler: tests/test_profiler.cpp src/profile/profiler.cpp
	@mkdir -p $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ)/test_scheduler: tests/test_scheduler.cpp src/backend/scheduler.cpp src/io/staging_reader.cpp src/io/async_dio_win.cpp src/io/async_dio_posix.cpp src/pool/expert_stats.cpp
	@mkdir -p $(OBJ)
	$(CXX) $(CXXFLAGS) -lpthread $^ -o $@

$(OBJ)/test_slot: tests/test_slot.cpp
	@mkdir -p $(OBJ)
	$(CXX) $(CXXFLAGS) -lpthread $^ -o $@

clean:
	rm -rf build

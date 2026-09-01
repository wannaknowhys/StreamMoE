# StreamMoE build dispatcher (POSIX). Build rules live in CMakeLists.txt;
# this Makefile only forwards to cmake + make/ninja.
# Targets mirror build.bat: llamalibs | test | clean
#   llamalibs: build vendored libllama static libs into build/<tag>/llama-build
#   test     : build + run unit tests
#   clean    : remove build/
# Usage: make llamalibs TAG=main  |  make test TAG=main
TAG ?= main
BUILD := build/$(TAG)
LLAMA_BUILD := $(BUILD)/llama-build
CMAKE ?= cmake
NINJA ?= ninja
UNAME_S := $(shell uname -s)
GENERATOR := $(if $(filter Linux%,$(UNAME_S)),Unix Makefiles,Ninja)
NPROC := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

.PHONY: llamalibs test clean help

llamalibs:
	@mkdir -p $(LLAMA_BUILD)
	$(CMAKE) -S third_party/llama.cpp -B $(LLAMA_BUILD) -G $(GENERATOR) \
		-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_CURL=OFF -DGGML_OPENMP=ON -DGGML_NATIVE=ON
	$(CMAKE) --build $(LLAMA_BUILD) -j $(NPROC) --target llama llama-common-base
	@echo [+] llamalibs done for tag $(TAG)

test:
	@test -f $(LLAMA_BUILD)/src/libllama.a || { echo "[-] run: make llamalibs TAG=$(TAG) first"; exit 1; }
	$(CMAKE) -S . -B $(BUILD)/cmake -G $(GENERATOR) -DCMAKE_BUILD_TYPE=Release \
		-DLLAMA_BUILD_DIR=$(abspath $(LLAMA_BUILD))
	$(CMAKE) --build $(BUILD)/cmake -j $(NPROC) --target test
	@echo [+] All unit tests passed for tag $(TAG)

clean:
	@rm -rf build
	@echo [+] Clean complete.

help:
	@echo "make llamalibs TAG=main | build | test | clean   (rules in CMakeLists.txt)"

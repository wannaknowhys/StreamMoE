# StreamMoE build rules (single source of truth for the build).
# Consumed via cmake: Windows -> cmake -G Ninja + ninja (build.bat); Linux -> cmake + make (Makefile).
# Pre-req: vendored libllama static libs in <build>/<tag>/llama-build (see `llamalibs`).

cmake_minimum_required(VERSION 3.20)
project(stream_moe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_CRT_SECURE_NO_WARNINGS")
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUNICODE -D_UNICODE")
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# vendored libllama build dir produced by `llamalibs` (cmake+ninja over third_party/llama.cpp)
set(LLAMA_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/../llama-build" CACHE PATH "vendored libllama build directory")
# OpenMP runtime library (Windows/clang); pass via -DSTREAMMOE_LIBOMP=...
set(STREAMMOE_LIBOMP "" CACHE FILEPATH "path to libomp.lib (Windows clang)")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../bin")

set(CORE_SRCS
    src/common/crash.cpp
    src/backend/moe_backend.cpp
    src/backend/minigraph_exec.cpp
    src/backend/scheduler.cpp
    src/io/staging_reader.cpp
    src/loader/moe_loader.cpp
    src/pool/expert_stats.cpp
    src/engine/llama_engine.cpp
    src/profile/profiler.cpp
)
if(WIN32)
    list(APPEND CORE_SRCS src/io/async_dio_win.cpp)
else()
    list(APPEND CORE_SRCS src/io/async_dio_posix.cpp)
endif()

function(streammoe_inc tgt)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
endfunction()

function(streammoe_target tgt)
    streammoe_inc(${tgt})
    if(WIN32)
        target_link_libraries(${tgt} PRIVATE
            "${LLAMA_BUILD_DIR}/src/llama.lib"
            "${LLAMA_BUILD_DIR}/ggml/src/ggml.lib"
            "${LLAMA_BUILD_DIR}/ggml/src/ggml-base.lib"
            "${LLAMA_BUILD_DIR}/ggml/src/ggml-cpu.lib"
            ws2_32 advapi32 synchronization)
        if(STREAMMOE_LIBOMP)
            target_link_libraries(${tgt} PRIVATE "${STREAMMOE_LIBOMP}")
        endif()
    else()
        target_link_libraries(${tgt} PRIVATE
            "${LLAMA_BUILD_DIR}/src/libllama.a"
            "${LLAMA_BUILD_DIR}/ggml/src/libggml.a"
            "${LLAMA_BUILD_DIR}/ggml/src/libggml-base.a"
            "${LLAMA_BUILD_DIR}/ggml/src/libggml-cpu.a")
    endif()
endfunction()

add_executable(stream_moe ${CORE_SRCS} src/main.cpp)
streammoe_target(stream_moe)

add_executable(stream_moe_server ${CORE_SRCS} src/server/http_server.cpp src/server_main.cpp)
streammoe_target(stream_moe_server)

# ---- unit tests ----
enable_testing()
add_executable(test_async_dio src/io/async_dio_win.cpp src/io/staging_reader.cpp tests/test_async_dio.cpp)
streammoe_inc(test_async_dio)
add_test(NAME test_async_dio COMMAND test_async_dio WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_moe_loader src/io/staging_reader.cpp src/loader/moe_loader.cpp tests/test_moe_loader.cpp)
streammoe_target(test_moe_loader)
add_test(NAME test_moe_loader COMMAND test_moe_loader WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_profiler src/profile/profiler.cpp tests/test_profiler.cpp)
streammoe_inc(test_profiler)
add_test(NAME test_profiler COMMAND test_profiler WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_scheduler src/backend/scheduler.cpp src/io/staging_reader.cpp src/io/async_dio_win.cpp src/pool/expert_stats.cpp tests/test_scheduler.cpp)
streammoe_inc(test_scheduler)
if(WIN32)
    target_link_libraries(test_scheduler PRIVATE synchronization)
endif()
add_test(NAME test_scheduler COMMAND test_scheduler WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_slot tests/test_slot.cpp)
streammoe_inc(test_slot)
if(WIN32)
    target_link_libraries(test_slot PRIVATE synchronization)
endif()
add_test(NAME test_slot COMMAND test_slot WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


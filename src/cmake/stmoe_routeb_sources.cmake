# StreamMoE route-B engine sources (parent repo src/), compiled into llama-common.
# Included (OPTIONAL) from vendored common/CMakeLists.txt; active only for route_b features.
if(STREAM_MOE_FEATURES MATCHES "route_b")
    set(STREAM_MOE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)
    target_include_directories(${TARGET} PRIVATE ${STREAM_MOE_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    # Feature frag include dirs are added globally by the root CMakeLists feature block.
    target_sources(${TARGET} PRIVATE
        ${STREAM_MOE_SRC}/server/route_b_inject.cpp
        ${STREAM_MOE_SRC}/backend/moe_backend.cpp
        ${STREAM_MOE_SRC}/backend/minigraph_exec.cpp
        ${STREAM_MOE_SRC}/backend/device_iface.cpp
        ${STREAM_MOE_SRC}/backend/mix_split.cpp
        ${STREAM_MOE_SRC}/backend/scatter_plan.cpp
        ${STREAM_MOE_SRC}/backend/scheduler.cpp
        ${STREAM_MOE_SRC}/backend/route_b_chain.cpp
        ${STREAM_MOE_SRC}/io/staging_reader.cpp
        ${STREAM_MOE_SRC}/loader/moe_loader.cpp
        ${STREAM_MOE_SRC}/loader/model_builder.cpp
        ${STREAM_MOE_SRC}/loader/topo_builder.cpp
        ${STREAM_MOE_SRC}/pool/expert_stats.cpp)
    # Device interface: Vulkan ops only when the vendored build enables
    # ggml-vulkan, otherwise the stmoe_vk_* symbols do not exist and a
    # CPU-only build would fail to link (BUG_TRACKER B33).
    if(GGML_VULKAN)
        target_compile_definitions(${TARGET} PRIVATE STREAMMOE_HAVE_VULKAN=1)
        target_sources(${TARGET} PRIVATE
            ${STREAM_MOE_SRC}/backend/device_vulkan.cpp)
    endif()
    # Device interface: CUDA ops only when the vendored build enables
    # ggml-cuda, otherwise the CUDA runtime symbols are unavailable.
    # device_cuda.cpp is a plain .cpp linked against cudart (no .cu file).
    if(GGML_CUDA)
        target_compile_definitions(${TARGET} PRIVATE STREAMMOE_HAVE_CUDA=1)
        target_sources(${TARGET} PRIVATE
            ${STREAM_MOE_SRC}/backend/device_cuda.cpp)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
    endif()
    # Async DIO backend by platform: IOCP (Windows) vs pread/io_uring (POSIX).
    if (WIN32)
        target_sources(${TARGET} PRIVATE ${STREAM_MOE_SRC}/io/async_dio_win.cpp)
        target_link_libraries(${TARGET} PRIVATE ws2_32 advapi32 synchronization)
    else()
        target_sources(${TARGET} PRIVATE ${STREAM_MOE_SRC}/io/async_dio_posix.cpp)
    endif()
endif()

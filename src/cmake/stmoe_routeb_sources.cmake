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
        ${STREAM_MOE_SRC}/backend/mix_split.cpp
        ${STREAM_MOE_SRC}/backend/scatter_plan.cpp
        ${STREAM_MOE_SRC}/backend/scheduler.cpp
        ${STREAM_MOE_SRC}/backend/route_b_chain.cpp
        ${STREAM_MOE_SRC}/io/staging_reader.cpp
        ${STREAM_MOE_SRC}/loader/moe_loader.cpp
        ${STREAM_MOE_SRC}/loader/model_builder.cpp
        ${STREAM_MOE_SRC}/loader/topo_builder.cpp
        ${STREAM_MOE_SRC}/pool/expert_stats.cpp)
    # Async DIO backend by platform: IOCP (Windows) vs pread/io_uring (POSIX).
    if (WIN32)
        target_sources(${TARGET} PRIVATE ${STREAM_MOE_SRC}/io/async_dio_win.cpp)
        target_link_libraries(${TARGET} PRIVATE ws2_32 advapi32 synchronization)
    else()
        target_sources(${TARGET} PRIVATE ${STREAM_MOE_SRC}/io/async_dio_posix.cpp)
    endif()
endif()

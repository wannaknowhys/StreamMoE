#pragma once

// [TMR] Raw-TSC clock for the async-load hot path (req/dio/done_tsc fields in
// async_load_t). A timestamp costs one __rdtsc (~20 cycles) and stores raw TSC
// ticks; nothing is calibrated or printed on that path. Profiling code converts
// stored ticks to ns via tsc_to_ns(), which lazily calibrates once against the
// steady_clock and shares the result process-wide (inline variables are a single
// entity across all translation units, so calibration happens at most once).

#include <chrono>
#include <cstdint>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <x86intrin.h>
#    endif
#endif

namespace stream_moe {

// Raw CPU timestamp. x86: invariant TSC (one counter across cores on modern
// CPUs); other targets fall back to monotonic ns so stored fields stay valid.
inline uint64_t tsc_now() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    return static_cast<uint64_t>(__rdtsc());
#else
    return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

namespace detail {

inline std::once_flag tsc_calib_flag;
inline uint64_t tsc_calib_base_tsc = 0;
inline int64_t  tsc_calib_base_ns  = 0;
inline double   tsc_calib_ticks_per_ns = 0.0;

inline void tsc_calibrate() {
    // Sample both clocks over a ~2ms window to map a raw TSC value to ns.
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t r0 = tsc_now();
    std::chrono::steady_clock::time_point t1;
    do {
        t1 = std::chrono::steady_clock::now();
    } while (t1 - t0 < std::chrono::milliseconds(2));
    const uint64_t r1 = tsc_now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    if (ns > 0.0 && r1 > r0) {
        tsc_calib_ticks_per_ns = static_cast<double>(r1 - r0) / ns;
        tsc_calib_base_tsc = r1;
        tsc_calib_base_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 t1.time_since_epoch()).count();
    }
    // On failure ticks_per_ns stays 0 and tsc_to_ns reports 0; the hot path is
    // never touched by calibration.
}

} // namespace detail

// Convert a raw TSC value (from tsc_now or a stored field) to ns since process
// start, in the steady_clock timeline. First call calibrates once, process-wide.
inline int64_t tsc_to_ns(uint64_t raw) {
    std::call_once(detail::tsc_calib_flag, detail::tsc_calibrate);
    const double tpn = detail::tsc_calib_ticks_per_ns;
    if (tpn <= 0.0 || raw < detail::tsc_calib_base_tsc) return 0;
    return detail::tsc_calib_base_ns +
           static_cast<int64_t>(static_cast<double>(raw - detail::tsc_calib_base_tsc) / tpn);
}

// Convert a TSC interval (e.g. done_tsc - req_tsc) to ns. Single division, no
// base term (absolutes cancel); profile path only, calibration runs once.
inline int64_t tsc_delta_ns(uint64_t delta_ticks) {
    std::call_once(detail::tsc_calib_flag, detail::tsc_calibrate);
    const double tpn = detail::tsc_calib_ticks_per_ns;
    if (tpn <= 0.0) return 0;
    return static_cast<int64_t>(static_cast<double>(delta_ticks) / tpn);
}

} // namespace stream_moe

# VRAM DMA Move Pipeline (demote/r2v over the transfer queue)

[English](VRAM_DMA_MOVE.md) | [简体中文](VRAM_DMA_MOVE.zh-CN.md)

> Status: **implemented + verified, 2026-09**. A working session measured the
> RX590 transfer paths, settled the approach and landed it with the M4 move
> worker (commits 816f8aa / 717bac8; see §5 for the per-step status). Builds on
> `docs/EXPERT_MOVE_PIPELINE.md` (the M4 async move worker + (L,E)-keyed
> eviction) and the VRAM data layer (docs/WORK_IN_PROGRESS.md J). Note that
> §5.2 of EXPERT_MOVE_PIPELINE ("v2r is CPU-memcpy-only") was overturned by
> this work - see the revision box there.

## 1. Problem

Demote (VRAM->RAM expert eviction) is the decode bottleneck. On this hardware
(Radeon RX 590, discrete, rebar BAR1) a **host-side memcpy FROM the VRAM
host-map reads at ~0.02 GB/s** - one gemma expert (3.63 MB) costs ~158 ms, and
a prefill/decode pass with vram pressure does hundreds of demotes.

Measured host-read bandwidth by memory type (direct vulkan test):

| memory type | heap | host read | note |
|---|---|---|---|
| DEVICE_LOCAL \| HOST_VISIBLE (rebar) | vram 8 GB | **0.02 GB/s** | current move source - unusable |
| HOST_VISIBLE \| COHERENT | sys-ram 64 GB | 0.27 GB/s | uncached |
| HOST_VISIBLE \| COHERENT \| **CACHED** | sys-ram 64 GB | **21-27 GB/s** | ggml sync_staging heap |

## 2. Transfer-queue DMA is the fix

`vkCmdCopyBuffer` on a transfer-capable queue (RX590 has a transfer-only
family) copies VRAM -> a host buffer at **~14 GB/s**, and the CPU can then read
that host buffer at cached speed:

| path | 3.63 MB expert |
|---|---|
| CPU memcpy from VRAM rebar map (current) | ~158 ms |
| `vkCmdCopyBuffer` VRAM -> CACHED staging | ~0.38 ms |
| + memcpy CACHED staging -> RAM slot | ~0.17 ms |
| total (DMA + memcpy) | **~0.5 ms (~300x faster)** |

10-expert concurrency over one fixed staging buffer showed **no bandwidth
loss**: sequential per-expert submits 9.5 GB/s, one batch of 10 copies 10.7
GB/s, pipelined 10 submits 10.2 GB/s (all ~0.35-0.39 ms/expert).

## 3. Rejected alternatives (measured)

- **Import the whole RAM pool as vulkan memory** (VK_EXT_external_memory_host):
  the driver accepts only **one ~2 GB host import** per process; 10 x 2 GB
  plain allocations also fail after the first. So the expert pool cannot be a
  multi-block vulkan buffer (its host map would be the fast heap).
- **Direct vkCmdCopyBuffer into ordinary malloc pool slots**: requires the
  pool memory to be a vulkan buffer (import, rejected above).

Host-map bandwidth is **asymmetric**: VRAM rebar CPU **write is ~8 GB/s** (PCIe
posted writes) but CPU **read is ~0.02 GB/s**. Consequences:
- r2v (loading RAM/disk bytes into VRAM, = CPU writes to the vram host-map)
  needs **no staging** - the current DIO-direct write path already runs at the
  fast write speed.
- Only v2r demote (reading VRAM back to RAM) is slow; that is the single path
  that needs the DMA/staging fix below.

Conclusion: **one fixed 2 GB CACHED staging buffer** (heap0 / memtype 7) as a
DMA bounce for v2r only, sized ~10 expert slots for concurrency, then a fast
memcpy into the ordinary RAM slot. The RAM slot addressing (base + offset) is
unchanged.

## 4. Design

### 4.1 Layering (keep scheduler ggml-free)

`expert_scheduler` is pure C++ (no ggml/vulkan). It must not learn vulkan
types. Implementation (as landed) exports the DMA read as **one plain function
with all-void arguments** from a frag included at the route-B anchor of
`ggml-vulkan.cpp`, where the internal vulkan types and the TU-static
`ggml_vk_buffer_read` are visible. The scheduler declares it `extern` and calls
it directly; no vulkan/ggml type crosses the boundary.

```
ggml-vulkan.cpp (vendored, patched)
   + route-B anchor includes stmoe_routeb_vk_dma.frag
      exports: stmoe_vk_dma_read(void*, size_t, void*, size_t)
               stmoe_vk_dma_available()
expert_scheduler.move_worker   (control plane unchanged)
   |  per move task column:
   |    src_pool != 0 && copy.dev_buf  -> stmoe_vk_dma_read(dev_buf, dev_off,
   |                                                        dst, bytes)   [v2r]
   |    else                          -> plain memcpy (RAM sources; r2v stays
   |                                     DIO-direct - host-map writes are fast)
   |  report per-stage rdtsc deltas
```

- The frag wraps ggml-vulkan's internal `ggml_vk_buffer_read` (declared static
  just above the anchor): a synchronous transfer-queue `vkCmdCopyBuffer` into
  ggml's HOST_CACHED staging buffer, then memcpy to `dst`. The staging buffer
  and fence wait are owned by ggml-vulkan - route B does not build its own.
- scheduler control plane (submit_move / state machine / drain_moves) is
  **unchanged**. `move_task_t::copy_t` only gains `dev_buf` (void* device region
  handle) + `dev_off` (byte offset inside it), filled at submit time from the
  region's vk buffer (scheduler.cpp:257).
- worker thread **stays** (it still issues the DMA + does the fence wait / any
  staging copy); it no longer reads the slow rebar map.

### 4.2 Staging

The staging is ggml-vulkan's internal `sync_staging` (HOST_CACHED, heap0 /
memtype 7). Each `dma_read` is one synchronous submit: transfer-queue copy
vram -> staging -> memcpy staging -> `dst`, one fence wait per call.

The earlier "fixed multi-slot staging ring (~10 slots, round-robin)" idea was
**not needed**: the resulting ~0.5-1 ms/expert (measured in §5) is already
~150-300x faster than the rebar read, so the move worker is no longer the
decode bottleneck. A pipelined staging ring (several copies in flight) would
only amortise the per-call fence wait; deferred until the move worker shows up
in profiles again.

### 4.3 Exported interface (frag side)

```cpp
// patches/route-b/common/stmoe_routeb_vk_dma.frag - included at the route-B
// anchor in ggml-vulkan.cpp; wraps TU-static ggml_vk_buffer_read.
void stmoe_vk_dma_read(void * buffer /* ggml_backend_buffer_t */, size_t off,
                       void * dst, size_t bytes); // synchronous
bool stmoe_vk_dma_available(void);                // diagnostic only
```

`buffer` is opaque at the call site (the vram region's `ggml_backend_buffer_t`,
stored as `void*` on the region and copied into the move task). No
moe_backend/route_b registration object is needed - the symbol links from the
same ggml-vulkan TU as the region buffers.

### 4.4 Timing (rdtsc, gated behind STREAM_MOE_LOG=debug)

Per move, log:
- `dma_us`  = vkCmdCopyBuffer submit -> queue idle (download time)
- `mc_us`   = staging -> RAM slot copy time (0 with the internal staging path,
              already inside the dma_read)
- plus the existing queued total

So the decision "drop the memcpy worker" can be re-evaluated from data: if
`dma_us + mc_us` is small and the worker is no longer the bottleneck, the worker
may merge into the scheduler thread (submit + fence wait there). The current
~1 ms/expert means it stays a worker for now.

## 5. Build order

1. Wrap ggml-vulkan's internal device->host path for route B: a frag at the
   route-B anchor exports `stmoe_vk_dma_read` (TU-static `ggml_vk_buffer_read`
   wrapper). **[done - 816f8aa]**
2. scheduler: move worker uses the DMA for device sources, keeps memcpy for RAM
   sources; add per-stage rdtsc logging. **[done - copy_t.dev_buf/dev_off +
   worker dma branch + dma/mc timing - 717bac8]**
3. Verify with the existing VRAM config (RAM 1GB + Vulkan0 64MB): demote storm
   should drop from ~158 ms/expert to ~0.5 ms; the previously ~unusable decode
   becomes responsive. **[done - 4-token 64MB: 238 DMA demotes at ~0.9-1.5 ms
   (first ~10 ms = lazy staging init); 129-token 64MB runs through (~14 s,
   dominated by cold DIO + CPU compute)]**
4. Regression: pure-RAM path must stay byte-IDENTICAL (RAM sources have
   dev_buf==null -> plain memcpy, path unchanged). Mixed execution numeric gate
   re-run. **[done - 4-token DMA-demote run vs pure-RAM golden: 64 columns,
   0 BAD (22 bit / 42 ulp as expected for the mixed CPU/GPU split)]**
5. r2v needs no change: VRAM host-map writes already run at ~8 GB/s (the DIO
   direct-load path), so the DMA service is v2r-only. **[done - r2v still DIO]**

## 6. Files touched

- `patches/route-b/common/stmoe_routeb_vk_dma.frag` - NEW: exports
  `stmoe_vk_dma_read` / `stmoe_vk_dma_available`; included at the route-B anchor
  in vendored ggml-vulkan.cpp (recorded in `streammoe-macros.patch`, one include
  line).
- `src/backend/scheduler.h` / `scheduler.cpp` - `move_task_t::copy_t` +
  `region.dev_buf` gain `dev_buf`/`dev_off`; move worker branches to
  `stmoe_vk_dma_read` (extern-declared, all-void args) for device sources; dma/mc
  rdtsc timing under STREAM_MOE_LOG=debug.
- `docs/WORK_IN_PROGRESS.md`, `docs/CHECKPOINT.md` - status (M-section,
  current state).
- Measurement probes (temp/, local only): vram_bw / vram_heaps / vram_dma /
  demote10 / rebar_write - produced the §1-§2 numbers.
- new UT (optional, not written): staging slot lifecycle - deferred (no
  dedicated staging ring was built, §4.2).

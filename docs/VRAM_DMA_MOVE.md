# VRAM DMA Move Pipeline (demote/r2v over the transfer queue)

[English](VRAM_DMA_MOVE.md) | [简体中文](VRAM_DMA_MOVE.zh-CN.md)

> Status: **design + decided, 2026-09**. Working session measured the RX590
> transfer paths and settled the approach; implementation follows. Builds on
> `docs/EXPERT_MOVE_PIPELINE.md` (the M4 async move worker + (L,E)-keyed
> eviction) and the VRAM data layer (docs/WORK_IN_PROGRESS.md J).

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

Conclusion: **one fixed 2 GB CACHED staging buffer** (heap0 / memtype 7) as a
DMA bounce, sized ~10 expert slots for concurrency, then a fast memcpy into the
ordinary RAM slot. The RAM slot addressing (base + offset) is unchanged.

## 4. Design

### 4.1 Layering (keep scheduler ggml-free)

`expert_scheduler` is pure C++ (no ggml/vulkan). It must not learn vulkan. A
device-side DMA service is owned by route B / moe_backend (which hold the vulkan
device/queue/buffers) and injected into the scheduler as a plain function
pointer / handle on the model pool.

```
route_b_inject (vulkan device/queue/buffers)
   |  provides: vram_dma service (function pointer + opaque ctx)
   v
expert_scheduler.move_worker   (unchanged control plane)
   |  per move task:
   |    1. dma_service->download(vram_buf, src_slot_off, bytes, staging_slot_ptr)
   |       = vkCmdCopyBuffer(vram_buf -> staging buf at slot)  [transfer queue]
   |    2. memcpy(RAM slot dst, staging_slot_ptr, bytes)        [worker CPU]
   |    3. report per-stage rdtsc deltas
```

- scheduler control plane (submit_move / state machine / drain_moves) is
  **unchanged**.
- move_task keeps its per-column (src,dst) ranges; the worker consults the DMA
  service for the source side instead of memcpy'ing from the rebar map.
- worker thread **stays** (it still does the staging->RAM memcpy + fence wait);
  it no longer reads the slow rebar map.

### 4.2 Staging layout

- One fixed CACHED staging buffer, ~10 expert slots (gemma expert = 3.63 MB ->
  ~37 MB), grown on demand up to the 2 GB driver window if a model's expert is
  larger.
- Slot stride = expert_size (compact). The worker round-robins the staging
  slots; a slot is reused only after its memcpy to the RAM slot finished, so a
  slot never has two in-flight DMA writes.

### 4.3 DMA service interface (moe_backend side)

```cpp
// owned by moe_backend/route_b; registered once per model pool
struct vram_dma_ctx;  // opaque
using vram_dma_download_t = bool (*)(vram_dma_ctx* c, uint32_t pool,
                                     void* vram_buf, size_t vram_off,
                                     uint8_t* staging_dst, size_t bytes);
```

The scheduler stores `{ctx, fn}` on its model pool; `move_worker_main` calls
`fn` when `src_pool != 0`, falls back to the plain memcpy for RAM sources.

### 4.4 Timing (rdtsc, gated behind STREAM_MOE_LOG=debug)

Per move, log:
- `dma_us`  = vkCmdCopyBuffer submit -> queue idle (download time)
- `mc_us`   = memcpy staging -> RAM slot (relocation time)
- plus the existing queued total

So the decision "drop the memcpy worker" can be re-evaluated from data: if
`dma_us + mc_us` is small and the worker is no longer the bottleneck, the worker
may merge into the scheduler thread (submit + fence wait there). Deferred until
the numbers land.

## 5. Build order

1. moe_backend: create the cached staging buffer + transfer queue on first use;
   expose `vram_dma_download` (vkCmdCopyBuffer + fence).
2. scheduler: inject the DMA service on the model pool; move_worker uses it for
   device sources, keeps memcpy for RAM sources; add per-stage rdtsc logging.
3. Verify with the existing VRAM config (RAM 1GB + Vulkan0 64MB): demote storm
   should drop from ~158 ms/expert to ~0.5 ms; the previously ~unusable decode
   becomes responsive.
4. Regression: pure-RAM path must stay byte-IDENTICAL (no DMA service -> plain
   memcpy). Mixed execution numeric gate re-run.
5. r2v (RAM->VRAM) later uses the same staging/dma service reversed.

## 6. Files touched

- `src/backend/scheduler.h` / `scheduler.cpp` - inject DMA service; worker path
- `src/backend/moe_backend.h` / `moe_backend.cpp` - cached staging buffer +
  transfer queue + `vram_dma_download`
- `src/server/route_b_inject.cpp` - create + register the service per pool
- `docs/WORK_IN_PROGRESS.md`, `docs/CHECKPOINT.md` - status
- new UT (optional): staging slot lifecycle

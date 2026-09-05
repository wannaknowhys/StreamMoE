# Expert Move Pipeline + (L,E)-Keyed Eviction - Design

[English](EXPERT_MOVE_PIPELINE.md) | [简体中文](EXPERT_MOVE_PIPELINE.zh-CN.md)

> Status: **design, 2026-09** - a working session decided the next scheduler
> redesign. The landed batch-pin layer (slot_request bitmap / wake-once, see
> WORK_IN_PROGRESS L) is the base this builds on. This doc captures the FULL
> design so nothing is lost between sessions.
> Related: `docs/WORK_IN_PROGRESS.md` L / J, `docs/Backend.md` §2/§4,
> `docs/EXPERT_SCHEDULER_DESIGN.md`, `src/backend/scheduler.{h,cpp}`,
> `src/backend/slot.h`.
>
> **2026-09-04 revision**: build-order steps 1-4 landed (M1-M4, commits
> 6b6300e / dbaff8f / 45b14de / fdf4982). M5 (active-slot accounting) and M7
> (doc convergence) still open; M6 partial. Step 4's v2r is now transfer-queue
> DMA, which **overturns §5.2's "CPU-memcpy-only" conclusion** - see the
> revision box there and `docs/VRAM_DMA_MOVE.md`.

## 1. Motivation / why we redesign

Observed while trying to validate SoA-column pools on VRAM:

1. **Per-expert blocking pin serialised DIO** and ping-ponged exec/scheduler -
   fixed by batch pin (WORK_IN_PROGRESS L, landed): a 129-token prefill went
   from tens of seconds to 0.11s on the pure-RAM path, byte-IDENTICAL.
2. **VRAM runs still stall** (~900+ demotes on a 129-token prefill, never
   reaching `exec_mm_vk`): every new expert load into a full device region
   synchronously demotes a victim (memcpy whole expert VRAM->RAM on the
   scheduler thread, recursively making RAM room). The batch-pin change
   exposed this: the DIO serialisation is gone, the demote serialisation is
   now the bottleneck.
3. **Demote must be asynchronous** (worker does the memcpy, scheduler never
   blocks on it), and the same worker must serve **v2r (VRAM->RAM) AND r2v
   (RAM->VRAM)** - a single move pipeline, not a demote special case.

A secondary driver: **eviction keying**. Today eviction scans the pool's slots
and reverse-looks-up (L,E) via `owner_[]`. The redesign keys eviction on (L,E)
with layer-distance preference (evict the just-finished layer first), which
removes the need for `owner_` entirely.

## 2. Landed base (WORK_IN_PROGRESS L) - do not regress

- `slot_request_t` 96B POD: `{layer, total_tokens, start_rdtsc, n_load_target(=batch
  target count), batch_ready ptr, needed[8]=512-bit bitmap}`.
- `mpsc_alloc_queue`: plain POD ring + per-slot publish generation
  (release/acquire), multi-producer safe, no ABA. NO `std::atomic<96B>`.
- `expert_scheduler::pin_layer(layer, bitmap, await, out)`: scan dir -> pin
  resident, submit ONE batch for the missing subset, wake-once on the counter,
  one retry round. `pin_expert`/`wait_ready` removed.
- `accept_requests`: pops a layer request, iterates the bitmap, device-first
  placement, per-bit batch bump, requeues unplaced bits.
- `drain_completions`: settle bumps the batch counter + wakes.
- init fails fast when `n_expert > MAX_EXPERTS_PER_LAYER (512)`.
- Verification: gemma v2align prefill-from IDENTICAL, 0.11s (RAM-only); 5/5 UT.

## 3. Directory as (L,E)-keyed lifecycle state table

### 3.1 Today (3 parallel arrays, all indexed by (L,E))

```
entries_  [(L,E)*n_pools + pool]  -> slot            (expert -> per-pool slot)
versions_ [(L,E)]                 -> u32             (wait/wake version)
last_used_[(L,E)]                 -> u64             (recency, cross-pool)
stats_.adaptive_scores_[(L,E)]    -> double          (EST1 freq, scheduler-private)
owner_    [slot]                  -> (L,E)           (slot -> expert, reverse)
```

Access pattern today: exec scans by (L,E); eviction scans slots then uses
`owner_` to get (L,E) then re-indexes last_used_/freq. So the only
**traversal** in the hot path is eviction's slot scan.

### 3.2 Problem: half-way states are invisible to exec

`entries_` only records READY residency (`dir_->set` fires at mark_ready).
Between "alloc reserved the slot" and "mark_ready", the slot is IO_INFLIGHT
and `owner_[slot]` already names (L,E) but `entries_[(L,E)]` is UNASSIGNED.
So a compute-side scan cannot see an expert that is *being loaded*, and a
scheduler request cannot tell "already loading" from "absent" - both look
identical through entries_. This creates a duplicate-load window (exec asks
for an expert that is mid-load; accept allocs a second slot; two loads race,
one becomes an orphan READY slot).

### 3.3 Proposal: state per (L,E) x pool

Keep the flat (L,E)-indexed layout (good cache behaviour for the dominant
random-access pattern) but widen each entry to carry the expert's lifecycle
phase. Two layout options:

- **A1 (recommended): two parallel atomics per (L,E,p)**
  `state_[(L,E)*n_pools+pool]` (u32) + `slot_[(L,E)*n_pools+pool]` (u32).
  State changes are frequent (moves), slot is stable once set. Read-mostly.
- **A2: single u64 `{state(4) | slot(28) | gen(32)}`** per (L,E,p).
  One atomic read for scan, but every state change rewrites the whole word.

Size: deepseek 43 layers x 256 experts x ~2 pools = ~22k entries; either
layout is a few hundred KB. Negligible.

States (per (L,E) x pool; an expert may hold RAM + VRAM copies simultaneously):

```
ABSENT       no slot    not in this pool
LOADING      slot Y     reserved, DIO in flight (slot IO_INFLIGHT), NOT ready
READY        slot Y     pin-able (current behaviour)
MOVING_OUT   slot Y     this copy is being moved away (v2r/r2v source), not
                        pin-able (source slot stays EVICTING until the move settles)
MOVING_IN    slot Y     this copy is being moved in (target), not pin-able;
                        slot Y is the ALREADY-RESERVED destination physical slot
                        (IO_INFLIGHT) - set together with the state, never a stub
FAILED       no slot    load/move failed
```

Notes on state entries:
- When a state carrying a slot is published, `slot` must be the real reserved
  physical slot id (LOADING = its IO_INFLIGHT slot; MOVING_IN = the reserved
  destination slot). Same visibility rule as §3.4: publish the intent with a
  valid slot, THEN the worker acts on it - never publish a placeholder slot.
- MOVING_OUT and MOVING_IN are both **not pin-able** (try_pin requires READY),
  symmetric - see §5.6.

A move is two entries in two pools at once: `MOVING_OUT(src pool)` +
`MOVING_IN(dst pool)`; on completion source -> ABSENT, destination -> READY.

**Where the full description lives - not in the directory.** The directory only
answers the compute-side question "is (L,E) usable here, and where?". The full
move description (src pool/slot, dst pool/slot, columns) lives in the move
task object (see §5). cq carries few bytes, but `drain_completions` gets the
task pointer back from `done[i]->user_data`, so the task holds everything.

**Single-writer discipline unchanged**: only the scheduler thread mutates these
states; compute threads read only. Existing `version[(L,E)]` bump + wake on
every transition stays - any half-way state change wakes waiters to rescan.

### 3.4 Ordering invariant (must be fixed)

When the scheduler decides to load (L,E) into pool p slot s:

```
dir(L,E,p) = LOADING(slot=s)   // publish intent FIRST (seq-cst store = visibility point)
owner side: (removed, see §4)
slots_[s].begin_reload()       // physical slot -> IO_INFLIGHT
```

The reverse order (reserve slot first, publish later) leaves a window where
the slot is loading but the directory cannot see it. Fixed order kills the
duplicate-load window.

**Completion-side ordering is the mirror and just as binding.** A compute
scan that sees `dir = READY` MUST be able to pin the physical slot. So the
completion tail is:

```
slots_[s].mark_ready()            // physical slot -> READY (releases readers)
memory fence (seq-cst)
dir(L,E,p) : LOADING -> READY     // only now may a compute scan observe READY
```

Reverse (dir READY before physical READY) lets a scan see READY and try_pin a
slot that is still IO_INFLIGHT - false wake / failed pin. Load completion,
move completion (MOVING_IN -> READY) and demote-to-RAM all use this order.

### 3.5 Compute-side pin semantics: stateless exec

Exec is deliberately **stateless about residency**: it never reads state
transitions, never distinguishes ABSENT / LOADING / MOVING_IN. It only tries to
pin and re-asks for whatever failed:

```
exec pin_layer(bitmap):
  loop:
    for each needed (L,E):
      find a READY copy (dir scan) -> try_pin -> success: record handle
      else (ABSENT / LOADING / MOVING_IN / pin race) -> mark still-need
    if none still-need: return all handles
    submit ONE request { layer, still-need bitmap, n_load_target = count } 
    sleep on the batch completion signal (wake-once on READY of all still-need)
    // scheduler makes every still-need expert READY (own load or an existing
    // in-flight one); exec re-pins after the wake; loops on any remaining race
```

Idempotent, self-healing: a retry round re-measures what is still not READY and
re-submits with a FRESH n_load_target equal to that re-measured count (never the
round-1 total). Exec does not need to know whether an expert was already being
loaded by someone else (incl. a prefetch) - the scheduler guarantees the expert
becomes READY, and exec is woken at that point.

FAILED (a load/move hard-failed) is surfaced as an error, not retried forever.

> Decision history: an earlier "decision 6a" (exec never *submits* in-flight
> experts because it can see LOADING) is superseded. Exec cannot reliably
> distinguish "will be loaded by my own request" from "already loading for
> someone else" at submit time (races + prefetch), so it stays stateless and
> the scheduler owns making every still-need expert READY.

## 4. Remove owner_ (slot -> (L,E))

Evidence: commenting out `owner_` produced exactly 8 compile errors, ALL inside
`alloc_or_evict` (registrations :258/:328, scoring reads :278/:281, victim
reverse-lookup :293-294). No other file/function touches it.

Once eviction is (L,E)-keyed (below), the scheduler never needs to ask "what
expert lives in slot i?":
- load completion `dir_->set(t->layer, t->expert, ...)` already carries (L,E)
  from the async_load_t;
- eviction selects by (L,E) and reads the slot via `entries_`.

So `owner_` is deleted. The slot side keeps only physical state (slot_meta).

## 5. Move pipeline: async v2r + r2v worker

### 5.1 Principle: demote is a cache migration, best-effort

Expert weights are read-only; the VRAM content equals the disk content. Moving
content to RAM is a bonus (saves a future disk read), never a requirement. So
the move pipeline is a low-priority background task; the scheduler must never
block on it. (Working decision: wait for the copy to finish before reusing the
source slot - do NOT drop mid-copy. Dropping was considered and rejected for
state clarity; a `MOVING` state + scheduler-side in-flight record keeps the
model clean.)

### 5.2 v2r device read: "CPU-memcpy-only" is REVISED (DMA via cached staging)

Original reasoning (kept for history): `vkCmdCopyBuffer` / copy engines can copy
device<->device and device<->host-visible-vulkan-buffer. On a discrete GPU (RX
590) the host-visible heap is still VRAM (rebar BAR1); the 128GB system-RAM pool
is NOT in the GPU address space, so a copy engine cannot write it. A real
`vkCmdCopyBuffer` vram->system-RAM target does not exist on discrete GPUs (only
on UMA/APU). Cross-backend `ggml_backend_tensor_copy` likewise falls back to
memcpy or requires a host-visible target buffer that would occupy VRAM.
Conclusion then: v2r to ordinary malloc RAM is CPU-memcpy-only on this
hardware; GPU copy-engine DMA is a future UMA/APU target.

> **2026-09-04 revision - that conclusion was wrong for a two-step path.**
> A discrete GPU's copy engine cannot write *ordinary malloc* RAM (correct
> above), but it CAN write a CACHED host-visible *vulkan* host buffer (heap0 /
> memtype 7), and reading that host buffer back at cached speed is fast.
> Measured on RX590: `vkCmdCopyBuffer` vram -> cached staging ~14 GB/s, memcpy
> staging -> RAM slot ~21+ GB/s => one demote went from ~158 ms (rebar CPU read
> at 0.02 GB/s) to ~0.5 ms. v2r is now transfer-queue DMA through ggml-vulkan's
> cached staging (`stmoe_vk_dma_read`, wrapping its internal
> `ggml_vk_buffer_read`), then a memcpy into the RAM slot. Full design and the
> rejected whole-pool-import alternative: `docs/VRAM_DMA_MOVE.md`.

### 5.3 Task shape (move_task)

One dedicated copy worker (parameterised count, start at 1). Single task per
expert move; submit in batches. Worker does pure per-column memcpy, never
touches control plane.

Why one worker (not a pool): the v2r bottleneck is the **device->host link
bandwidth**, not CPU core count - extra workers do not add bandwidth, they only
add concurrency. Pre-DMA this was the PCIe read of the rebar map (0.02 GB/s);
after the §5.2 revision it is the transfer-queue DMA + staging memcpy (~14
GB/s), still not CPU-bound. One worker (submit + fence wait + staging memcpy)
saturates it; the count stays a parameter in case a faster link or UMA target
changes that calculus.

```cpp
struct move_task_t {
    // kind: MOVE_V2R (src=vram pool, dst=RAM pool) or MOVE_R2V (reverse)
    uint32_t layer, expert;
    uint32_t src_pool, src_slot;
    uint32_t dst_pool, dst_slot;
    // per-column pointers are derived from slot_col_mem() by the worker;
    // columns count/stride from the group topology (shared src/dst group).
    expert_scheduler* owner;         // to dispatch the completion back
    uint64_t req_tsc, done_tsc;      // [profile]
};
```

A ring of move tasks, mirroring the async-load ring buffer pattern.

### 5.4 Communication (two queues, mirrors DIO)

- **submit queue** (scheduler -> worker): move_tasks. Scheduler batches
  several victim moves at once (`submit_moves(tasks, n)`).
- **completion queue** (worker -> scheduler): finished tasks. Worker only
  memcpys then pushes the task back; the scheduler thread does the control
  plane tail: dst `mark_ready` + `dir_->set` (MOVING_IN -> READY) + version
  bump + wake + batch count; then the source slot `begin_reload()` (now free
  to take the new expert).

### 5.5 State dance for one v2r move

```
scheduler (device region full, needs slot for new expert X):
  pick victim V by (L,E) eviction (§6)
  begin_evict(src slot)                // READY -> EVICTING (blocks new try_pin)
  dir_->clear(V, src_pool)             // no longer pin-able
  while (src slot refcount != 0) yield // wait for any existing compute reader
                                       // (usually 0) - worker must never read
                                       // a slot a compute thread still holds
  entries_(V, src_pool)   : READY  -> MOVING_OUT
  dst = alloc_or_evict(RAM)           // dst IO_INFLIGHT (exclusive)
  entries_(V, RAM)        : ABSENT -> MOVING_IN
  enqueue move{V src, RAM dst}        // non-blocking; worker only sees the task
                                       // once src is EVICTING and refcount==0
  src slot stays EVICTING (NOT reloaded) until cq

worker: per-column memcpy(RAM dst, vram src); push completion

scheduler drain:
  dst slot mark_ready()               // IO_INFLIGHT -> READY
  entries_(V, RAM)     : MOVING_IN -> READY   (+ version bump + wake + batch count)
  src slot begin_reload()             // reuse for X
  entries_(V, src_pool): MOVING_OUT -> ABSENT (or reloaded as X)
```

r2v is the mirror; the worker code is direction-agnostic (swap src/dst).

### 5.6 Exclusivity: use slot states, NOT refcount

Eviction today drains refcount (`while (refcount != 0) yield`) waiting for
compute readers. If the move worker also took a refcount pin on the source
slot, that drain would self-deadlock (refcount is "compute-reader shared
count", not "migration lock"). Exclusivity comes from states:
source = EVICTING / MOVING_OUT, destination = IO_INFLIGHT / MOVING_IN - both
reject `try_pin` (READY required). No extra refcount on either slot.

## 6. Eviction keyed on (L,E) with layer-distance preference

### 6.1 Semantics

Eviction answers "this pool needs K free slots for a layer-L batch". Victims
are chosen from layers nearest to L (the just-finished layer is most likely
stale within a decode), then further layers, with the score threshold relaxed
by distance:

```
for delta = 1, 2, ...:
  for each (L-delta, e) resident in this pool:
     if slot READY && refcount==0 && score(e) <= threshold(delta): candidate
  pick the lowest-score candidate from the nearest layer that yields one
```

Worked example (as decided): need K slots -> take top K/2 lowest-score
resident experts of layer L-1, top K/2 of L-2, ... (score = low evicts first;
score-line estimation deferred, see §8).

**Cross-layer score comparison caveat**: `adaptive_scores_` distributions may
differ per layer (some layers are globally hot). NEVER compare raw score
values ACROSS layers. The worked example already avoids this by ranking
WITHIN a layer and taking top K/2 per layer - keep that. If a later change
mixes layers in one ranking, normalise/rank within each layer first (see §8).

### 6.2 How to enumerate (L-delta) experts resident in this pool

Two options:

- **A: per-layer scan** - for each expert e in layer L-delta, query
  `entries_[(L-delta, e)][pool]`; if resident -> check slot refcount + score.
  Cost O(n_expert) per layer scanned (gemma 128 / deepseek 256), and we only
  scan a few near layers. Simple, no extra index.
- **B: per-(pool, layer) residency list** - maintained incrementally. Faster
  eviction but pays bookkeeping on every set/clear.

Decision: **A** for now (few near layers, ~256 queries each, negligible). B is
a later optimisation if eviction shows up in profiles.

### 6.3 Batch planning vs per-expert alloc

Current `accept_requests` calls `alloc_or_evict` per expert. The (L,E) layer-
distance scheme naturally wants **batch planning**: on receiving a layer-L
batch, count how many new slots the pool needs, and evict K victims in one
planning pass (top K/2 from L-1, etc.) before loading. This is the intended
final shape; the per-expert `alloc_or_evict` wrapper can remain as the
single-slot primitive underneath.

## 7. Scheduler side: per-model single active request slot

Decided: batch progress accounting lives on the scheduler thread; exec wakes
ONCE when its whole request is READY. The scheduler serialises exec requests to
**one active request per model** - there is at most one exec request being
served per `expert_scheduler` at a time (the scheduler is already per-model,
MULTI_MODEL_POOL). This makes "who is waiting" a single object per model, no
waiter lists needed.

### 7.1 The active-slot discipline

```
scheduler loop (global worker polls each per-model scheduler; models don't block
each other - each has its own active slot):
  per model:
    if model.active empty:
      pop one exec request -> active  (records still-need bitmap + n_load_target)
      for each still-need (L,E):
        ABSENT                        -> publish LOADING, start own load
        LOADING / MOVING_IN (in flight, e.g. prefetch or a prior race) -> do
                                          NOT start a second load; it is already
                                          someone else's in-flight load, which
                                          will reach READY through drain
    else:
      keep draining (active in progress); do not pop a new request
  drain: every (L,E) that settles READY (own load, prefetch, or move):
      mark_ready + state -> READY
      if (L,E) is in model.active.still-need: bump active.counter (once)
      profile delta rdtsc (future, parallel - not the completion's only job)
  active.counter == 0 -> wake exec once -> exec re-pins -> active cleared
```

Because drain is the single point where any slot becomes READY, an in-flight
expert that belongs to the active request (loaded by prefetch or a raced prior
submit) bumps the active counter exactly when it actually becomes READY. No
"skip counts as done" anywhere, and no per-(L,E) waiter list - the active
request is the only waiter for this model.

### 7.2 Prefetch is NOT a separate half-event - it reuses the full DIO path

A prefetch (scheduler-initiated predictive load) is just another submit on the
same `async_dio_engine`; its completion lands in the SAME `drain_completions`
and performs the full settle (mark_ready + dir set + version bump + wake), plus
the drain-level profile delta-rdtsc accounting as a parallel extra. It is never
"a separate event with no lower half that only records rdtsc". Consequences:

- a prefetch must also publish `dir = LOADING` before submitting (so exec's
  scan sees "in flight", does not double-load; §3.4 ordering applies);
- a prefetched expert that settles while belonging to the active request bumps
  the active counter through the same drain rule - no special prefetch->active
  notification channel is needed.

### 7.3 Retry-round reset (exec side)

When `pin_layer` retries, the re-submitted `n_load_target` MUST equal the
still-need count re-measured by that round's scan - never the round-1 total.
The scheduler counts down the round's own n_load_target.

## 8. Open questions / deferred

1. **Score-line estimation**: "given K slots needed, what score threshold for
   layer L-delta?" - deferred (user: can be done later). Start with "top K/2
   per layer by score, nearest layer first" and refine. Cross-layer raw-score
   comparison stays forbidden (see §6.1 caveat).
2. **A1 vs A2** directory layout: A1 (two parallel atomics) recommended.
3. Whether `dir_->set` semantics stay "only at READY" or move to "reserve =
   LOADING, READY = READY" fully (see §3.4 ordering).
4. **Single-active boundary**: the §7 model serialises exec requests per model
   (one active slot). This matches today's single-ctx single-decode-thread
   reality (exec blocks until its layer's request completes). If the same model
   later runs multiple concurrent decode contexts, multiple active slots (one
   per ctx) or a waiter set would be needed - flagged, not built now.

> Resolved earlier in this session: move worker count = 1 (see §5.3). Exec is
> stateless (see §3.5); prefetch reuses the full DIO path (see §7.2); in-flight
> duplicates need no special handling beyond the active-slot + drain rule (§7).

## 9. Build order (final shape, component by component)

1. Directory: widen to (L,E)-keyed state (A1); keep owner_ until step 3 compiles. **[done - M1, 6b6300e]**
2. Load path: publish LOADING before begin_reload (ordering invariant); single
   active request slot per model (removes duplicate-load window; §7.1).
   **[done for the LOADING-before + state-aware accept side - M2, dbaff8f;
   single-active-slot NOT done - it is part of M5]**
3. Eviction: (L,E)-keyed layer-distance selection inside alloc_or_evict; delete
   owner_ (8 uses, all in alloc_or_evict, already enumerated). **[done - M3, 45b14de]**
4. Move pipeline: move_task ring + 1 worker (v2r + r2v) + completion drain;
   wire v2r into eviction of a device victim; r2v available for placement
   policy later. **[done - M4, fdf4982; v2r now DMA per §5.2 revision - 717bac8]**
5. Scheduler-side accounting: active-slot counter, drain-unified bump, wake-once
   (§7); exec side becomes stateless pin+resubmit (§3.5). **[NOT done - M5 open]**
6. Verify: pure-RAM IDENTICAL (as today); VRAM single-device run reaches
   `exec_mm_vk` (this is the K6/L6b blocker); UT for directory states + move
   ring + eviction ordering. **[partial - M6: pure-RAM path unchanged (RAM sources
   have dev_buf==null -> memcpy), DMA content gate 0 BAD, VRAM demote storms
   eliminated; exec_mm_vk arrival + state/move UT still open]**

## 10. Files touched

- `src/backend/slot.h` - directory state widen (A1) or new move-task structs
- `src/backend/scheduler.h` / `scheduler.cpp` - eviction, owner_ removal,
  move worker + rings, batch accounting
- `src/pool/expert_stats.h` - review only: scoring already keyed by (L,E)
  (`stats_.adaptive_scores_[(L,E)]`); (L,E)-keyed eviction reads it directly,
  no slot reverse-lookup needed
- `src/backend/minigraph_exec.cpp` - none expected (exec API unchanged: still
  pin_layer), except batch-await plumbing if signals move scheduler-side
- `tests/test_slot.cpp`, `tests/test_scheduler.cpp` - new state/move/eviction UT
- `docs/WORK_IN_PROGRESS.md` - task tracking (new section M)

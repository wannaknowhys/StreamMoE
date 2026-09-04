# Expert Move Pipeline + (L,E)-Keyed Eviction - Design

[English](EXPERT_MOVE_PIPELINE.md) | [简体中文](EXPERT_MOVE_PIPELINE.zh-CN.md)

> Status: **design, 2026-09** - a working session decided the next scheduler
> redesign. The landed batch-pin layer (slot_request bitmap / wake-once, see
> WORK_IN_PROGRESS L) is the base this builds on. This doc captures the FULL
> design so nothing is lost between sessions.
> Related: `docs/WORK_IN_PROGRESS.md` L / J, `docs/Backend.md` §2/§4,
> `docs/EXPERT_SCHEDULER_DESIGN.md`, `src/backend/scheduler.{h,cpp}`,
> `src/backend/slot.h`.

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

### 3.5 Compute-side pin semantics under the state table

`pin_layer` scans by (L,E):

- first READY copy -> pin it (unchanged);
- a LOADING / MOVING_IN copy -> it is in flight - do NOT pin and do NOT submit
  a request for it; wait on that (L,E)'s version word and rescan (see §7 - this
  is decision 6a: exec never submits in-flight experts);
- all ABSENT -> submit (the expert is genuinely not being loaded anywhere);
- FAILED -> error.

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

### 5.2 Why not GPU/vulkan DMA for v2r

`vkCmdCopyBuffer` / copy engines can copy device<->device and
device<->host-visible-vulkan-buffer. On a discrete GPU (RX 590) the
host-visible heap is still VRAM (rebar BAR1); the 128GB system-RAM pool is NOT
in the GPU address space, so a copy engine cannot write it. A real
`vkCmdCopyBuffer` vram->system-RAM target does not exist on discrete GPUs
(only on UMA/APU). Cross-backend `ggml_backend_tensor_copy` likewise falls back
to memcpy or requires a host-visible target buffer that would occupy VRAM.
Conclusion: v2r to ordinary malloc RAM is CPU-memcpy-only on this hardware.
Asynchronous CPU memcpy workers are the fix; GPU copy-engine DMA is noted for
a future UMA/APU target.

### 5.3 Task shape (move_task)

One dedicated copy worker (parameterised count, start at 1). Single task per
expert move; submit in batches. Worker does pure per-column memcpy, never
touches control plane.

Why one worker (not a pool): the v2r bottleneck is **PCIe read bandwidth**
(vram host-map -> RAM), not CPU core count - extra workers do not add bandwidth,
they only add concurrency. One worker saturates the link; the count stays a
parameter in case a faster link or UMA target changes that calculus.

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

## 7. Batch progress accounting on the scheduler side

Decided: the "how many of this batch are still loading" bookkeeping belongs on
the scheduler thread (it processes each bit anyway). Exec should wake ONCE when
the whole batch is READY, not N times to compare a counter it does not own.

**Completion semantic (single rule, do not split)**: the batch's wake signal
fires only when EVERY needed (L,E) is truly READY. `remaining` decrements only
on a real READY, never "on skip". Concretely, for each needed bit the scheduler
encounters:

- **already READY** (raced since exec scanned) -> decrement immediately (exec
  can pin it - it really is READY);
- **LOADING / MOVING_IN** (in flight) -> do NOT decrement here; it is exec-side
  decision 6a that exec never submits in-flight experts, so this only happens
  for a race window (exec saw ABSENT, then another submit started loading it).
  The scheduler does not start a second load; the expert's existing load/move
  will reach READY and decrement through the single drain completion point.
- **ABSENT** -> start the load; it decrements when drain marks it READY.

The decrement therefore always happens at the ONE place a slot becomes READY
(drain_completions mark_ready + state -> READY). A batch's counter reaches zero
only when every bit has been observed READY - exec can then pin every handle it
asked for. No "skip counts as done" anywhere.

**Retry-round reset (exec side)**: when `pin_layer` retries (round 2), the
re-submitted `slot_request_t.n_load_target` MUST equal the still-missing count
re-measured by the second scan - NOT the round-1 total. The scheduler counts
down the round's own n_load_target.

## 8. Open questions / deferred

1. **Score-line estimation**: "given K slots needed, what score threshold for
   layer L-delta?" - deferred (user: can be done later). Start with "top K/2
   per layer by score, nearest layer first" and refine. Cross-layer raw-score
   comparison stays forbidden (see §6.1 caveat).
2. **A1 vs A2** directory layout: A1 (two parallel atomics) recommended.
3. Whether `dir_->set` semantics stay "only at READY" or move to "reserve =
   LOADING, READY = READY" fully (see §3.4 ordering).

> Resolved: in-flight duplicate requests (decision 6a - exec never submits
> in-flight experts; LOADING/MOVING_IN experts are waited on via version word,
> see §3.5), and move worker count = 1 (see §5.3). Both were earlier open items
> and are now settled in their sections.

## 9. Build order (final shape, component by component)

1. Directory: widen to (L,E)-keyed state (A1); keep owner_ until step 3 compiles.
2. Load path: publish LOADING before begin_reload (ordering invariant); add
   in-flight skip in accept (removes duplicate-load window).
3. Eviction: (L,E)-keyed layer-distance selection inside alloc_or_evict; delete
   owner_ (8 uses, all in alloc_or_evict, already enumerated).
4. Move pipeline: move_task ring + 1 worker (v2r + r2v) + completion drain;
   wire v2r into eviction of a device victim; r2v available for placement
   policy later.
5. Batch progress accounting on the scheduler side (wake once).
6. Verify: pure-RAM IDENTICAL (as today); VRAM single-device run reaches
   `exec_mm_vk` (this is the K6/L6b blocker); UT for directory states + move
   ring + eviction ordering.

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

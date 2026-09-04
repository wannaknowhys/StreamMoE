# b4-3 arena-clone Experiment Archive (2026-09)

Dead-code archive for the M2-C2b4 b4-3 arena-clone whole-layer device executor
experiment line. The experiment was retired after root-causing the gate_up mm
NaN@2348 to a ggml-vulkan MUL_MAT_ID incompatibility with the route-B compact
slot layout (w3d shell `ne[2]=n_slots / nb[2]=expert_size`): vulkan hardcodes the
per-expert stride to `ne0*ne1` and ignores `nb[2]`, while route-B slots stride by
`expert_size` (one slot = one (layer,expert) full weight set). CPU mul_mat_id
reads `nb02` and is correct; vulkan never was. The fix is a layout change
(per-tensor pools), not an arena change - so the whole arena-clone executor is
dead weight.

Restore (if ever needed):

    git apply patches/<file>.patch        # cumulative, or
    git am commits/<file>.patch           # per-commit (keeps history/messages)

Files:

- `b43_arena_clone_cumulative.patch` - net diff `860f9f4..HEAD` over the touched
  files (minigraph_exec.cpp, route_b_chain.*, scheduler.cpp/.h, build.bat,
  docs/WORK_IN_PROGRESS.md).
- `commits/0001..0006.patch` - the six b4-3 commits as individual patches.

Retired 2026-09: HEAD reset to `860f9f4` (pre-b4-3 clean baseline). v1
(exec_mm_vk single-node vulkan mm + exec_one_burst CPU burst) is intact at the
baseline; device-domain vulkan mm is not numerically correct on the slot layout.

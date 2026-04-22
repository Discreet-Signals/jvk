# CommandScheduler — Implementation Plan

A post-process pass over the deferred draw-command buffer that rewrites it for
optimal dispatch: **reorders** commands to cluster same-pipeline work, **culls**
commands hidden behind later opaques, and **fuses** compatible commands into
single batched dispatches where the pipeline supports it.

Models the command stream as an IR, builds a dataflow DAG over its read/write
regions, and list-schedules that DAG with fusion-aware priority. Uses the
existing clip-scope stack (which mirrors the JUCE component tree in practice)
as a hierarchical bbox structure so overlap queries prune by subtree.

Runs on the existing render **worker thread** inside `Renderer::execute()`,
between `flushToGPU()` and the pipeline dispatch loop. Recording cost on the
paint thread is unchanged.

---

## Design invariants

These must hold no matter what scheduler mode is active:

1. **Observable equivalence to record order.** For every pixel `p`, the final
   colour written by the rewritten command list equals the colour that record
   order would produce. Scheduler legality is defined entirely by this.

2. **Hard barriers partition the stream.** `PushClipPath`, `PopClip`, and any
   op that modifies stencil state are barriers. No command moves across them.
   The scheduler processes each barrier-delimited region independently.

3. **Scope-local ordering.** Within a scope, commands retain record-order
   relative to any other command whose `writesPx` overlaps their `readsPx` (or
   vice versa). That's the only ordering constraint — everything else is free.

4. **Subtree spatial invariant.** Every command inside scope `S` writes only
   to pixels in `scopeBounds[S]`. A command inside scope `T`, disjoint from
   `S`, can never overlap any command inside `S`. This is the pruning lever.

---

## Data model

### `DrawCmd` (existing struct, extended)

Add these fields. The existing `zOrder`, `clipBounds`, `stencilDepth`,
`scopeDepth`, `dataOffset`, `op` stay unchanged.

```cpp
struct DrawCmd {
    // ... existing fields ...

    // Tight physical-pixel write region — where this command actually
    // modifies the scene buffer. Smaller than clipBounds for most ops;
    // equal to clipBounds for fullscreen effects until they gain a scissor.
    juce::Rectangle<int> writesPx;

    // Tight physical-pixel read region. For non-effect ops: equals writesPx.
    // For blurs: writesPx dilated by (maxRadius + falloff). For HSV: equals
    // writesPx (reads its own output region).
    juce::Rectangle<int> readsPx;

    // 64-bit hash combining everything that must match for two commands to
    // share a batched dispatch. Encoded as:
    //   [ 8b op | 24b pipelineId | 16b resourceKey | 16b blendKey ]
    // Two commands with equal stateKey are candidates for fusion; legality
    // still requires pairwise DAG independence.
    uint64_t stateKey;

    // Monotonically increasing per clip-push within a frame. Maps to the
    // scope entry this command was recorded inside. Used to look up
    // scopeBounds / scopeSubtreeBounds during DAG build.
    uint32_t scopeId;

    // Set by the pipeline at record time. True iff this command produces
    // alpha=1 output everywhere inside writesPx (so a later overlap
    // entirely covers it → eligible for cull). Conservative: false when
    // uncertain (AA edges, gradients with alpha stops, textures).
    bool isOpaque;

    // Original record index. Used as stable tiebreaker in scheduler heuristics
    // and for debug dumps.
    uint32_t recordOrder;
};
```

### `Scope` (new, one per `PushClipRect` / `PushClipPath`)

Stored in a flat `std::vector<Scope> scopes_` on the Renderer, indexed by
`scopeId`. Scope 0 is the implicit root (full viewport).

```cpp
struct Scope {
    uint32_t             parentId;        // 0 for root
    juce::Rectangle<int> bounds;          // the clip rect/bbox at this scope
    juce::Rectangle<int> subtreeBounds;   // ⋃ bboxes of cmds inside this
                                          // scope + all descendants. Set in a
                                          // single bottom-up pass after
                                          // recording ends.
};
```

### `CommandScheduler` (new)

Lives at `modules/jvk/include/jvk/core/CommandScheduler.h`.

```cpp
namespace jvk {

class CommandScheduler {
public:
    enum class Mode : uint32_t {
        Identity  = 0,  // passthrough — used as baseline for bisection
        SortOnly  = 1,  // DAG build + list schedule, no fusion, no cull
        SortCull  = 2,  // SortOnly + opaque-overdraw DCE pass
        Fuse      = 3,  // SortCull + fusion for opted-in pipelines
    };

    void setMode(Mode m) { mode_.store(m, std::memory_order_relaxed); }
    Mode getMode() const { return mode_.load(std::memory_order_relaxed); }

    // Rewrites cmds in place. Reads scopes[] for bounds queries. May shrink
    // the list (cull) or reorder it (sort/fuse). Emits fused commands by
    // packing their members into arena-allocated primitive arrays referenced
    // by the new command's dataOffset.
    void run(std::vector<DrawCmd>& cmds,
             std::vector<Scope>&   scopes,
             Arena&                arena);

private:
    std::atomic<Mode> mode_{Mode::Identity};
    // ... working buffers reused across frames (DAG nodes, ready queue) ...
};

} // namespace jvk
```

---

## Step 1 — Wire the metadata into every draw command

**Goal.** Extend `DrawCmd` and the push path so every recorded command carries
`writesPx`, `readsPx`, `stateKey`, `scopeId`, `isOpaque`, and `recordOrder`.
Populate `scopes_` with bounds; compute `subtreeBounds` at end-of-frame. No
scheduler yet — the command list goes straight to the renderer as today.

**Acceptance.** Build + run, nothing visually changes, no perf change. The
bytes-per-command grow ~32 B; at ~5k commands/frame that's 160 KB, fine.

### Files changed

- `modules/jvk/include/jvk/core/DrawCmd.h` (or wherever `DrawCmd` lives) —
  add the new fields.
- `modules/jvk/include/jvk/core/Scope.h` (new) — declare `Scope`.
- `modules/jvk/include/jvk/core/Renderer.h` — add `std::vector<Scope> scopes_`,
  `uint32_t currentScopeId_`, helpers `pushScope(bounds)` / `popScope()`,
  accessor `currentScopeId() const`.
- `modules/jvk/include/jvk/graphics/Graphics.h` — each push-site needs to
  populate the new fields. Clip operations (`clipToRectangle`, `clipToPath`)
  call `renderer_.pushScope(clipBoundsPx)` after the clip computation;
  `restoreState` calls `renderer_.popScope()`.
- Every pipeline's `push*()` call sites (FillRect, FillRoundedRect,
  DrawImage, FillPath, BlurShape, BlurPath, DrawShader, DrawGlyphs, DrawLine,
  Stroke*, HSV, etc.) need to compute and pass `writesPx`, `readsPx`, and
  `isOpaque`. For most this is a one-liner; for glyphs and paths it requires
  the bounding box of the flattened geometry.

### `stateKey` encoding

Centralised in a `StateKeyBuilder` helper. Fields are computed once per push:

```cpp
uint64_t k = 0;
k |= uint64_t(op)         & 0xFF;                    // 8b  op class
k |= (uint64_t(pipelineId) & 0xFFFFFF) << 8;         // 24b pipeline
k |= (uint64_t(resourceKey) & 0xFFFF)  << 32;        // 16b fill/texture hash
k |= (uint64_t(blendKey) & 0xFFFF)     << 48;        // 16b blend/edge/mode
```

`resourceKey` for a fill is `fillIndex`; for an image it's `imageHash`; for a
blur it's `hash(mode, maxRadius, falloff, edge, inverted)`. When in doubt,
include it — over-splitting is correctness-preserving, under-splitting isn't.

### `readsPx` rules per op

| Op                    | readsPx                                   |
|-----------------------|-------------------------------------------|
| FillRect, DrawImage   | writesPx                                  |
| FillPath              | writesPx (reads dest for blend, but that's pixel-local) |
| DrawGlyphs, DrawLine  | writesPx                                  |
| BlurShape, BlurPath   | writesPx dilated by (maxRadius + falloff) |
| HSV                   | writesPx                                  |
| DrawShader            | writesPx (shaders never sample the scene) |
| PushClip*, PopClip    | writesPx = their cover rect               |

### `isOpaque` rules per op

Set true only when:

- The op is `FillRect`, `FillRoundedRect` (with inset AA, not outline),
  `DrawImage` (only if the image is known-opaque — PNGs often aren't), or a
  rectangular path.
- The fill has alpha==1 and no gradient stops with alpha<1.
- Blend mode is the default source-over (which at alpha=1 is pure replace).

For everything else: false. The cull pass needs conservative truth.

### Computing `subtreeBounds`

One pass at end-of-frame on the paint thread, before handoff to the worker:

```cpp
// After record ends, before submit():
for (auto& cmd : cmds)
    scopes[cmd.scopeId].subtreeBounds.enlargeIfRequired(cmd.writesPx);

// Propagate up the tree (children into parents).
// scopes are pushed in order, so iterating in reverse guarantees children
// are processed before their parent.
for (size_t i = scopes.size(); i-- > 1; )
    scopes[scopes[i].parentId].subtreeBounds
        .enlargeIfRequired(scopes[i].subtreeBounds);
```

O(N_cmds + N_scopes), trivial.

### Testing

- No visual regression across the full plugin UI.
- Assert that for every command, `writesPx ⊆ scopes[cmd.scopeId].bounds`.
- Assert `isOpaque == false` for anything with alpha < 1.

---

## Step 2 — Wire up `CommandScheduler` with mode switch and debug UI

**Goal.** Instantiate the scheduler, run it on the worker thread in Identity
mode, confirm byte-for-byte identical output. Add an on-screen button that
cycles `Mode` so modes can be compared live.

**Acceptance.** Build + run; Identity mode is the default; flipping to any
other mode (currently stubs that also passthrough) changes nothing. A visible
on-screen indicator shows the current mode.

### Files changed

- `modules/jvk/include/jvk/core/CommandScheduler.h` (new) — class skeleton
  with `Mode`, `setMode`, `run`. All modes `Identity` through `Fuse` just
  passthrough in this step.
- `modules/jvk/include/jvk/core/Renderer.h` — own a `CommandScheduler
  scheduler_`, expose `commandScheduler()` accessor.
- `modules/jvk/include/jvk/core/Renderer.cpp` — inside `execute()`, after
  `flushToGPU` and before the dispatch loop, call
  `scheduler_.run(commands_, scopes_, arena_)`.
- `source/components/ui/` — add a debug button (hidden by default, or behind
  a build flag) that cycles the scheduler mode. Label it with the current
  mode. Wire to `editor.vulkanWindow()->renderer().commandScheduler().setMode(...)`.

### Placement in `execute()`

```cpp
void Renderer::execute()
{
    auto frame = target_.beginFrame();
    if (frame.cmd == VK_NULL_HANDLE) return;

    // ... existing prologue: flushRetired, prepare, gradient/texture upload,
    // pathPipeline_->flushToGPU, flushUploads ...

    scheduler_.run(commands_, scopes_, arena_);   // <-- NEW

    // ... existing dispatch loop ...
}
```

The scheduler runs **before** any pipeline-specific uploads that assume command
order is stable (paths are already uploaded by `pathPipeline_->flushToGPU()`
above — that's fine, segment offsets stay valid because the scheduler rewrites
the command list but not the SSBO contents).

### Debug UI

Simplest form: a small `juce::TextButton` in a corner, label `"Mode: Identity"`,
`onClick` cycles through `Identity → SortOnly → SortCull → Fuse → Identity`.
The button lives in the plugin editor and calls into the renderer. Safe
because `setMode` is lock-free and the worker reads it once per frame at the
top of `run()`.

### Testing

- Mode = Identity: exact same visual output as before Step 1.
- Mode = anything else: still exact same output in this step (they're all
  passthrough stubs).
- Button cycles visibly; UI label updates.

---

## Step 3 — Implement SortOnly: DAG build + list scheduler, no fusion

**Goal.** `Mode::SortOnly` builds a dataflow DAG from read/write regions and
list-schedules it with a priority that prefers same-stateKey adjacency. No
fusion, no cull. Output is a reordered command list with identical pixel
output.

**Acceptance.** Build + run. Mode = SortOnly produces identical pixels to
Mode = Identity across every screen in the plugin. Same-stateKey runs are
visibly longer in a debug dump. No perf regression (no wins yet either —
dispatch is still one-command-at-a-time).

### Algorithm

For each barrier-delimited region:

1. **Build the DAG.** For each command `C` in record order, compute deps:

   ```cpp
   for (uint32_t i = firstCmdInRegion; i < C.recordOrder; ++i) {
       auto& P = cmds[i];
       bool rawHazard = P.writesPx.intersects(C.readsPx);
       bool wawHazard = P.writesPx.intersects(C.writesPx);
       if (rawHazard || wawHazard)
           dag.addEdge(P → C);
   }
   ```

   Optimisation: use the subtree pruning. Walk scope ancestors of `C`; for
   each, iterate sibling scopes whose `subtreeBounds` intersects `C.readsPx`
   and only test commands within those. Most scope subtrees will be pruned.

2. **Initialise ready set.** All commands with `depCount == 0` go in.

3. **Pick next command.** From the ready set, pick by priority:

   1. Same `stateKey` as last-emitted command (keeps pipeline hot).
   2. Smallest `recordOrder` among same-stateKey candidates (stable order
      → reproducible bugs).
   3. If no same-stateKey match, pick smallest `recordOrder` overall.

4. **Emit + update.** Append picked command to output list. Decrement
   `depCount` on all its dependents; those that hit 0 enter the ready set.

5. **Repeat** until all commands emitted. Assert empty ready set and all
   commands consumed.

### Data structures

```cpp
struct DagNode {
    uint32_t cmdIndex;
    uint32_t depCount;
    uint32_t firstDependent;  // index into dependents_ pool
    uint16_t dependentCount;
};

std::vector<DagNode>   nodes_;
std::vector<uint32_t>  dependents_;  // flat CSR-style pool
std::vector<uint32_t>  readyQueue_;  // priority queue by stateKey match
```

Reuse these vectors across frames — clear, don't reallocate.

### Complexity

DAG build: O(N²) naive. With subtree pruning it's closer to O(N · log N) in
practice for UI trees (most commands in disjoint subtrees). If profiling
shows DAG build dominates, add a tile-grid index (e.g. 64×64 px) to accelerate
the overlap test — each tile stores a small vector of command indices that
touch it; `C`'s overlap candidates are the union of commands in the tiles
`C.readsPx` covers.

### Testing

- **Pixel-diff test**: render the same frame in Identity mode, then SortOnly
  mode, compare swapchain output. Must be byte-identical.
- **Stress test**: worst-case UI — all pedals open, modulation matrix
  visible, backdrop blur active. Profile DAG-build time; verify sub-ms on M1.
- **Barrier test**: confirm PushClip/PopClip partition correctly — no command
  crosses one.

### What to watch for

- **Transparent ops over transparent ops.** Two semi-transparent fills
  overlapping must preserve record order. The WAR hazard check via bbox
  intersection handles this automatically — don't "optimise" it away.
- **Effect ops look like write-everywhere unless you already have tight
  `readsPx`.** If blurs still read "whole viewport" they'll chain-depend on
  everything and SortOnly degenerates to Identity. Tightening blur readsPx
  is strictly a Step 1 concern but it's the thing that makes Step 3 interesting.

---

## Step 4 — Implement SortCull: opaque-overdraw dead-code elimination

**Goal.** Add a DCE pass that deletes commands whose entire `writesPx` is
covered by a later opaque command within the same or outer scope, where no
op between them reads the covered region.

**Acceptance.** Build + run. Mode = SortCull produces identical pixels to
SortOnly. In high-overdraw screens (e.g., a big opaque backdrop behind many
small ops that are fully covered), command count drops. Optional: add a debug
HUD showing `culled/total` commands per frame.

### Algorithm

Runs **after** DAG build but **before** scheduling:

```cpp
// Walk commands in reverse record order.
// Maintain a "covered" RectangleList — pixels already guaranteed to be
// painted by a later opaque op.
juce::RectangleList<int> covered;

for (int i = cmds.size() - 1; i >= 0; --i) {
    auto& C = cmds[i];

    // If C's writesPx is entirely inside `covered` AND no op in (i, end)
    // reads it, C is dead.
    bool fullyCovered = covered.containsRectangle(C.writesPx);
    bool readByLater  = /* see below */;

    if (fullyCovered && !readByLater) {
        C.op = DrawOp::Noop;   // mark; don't erase yet — DAG refs stable
        continue;
    }

    // Extend covered set if C is opaque.
    if (C.isOpaque)
        covered.addWithoutMerging(C.writesPx);

    // Any op that READS a region invalidates it from `covered` because the
    // read consumes the current state. Simplest correct rule: after any
    // non-opaque op or any effect, clear `covered` of its readsPx.
    if (!C.isOpaque)
        covered.subtract(C.readsPx);
}

// Scheduler then skips noop'd commands during emit.
```

Important: **the `readByLater` check is a proxy**, not a literal pass over
subsequent ops. Easier correct formulation:

- An effect op (`BlurShape`, `BlurPath`, `HSV`) is a strong barrier for DCE —
  it reads the scene, so anything it overlaps must be preserved. Model this
  by pre-seeding `covered` to empty at every effect op as we walk backward
  past it (i.e. the effect "contaminates" older pixels, they can't be DCE'd
  across it).

### Files changed

- `modules/jvk/include/jvk/core/CommandScheduler.cpp` — add `cullOpaqueOverdraw()`
  phase before `listSchedule()`.

### Why this is small but real

In the plugin UI today, every frame paints the full backdrop image, then a
gradient, then every pedal's background, then every knob's background, then
the knobs themselves. The backdrop and gradient are fully covered — they
can't be DCE'd because the gradient has transparent edges. But for dense
overdraw regions (e.g., the presets menu background behind the list), DCE
often wins.

### Testing

- Pixel-identical to SortOnly.
- Add a counter; confirm it's > 0 on overdraw-heavy screens.
- Hand-construct a test case: opaque rect over smaller rect, smaller rect
  should be DCE'd.

---

## Step 5 — Implement Fuse: batched pipelines, incrementally

**Goal.** For each pipeline that opts in (`supportsFusion = true`,
`maxBatchSize = N`), the scheduler fuses eligible same-stateKey ready-set
members into one batched dispatch. Roll out one pipeline at a time.

**Acceptance.** For each batched pipeline landing, measured paint-thread +
worker-thread time on representative screens. No visual regression.

### Fusion rule

At ready-set emit time, instead of picking one command:

1. Take all commands in the ready set with the same `stateKey` as the one
   we'd pick.
2. Prune to those that are *pairwise independent* in the DAG. Two candidates
   `A`, `B` are pairwise independent iff neither transitively depends on the
   other. Precompute per-node an index into a reachability structure; for
   small DAGs (N < 1000) an O(N²) transitive closure bitmap is fine.
3. Clamp to `pipeline.maxBatchSize`.
4. Emit one fused command whose `dataOffset` points to an arena-allocated
   primitive array containing each member's per-primitive params.
5. Mark all fused members as emitted.

### Pipeline changes

Each pipeline that wants to batch:

```cpp
class Pipeline {
public:
    virtual bool     supportsFusion() const { return false; }
    virtual uint32_t maxBatchSize()    const { return 1; }
    virtual void     executeBatch(Renderer&, Arena&,
                                  std::span<const DrawCmd>) {
        // default: per-command fallback
        for (auto& c : cmds) execute(r, a, c);
    }
};
```

Scheduler calls `executeBatch` when it emits a fused command; pipelines that
don't override still work via the fallback.

### Roll-out order

1. **`FillRectPipeline`.** Simplest win. Per-primitive SSBO with `{rect,
   colorOrGradientIndex, transform}`. Shader draws N instanced quads keyed by
   `gl_InstanceIndex`. `maxBatchSize = 1024`. Land this, measure.

2. **`FillRoundedRectPipeline` + `FillEllipsePipeline`.** Same pattern, add
   a primitive type discriminator per instance. Candidate to merge into one
   "SDF primitive" pipeline already, since the fragment shader is a thin SDF
   switch.

3. **`FillPathPipeline`.** Already uses a segment SSBO. Fusion here means
   multiple paths share the SSBO via `(segmentStart, segmentCount, fillRule,
   fillIndex)` per primitive in the batch. The fragment shader unions their
   SDFs by min().

4. **`ShapeBlurPipeline` + `PathBlurPipeline`.** This is where fusion pays
   off most — each fused dispatch collapses N ping-pong passes into 1 (or 2
   for Low mode). Requires the super-shader you mentioned: takes both a
   primitive array (rect/rrect/ellipse SDFs) and a path-segment array, with
   per-primitive `(bbox, maxRadius, falloff, edge, inverted, mode, segRange,
   ...)`. Fragment iterates primitives, picks the one whose bbox contains
   the fragment (or argmin-SDF for multi-coverage), computes `t` from that
   primitive's SDF + params, runs the kernel.

5. **Text (`DrawGlyphsPipeline`).** Glyphs already batch within one draw
   call today; fusion here means merging multiple `DrawGlyphs` calls that
   share a font + fill. Smaller win but easy.

6. **`DrawImagePipeline`.** Needs bindless texture array for true batching
   across distinct images. Can skip until there's a reason.

### The fused blur super-shader sketch

Push constants:

```glsl
layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  direction;          // (1,0)/(0,1) for Low, unused otherwise
    uint  primitiveCount;
    int   kernelType;         // 0/1/2 — all primitives in a batch share this
} pc;
```

SSBO of primitives:

```glsl
struct Primitive {
    // Bbox in physical pixels — fragment uses this to skip primitives
    // whose band doesn't reach it.
    vec4  bboxMinMax;           // xy = min, zw = max

    // Blur params — per-primitive.
    float maxRadius, falloff, blurStep;
    int   edgePlacement;
    int   inverted;

    // Shape discriminator + geometry. 0=rect, 1=rrect, 2=ellipse,
    // 3=capsule, 4=path.
    int   shapeType;
    vec2  shapeHalf;            // for primitives
    vec2  lineB;                // capsule only
    float cornerRadius;
    float lineThickness;

    // Affine inverse (3x2 matrix packed as three vec2s) for primitives.
    vec2  invCol0, invCol1, invCol2;

    // Path data range (shapeType==4 only).
    uint  segmentStart, segmentCount, fillRule;
};
```

Fragment algorithm:

```glsl
float tMax = 0.0;
float effectiveRadius = 0.0;

for (uint i = 0; i < pc.primitiveCount; ++i) {
    Primitive P = prims[i];

    // Early-out by bbox + read footprint.
    vec2 c = fragCoord;
    float reach = P.maxRadius + P.falloff;
    if (c.x < P.bboxMinMax.x - reach || c.x > P.bboxMinMax.z + reach) continue;
    if (c.y < P.bboxMinMax.y - reach || c.y > P.bboxMinMax.w + reach) continue;

    float t = computeT(P, c);        // shape SDF + band smoothstep
    if (t > tMax) {
        tMax = t;
        effectiveRadius = P.maxRadius * t * P.blurStep;
    }
}

if (effectiveRadius <= 0.5) { outColor = texture(srcTexture, fragUV); return; }

// Run the chosen kernel with effectiveRadius — same as single-primitive case.
```

Two design choices:

- **Argmax of `t`** (shown above) is cheap and usually correct. If two
  overlapping blurs have very different radii, the larger `t` wins; the
  smaller one is masked. Documented behaviour; user can just not overlap them
  with different params.
- **Sum or max radius** is an alternative if users want blurs to compound.
  Punt on this — argmax is the MVP.

### Testing each pipeline landing

- Pixel-identical to SortCull for a single-primitive batch (batch of 1 must
  equal singleton dispatch).
- Pixel-identical across batch sizes for independent same-stateKey primitives
  (e.g., 10 non-overlapping fills in Fuse mode equals 10 singleton dispatches).
- Perf: measure dispatch count per frame; confirm it drops.

---

## Open questions to revisit later

- **Reachability structure scaling.** If the DAG ever grows beyond ~5k
  nodes per barrier region, the O(N²) transitive closure bitmap becomes
  hot. Options: segment tree, or a scoped-reachability approximation where
  we only test pairwise independence within the current ready set (O(R²),
  R typically < 30).

- **Cross-barrier fusion.** A `PushClip` barrier technically prevents
  fusion across it, but in practice many clips are `clipToRectangle` fast-
  paths that don't touch the stencil. We could split `PushClip` into two
  ops — `PushScissorClip` (non-stencil, not a fusion barrier) and
  `PushStencilClip` (barrier). Defer until profiled.

- **Reading from the cull pass's covered set for DAG simplification.** A
  command that's fully covered needn't have dependencies emitted for it.
  Today we DCE after DAG build; doing the DCE pass first (before DAG) would
  avoid building edges for doomed commands. Correctness is the same; it's
  a perf optimisation for later.

- **Scheduler warm-start.** Frame-to-frame command lists are very similar
  for a static UI. We could cache the emit order and invalidate only when
  the command count or any stateKey changes. Reconsider if DAG build ever
  shows up in profiles.

---

## Summary of file touch points

| File | Step | Change |
|------|------|--------|
| `jvk/core/DrawCmd.h` | 1 | Extend struct |
| `jvk/core/Scope.h` (new) | 1 | Declare `Scope` |
| `jvk/core/Renderer.h/.cpp` | 1, 2 | Own `scopes_`, `scheduler_`, call in `execute()` |
| `jvk/graphics/Graphics.h` | 1 | Push `Scope` on clip, populate new cmd fields |
| All `pushBlur*`, `pushFill*`, etc. call sites | 1 | Compute writesPx/readsPx/stateKey/isOpaque |
| `jvk/core/CommandScheduler.h/.cpp` (new) | 2, 3, 4, 5 | Class body, modes, DAG, cull, fuse |
| `source/components/ui/` | 2 | Debug mode-switch button |
| Each opted-in `*Pipeline.h` | 5 | `supportsFusion`, `maxBatchSize`, `executeBatch` |
| Each opted-in shader | 5 | Primitive-SSBO variant |

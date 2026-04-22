# GeometryPrimitive — Implementation Plan

A coordinated refactor of the pipelines + shaders so that every raster op in
`jvk` speaks one common **geometry format**, and every dispatch is shaped as a
**style-B instanced quad draw** (`vkCmdDraw(6, N, 0, 0)`) over a per-instance
SSBO. Two changes that only make sense together:

1. **Geometry abstraction.** One `GeometryPrimitive` struct describes a rect,
   rounded rect, ellipse, capsule/line, glyph, or a single closed sub-path.
   The fill, blur, and clip shaders each take the *same* primitive SSBO and
   select behaviour per-instance via `geometryTag` + per-pipeline payload.
2. **Style-B batched quads.** One GPU quad per primitive, carried in the SSBO
   (no chained-SDF union over a union-bbox). A batch of N compatible
   primitives becomes one `vkCmdDraw(6, N, …)`. Warp execution stays
   instance-uniform (every fragment in an instance runs the same tag branch),
   so the cost of mixing geometry types across a batch is zero.

The existing `CommandScheduler::Fuse` mode consumes this: it packs same-
`stateKey` independent primitives into one batched command whose `dataOffset`
points to a run of `GeometryPrimitive`s in the arena. Without this refactor,
`Fuse` has nothing to fuse *into*.

---

## Design invariants

These must hold across the whole refactor:

1. **Pixel equivalence.** Every phase must produce pixels byte-identical to
   the current implementation for representative frames. Differences are
   bugs, not improvements. Only the final perf measurement is allowed to
   change; visuals do not.

2. **Unified primitive format across pipelines.** If fill takes a
   `GeometryPrimitive`, blur and clip take the same `GeometryPrimitive`.
   Never diverge the struct per pipeline — payloads hang off it, but the
   geometry fields are shared. A caller who has already computed the tag,
   bbox, and inverse affine for one op can reuse them for another.

3. **One quad per closed sub-shape.** Arbitrary `juce::Path`s are
   decomposed into their closed sub-paths at record time, and *each* sub-
   path gets its own primitive + its own quad. Non-convex ORs of segments
   on the CPU are cheaper than fat-bbox SDF-walks on the GPU.

4. **Descriptor slot discipline.** Set indices are fixed across pipelines so
   State's descriptor-tracker doesn't have to care which geometry-abstracted
   pipeline is bound. Adding a new pipeline never renumbers an existing slot.

5. **Style-B dispatch is the only shape.** No raw `vkCmdDraw(3, 1)` full-
   screen triangles, no CPU-emitted 6-vertex quad vertex buffers. Vertex
   shader runs `gl_VertexIndex` against a tiny lookup for the 6 quad corners,
   `gl_InstanceIndex` reads the primitive out of the SSBO.

---

## Data model

### `GeometryPrimitive` — shared, 112 bytes

Defined once in `jvk/graphics/pipelines/Params.h`, mirrored in
`shaders/geometry.glsl`. `std430` layout on the GLSL side; the C++ struct is
POD with matching scalar/vec4 packing so `memcpy` into an arena span is
verbatim. Size is chosen so that a 1024-primitive batch SSBO is 112 KB —
fits comfortably in a `VkBuffer` frame slice, stays under the 128 KB
push-descriptor threshold on most drivers.

```cpp
// C++ side — aliased as std::array<float, 4> / std::array<uint32_t, 4> so
// there's no anonymous-struct padding surprise.
struct GeometryPrimitive {
    float    bbox[4];              // xyxy — physical-pixel AABB, incl. AA margin
    uint32_t flags[4];             // (geometryTag, shapeFlags, pathStart, pathCount)
    float    invXform01[4];        // m00, m10, m01, m11  (physical → local)
    float    invXform23_half[4];   // m02, m12, shapeHalf.x, shapeHalf.y
    float    extraA[4];            // geometry-type scratch (see below)
    float    extraB[4];            // geometry-type scratch
    float    payload[4];           // pipeline-specific payload (fill/blur/clip)
};
static_assert(sizeof(GeometryPrimitive) == 112, "std430 mirror");
```

`flags[0] = geometryTag` is the warp-uniform discriminator (all fragments of
one instance share it, so the branch never diverges within a warp):

| tag | geometry           | extraA                        | extraB                  |
|-----|--------------------|-------------------------------|-------------------------|
| 0   | Rect               | unused                        | unused                  |
| 1   | RoundedRect        | (cornerRadius, 0, 0, 0)       | unused                  |
| 2   | Ellipse            | unused (shapeHalf = radii)    | unused                  |
| 3   | Capsule / Line     | (lineB.x, lineB.y, thick, 0)  | unused                  |
| 4   | MSDF glyph         | (atlasU0, atlasV0, atlasU1, atlasV1) | (screenPxRange, weight, 0, 0) |
| 5   | Path sub-shape     | unused — (pathStart, pathCount) live in flags[2..3] | unused |
| 6   | Image              | unused — textureIndex in flags[1] (bindless) | unused |

`flags[1] = shapeFlags` is a small bitfield: `bit0 = inverted`, `bit1 =
stroke (thickness in extraA.z for non-capsule)`, `bit2..3 = fill rule`, `bit4
= opaque` (matches `DrawCommand::isOpaque` for style-A-free cull fast-paths).

`payload[4]` is *pipeline-specific* but lives in the shared struct so the
CPU can emit one `memcpy` instead of two. The fields below are overlaid on
`payload` per-pipeline:

```cpp
// Fill:  payload = { colorRGBA_packed, gradientT_x, gradientT_y, gradientRowNorm }
// Blur:  payload = { maxRadius, falloff, blurStep, edge | (mode<<8) }
// Clip:  payload = { 0, 0, 0, 0 }  // clip ops need no per-primitive payload
// Image: payload = { opacity, 0, 0, 0 }
```

When fill gradients and glyphs coexist in a batch, gradient coefficients
travel per-instance here — no per-vertex `gradientInfo` attribute survives.

### `GeometryQuadVert` — the vertex shader's lookup

The vertex shader is a single thin body shared across all pipelines (lives
in `shaders/geometry.vert`). Given `gl_InstanceIndex` + `gl_VertexIndex`, it
reads the primitive's `bbox` and expands to one of six corner positions:

```glsl
const vec2 CORNERS[6] = vec2[](
    vec2(0, 0), vec2(1, 0), vec2(1, 1),
    vec2(0, 0), vec2(1, 1), vec2(0, 1)
);
// corner = CORNERS[gl_VertexIndex]
// worldPos = mix(bbox.xy, bbox.zw, corner)
// uv       = corner  (0..1 across the primitive's quad)
```

No vertex buffer is bound — `vkCmdBindVertexBuffers` goes away for
geometry-abstracted pipelines. This kills one of the larger per-command
costs in the current recorder (`flushToGPU` of the UIVertex ring).

### `geometry.glsl` — shared fragment include

Every geometry-abstracted fragment shader `#include`s this. It exposes:

- `Primitive fetchPrimitive(uint instanceIdx)` — reads one primitive out of
  the SSBO at set=1/binding=0.
- `float shapeSDF(Primitive, vec2 localPos)` — dispatches on `geometryTag`:
  `sdRect` / `sdRoundedRect` / `sdEllipse` / `sdCapsule` / `sdPathWinding`
  (consumes the segment SSBO at set=2/binding=0).
- `vec4 shapeMask(Primitive, vec2 localPos, vec2 fragCoord)` — returns
  RGBA mask for MSDF (tag=4) and images (tag=6); identity `vec4(1)` for SDF
  shapes. Fill pipelines multiply against `sampleColor()`.
- `vec2 toLocal(Primitive, vec2 fragCoord)` — applies `invXform01/23_half`
  to get shape-local pixel position.

Path winding lives in this shared include, so `fill`, `blur`, and `clip`
all walk the same segment SSBO the same way. No more copy-pasted
`crossing()` + `sdSegment()` across three shaders.

---

## Descriptor slot conventions

Fixed globally across all geometry-abstracted pipelines. Each pipeline binds
only the sets it reads, but the *slot numbers* never shift:

| Set | Binding | Resource                          | Required by                      |
|-----|---------|-----------------------------------|----------------------------------|
| 0   | 0       | Scene source texture (sampler2D)  | Blur, PointwiseEffect            |
| 1   | 0       | `GeometryPrimitive[]` SSBO        | Fill, Blur, Clip, Effect         |
| 2   | 0       | Path segment SSBO (`vec4` list)   | Fill (tag=5), Blur (tag=5), Clip |
| 3   | 0       | MSDF glyph atlas (sampler2D)      | Fill (tag=4)                     |
| 3   | 1       | Gradient atlas (sampler2D)        | Fill (gradient mode on payload)  |
| 4   | 0       | Image texture array (bindless)    | Fill (tag=6)                     |

Set 0 is bound per-dispatch (scene source is ping-ponged); sets 1–4 are
bound once at pipeline-bind time and unchanged across the fuse run, which
is the entire point of batching.

The clip pipeline binds only {1, 2}. The blur pipeline binds {0, 1, 2}. The
fill pipeline binds {1, 2, 3, 4}. PointwiseEffect (HSV etc.) binds {0, 1}.

This replaces the current ad-hoc arrangement where `ColorPipeline` uses
set 0 for colorLUT + set 1 for shapeTex, `PathPipeline` uses set 0 for
colorLUT + set 1 for segments, `ShapeBlurPipeline` uses set 0 for the
source image, and so on.

---

## Sub-path decomposition

The `FillPath` / `BlurPath` / clip-path recorders currently upload one flat
list of segments for the whole path, one quad covering the whole path's
bbox. That's style-A. Under the new scheme:

1. Walk `juce::Path` in record order, splitting at every `closeSubPath()` /
   `startNewSubPath()`. Each run between those markers is one *sub-shape*.
2. For each sub-shape:
   - Flatten its beziers to line segments (same as today, via
     `PathFlatteningIterator`).
   - Compute its AABB in physical pixels (already cheap).
   - Upload its segment range to `PathPipeline::uploadSegments` and
     remember `(segmentStart, segmentCount)`.
   - Emit *one* `GeometryPrimitive` with `geometryTag = 5` and
     `flags[2..3] = (segmentStart, segmentCount)`.
3. The caller's single `fillPath` call becomes a `push(DrawOp::FillPath)`
   whose arena payload is a *span* of N primitives. The scheduler sees one
   command with `writesPx = union(primitive bboxes)`; at dispatch the
   pipeline draws `vkCmdDraw(6, N, 0, 0)`.

**Why sub-paths.** A letter 'O' is two sub-paths (outer + inner). Today both
live in one segment range; the whole union-bbox pays the per-fragment cost
of walking both. Under sub-paths, each fragment walks only the segments of
the quad it's in — the inner ring fragments don't pay the outer's cost and
vice versa. For a typical UI (the Fine Classics chain headers, volume arcs,
knob indicators), path flattening is already what it is; the win is that
*fragments* stop over-paying.

The even-odd vs non-zero fill-rule semantics are preserved per-primitive in
`shapeFlags`, so an even-odd path with holes still renders correctly —
`tag=5` fragments accumulate winding over their own sub-path only, and the
outer + inner primitives simply composite via alpha. (Verify: this is
exactly what the current whole-path even-odd gives you modulo a pixel-
precise ±ε on shared edges. If that ε shows up, the fix is to use signed
SDF + `min()` across siblings in a single primitive — punt unless seen.)

### Record-time helper

`Graphics::pushPathPrimitives(juce::Path&, const FillPayload&)` does the
sub-path walk + primitive emission. Lives in `Graphics.h`. Every caller
(`fillPath`, `strokePath`, `fillBlurredPath`, `drawBlurredPath`,
`clipToPath`) funnels through it — one place for the segmentation logic,
one place to change fill rules, one place to handle degenerate sub-paths.

---

## Shader organisation

```
shaders/
  geometry.glsl       shared fragment include — Primitive struct, SDFs,
                      toLocal/shapeMask/shapeSDF helpers
  geometry.vert       shared vertex shader — bbox → quad expansion
  geometry.vert.spv

  fill.frag           tag-dispatched fragment — samples color * shape
  fill.frag.spv

  blur.frag           tag-dispatched fragment — computes t from shapeSDF,
                      runs one of three kernels (Low/Medium/High), blurs
                      against the scene source
  blur.frag.spv

  clip.frag           tag-dispatched fragment — discards out-of-shape,
                      no color write (stencil INCR/DECR fixed-function)
  clip.frag.spv

  effect.frag         tag-dispatched fragment — pointwise ops (HSV, Blend).
                      Samples scene source and applies payload op. No SDF.
  effect.frag.spv

  copy.frag           (unchanged — passthrough copy)
```

Current shaders retired: `ui2d.vert`, `ui2d.frag`, `path_sdf.vert`,
`path_sdf.frag`, `shape_blur.frag`, `path_blur.frag`, `clip.vert`,
`clip.frag`, `hsv.frag`, `shader_region.vert`. Retained:
`blur.vert/.frag` remains only as the **plain 2-pass separable blur** for
`EffectBlend` / `EffectResolve` / any non-shape-aware full-screen blur
(name stays `blur.frag`, but the variable-radius / path-aware logic moves
into the new `blur.frag`). TODO: pick distinct filenames — leaning toward
renaming the old plain one to `kernel.frag` to free up `blur.frag` for the
geometry-abstracted one.

`compile.sh` adds the shared include to its dependency list (`geometry.glsl`
is `#include`d by several shaders; touch-recompile them all on change).

---

## Pipeline consolidation

Before the refactor:

| Current pipeline       | Ops handled                                  |
|------------------------|----------------------------------------------|
| `ColorPipeline`        | FillRect, FillRoundedRect, FillEllipse, Stroke*, DrawImage, DrawGlyphs, DrawLine |
| `PathPipeline`         | FillPath                                     |
| `ClipPipeline`         | PushClipPath, PopClipPath                    |
| `ShapeBlurPipeline`    | BlurShape                                    |
| `PathBlurPipeline`     | BlurPath                                     |
| `HSVPipeline`          | EffectHSV                                    |
| `EffectPipeline`       | EffectBlend, EffectResolve, EffectKernel     |
| `BlendPipeline`        | (low-level compositor, kept)                 |
| `ShaderPipeline`       | DrawShader (user custom — kept, out of scope)|

After:

| New pipeline             | Ops handled                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `FillPipeline`           | FillRect, FillRoundedRect, FillEllipse, Stroke*, DrawImage, DrawGlyphs, DrawLine, FillPath |
| `BlurPipeline`           | BlurShape, BlurPath                                                          |
| `ClipPipeline`           | PushClipPath, PopClipPath                                                    |
| `PointwiseEffectPipeline`| EffectBlend, EffectResolve, EffectKernel, EffectHSV                          |
| `BlendPipeline`          | unchanged                                                                    |
| `ShaderPipeline`         | unchanged (user custom)                                                      |

`FillPipeline` absorbs `ColorPipeline + PathPipeline`. `BlurPipeline`
absorbs `ShapeBlurPipeline + PathBlurPipeline`. `PointwiseEffectPipeline`
absorbs `HSVPipeline + EffectPipeline` — the former already share an
architecture (full-screen pass, sample-source-then-transform), they just
differ in the per-pixel math, which is a `geometryTag`-style dispatch.

---

## Step A — Land the scaffold (no behaviour change)

**Goal.** `GeometryPrimitive` + `geometry.glsl` + `geometry.vert` exist and
compile. Nothing uses them yet. Keeps the build green through the refactor.

**Acceptance.** Clean build. No pipeline switches. Visuals unchanged.

### Files changed / added

- `modules/jvk/include/jvk/graphics/pipelines/Params.h` — add
  `GeometryPrimitive` struct with the static_assert on size.
- `modules/jvk/shaders/geometry.glsl` (new) — the shared include. Contains
  Primitive struct, SDF helpers (rect/rrect/ellipse/capsule — lifted from
  `shape_blur.frag`), `fetchPrimitive`, `toLocal`, `shapeSDF`, `shapeMask`.
  Guarded by an `#ifdef SEGMENTS_SSBO_BOUND` for the path-winding branch so
  a pipeline that doesn't bind set 2 still compiles.
- `modules/jvk/shaders/geometry.vert` (new) — bbox-expanding vertex shader.
  Binds set 1 / binding 0 (the primitive SSBO) at vertex stage.
- `modules/jvk/shaders/compile.sh` — include the new files in the
  compile list.

### Testing

- Build succeeds. `.spv` files land.
- Hand-run a tiny offline test that memcpy's a known `GeometryPrimitive`
  into a buffer and reads it back through a compute shader — offsets match
  the `std430` expectation. (Skip if time-pressed; a visible mismatch in
  Step B will surface it.)

---

## Step B — BlurPipeline, single-primitive dispatch (style-B-of-1)

**Goal.** Replace `ShapeBlurPipeline` and `PathBlurPipeline` with one
`BlurPipeline` that dispatches `vkCmdDraw(6, 1, 0, 0)` against a 1-primitive
SSBO per command. `BlurShape` and `BlurPath` dispatch through the same
pipeline; the tag discriminates. No batching yet.

Chosen first because it's the biggest visible win and the two sources
(`ShapeBlurPipeline.h` + `shape_blur.frag`) already contain almost all the
SDF + kernel logic — porting is mostly a layout reshuffle.

**Acceptance.** Visuals identical for every screen with a blur effect
(backdrop blur, TopBar button blurs, any `fillBlurredPath` callsites). Perf
neutral; no per-frame dispatch count change yet.

### Files changed

- `modules/jvk/include/jvk/graphics/pipelines/effect/BlurPipeline.h` (new)
  — absorbs `ShapeBlurPipeline` and `PathBlurPipeline`. Descriptor layout =
  {set 0: scene source, set 1: primitive SSBO, set 2: segments SSBO}.
  Owns one `VkPipeline` per `BlurMode` (shader spec constants select the
  kernel branch); the primitive's `mode` field picks which pipeline to
  bind at dispatch.
- `modules/jvk/shaders/blur.frag` (new) — tag-dispatched. Covers the
  existing shape_blur body for tags 0..3 and the path_blur body (walk
  segments, unsigned-distance + winding) for tag 5. `effectiveRadius` is
  derived the same way: `shapeSDF → band → t → t * maxRadius * blurStep`.
- `modules/jvk/include/jvk/graphics/Graphics.h` — `pushBlurShape` /
  `pushBlurPath` emit a one-element primitive span to the arena and tag
  the command as `DrawOp::BlurShape` / `DrawOp::BlurPath`. The existing
  `BlurShapeParams` / `BlurPathParams` structs go away (the primitive *is*
  the params now); `Params.h` keeps `BlurParams` (non-shape-aware) for
  `Graphics::blur()`.
- `modules/jvk/include/jvk/core/Renderer.cpp` — the `EffectBlurShape` /
  `EffectBlurPath` dispatch handlers fold into one `EffectBlur` handler that
  reads `cmd.dataOffset → GeometryPrimitive[N]` (N=1 here), binds
  `BlurPipeline`, and dispatches.
- Delete `ShapeBlurPipeline.h`, `PathBlurPipeline.h`, `shape_blur.frag`,
  `path_blur.frag` once the new pipeline is proven.

### Retained behaviours

- **Kernel type selection** stays on the push constant, not in the
  primitive. A batch of N primitives shares `kernelType` — the scheduler
  already enforces this via `stateKey`.
- **Separable 2-pass for Low mode**: still works, the pipeline binds the
  same pipeline twice with `direction = (1,0)` then `(0,1)`, one-primitive
  SSBO reused across both passes. No per-pass CPU work beyond swapping the
  push constant, same as today.
- **Ping-pong scene buffer**: unchanged. Set 0 gets rebound each pass.
- **Stroke vs fill for paths**: `strokeHalfWidth > 0` moves into
  `extraB.x` on the primitive; `payload` still carries radius/falloff.

### Testing

- Visual diff: take a screenshot of the plugin's static UI with backdrop
  blur + TopBar button blurs visible. Compare against the same UI on
  `main`-or-pre-refactor branch. Byte-identical on the relevant regions
  (allow ±1 on the last-bit of alpha only if MSAA is involved; blur is
  FP32-shader so bit-identical is the right bar).
- Stress: `fillBlurredPath` over a complex path with multiple sub-paths.
  Confirm visually-identical output before the sub-path decomposition
  lands (i.e. Step B still treats a path as ONE primitive with one full
  segment range; Step D decomposes).
- Dump + inspect command stream: `BlurShape` / `BlurPath` commands carry a
  primitive-span `dataOffset` with count=1, `stateKey` unchanged.

---

## Step C — BlurPipeline, batched dispatch

**Goal.** The scheduler's `Fuse` mode packs runs of same-`stateKey`,
pairwise-independent blur commands into one fused `DrawOp::BlurShape` (or
`BlurPath`) whose arena payload is a `GeometryPrimitive[N]`.
`BlurPipeline` dispatches `vkCmdDraw(6, N, 0, 0)` once. The scene
render-pass amortisation already landed in the partial Step 5 keeps
paying — N blurs, 2 pass transitions for Low kernel, 1 for High.

**Acceptance.** For the TopBar-buttons blur cluster (~5 blurs today), the
dump shows one fused command with N=5 primitives where previously there
were 5 separate commands. Paint-thread + worker-thread time on that frame
drops measurably (target: 40–60% of the current draw-dispatch cost for that
cluster, the savings being per-dispatch setup + pipeline rebinds).

### Changes

- `CommandScheduler.cpp` — extend the list-scheduler emit step (already
  sketched in `CommandScheduler.md` Step 5) to take the full same-
  `stateKey` subset of the ready set, check pairwise independence
  (transitive-closure bitmap — ready-set sizes are small, O(R²) is fine),
  and emit one fused command with `dataOffset → Arena.pushSpan<…>`.
- `BlurPipeline::dispatch` — takes `(GeometryPrimitive*, uint32_t count)`,
  writes the count into the fused SSBO binding, issues
  `vkCmdDraw(6, count, 0, 0)`.
- `pipeline.supportsFusion() = true`, `pipeline.maxBatchSize() = 1024`
  (matches the 112 KB SSBO budget).

### Notes

- Both 2-pass separable (Low) and single-pass (High) work: the fused
  dispatch runs the 6×N instanced draw once per pass, reading the same
  N primitives both times.
- If the batch's primitives have wildly different `maxRadius`, the GPU
  still correctly per-fragment picks the primitive that contains it (via
  bbox pre-test + in-bbox SDF evaluation). Overlapping blurs with
  different radii: argmax-of-t, same rule we committed to earlier. If a
  user needs true summation they can author it explicitly by stacking
  non-overlapping regions (documented).

### Testing

- Visual diff vs Step B: identical pixels regardless of fuse on/off for
  the TopBar blurs. The scheduler mode button is the easy A/B.
- Perf HUD: dispatch-count drops.
- Artificial stress: synthesize 64 non-overlapping blurs in a test screen;
  confirm one fused dispatch, no visual surprises at the boundary between
  primitives' bboxes.

---

## Step D — FillPipeline, geometry-abstracted + batched

**Goal.** `ColorPipeline` and `PathPipeline` merge into `FillPipeline`.
Every fill op (rect, rrect, ellipse, stroke variants, glyphs, line, path,
image) becomes one or more `GeometryPrimitive`s routed through a single
instanced draw. Sub-path decomposition lands here.

Largest refactor; unblocks the biggest long-term perf win (batched fills
dominate commands per frame).

**Acceptance.** Visuals byte-identical across the full plugin UI in
Identity mode. In `Fuse` mode, runs of same-`stateKey` fills collapse —
e.g. the 10-letter label rendering path (10 DrawGlyphs today) becomes one
fused command, and a row of same-color meters becomes one fused FillRect
command.

### Files changed

- `modules/jvk/include/jvk/graphics/pipelines/fill/FillPipeline.h` (new)
  — absorbs `ColorPipeline` and `PathPipeline`. Descriptor layout =
  {set 1: primitive SSBO, set 2: segments SSBO, set 3: {0: glyph atlas,
  1: gradient atlas}, set 4: image array (optional/bindless)}.
- `modules/jvk/shaders/fill.frag` (new) — tag-dispatched. Bodies port
  wholesale from `ui2d.frag` and `path_sdf.frag`:
  - tag 0..3 → SDF rect / rrect / ellipse / capsule (from ui2d.frag,
    types 1, 2, 8, 9, 10).
  - tag 4 → MSDF glyph (ui2d.frag type 4).
  - tag 5 → path winding + segment distance (from path_sdf.frag).
  - tag 6 → image sample (ui2d.frag type 3).
- `modules/jvk/include/jvk/graphics/Graphics.h` — every `push*` site
  routes through `emitPrimitive(DrawOp, GeometryPrimitive&&, fillPayload)`
  plus the sub-path walk for `fillPath`/`strokePath`/`clipToPath`.
- `modules/jvk/include/jvk/core/Renderer.cpp` — fold the FillRect /
  FillRoundedRect / … / FillPath / DrawGlyphs / DrawImage handlers into
  one `FillDispatch` handler that reads a primitive span and issues one
  `vkCmdDraw(6, N, 0, 0)`. Stencil state per-op respects `cmd.stencilDepth`
  exactly as today.
- Retire `ColorPipeline.h`, `PathPipeline.h` (or keep `PathPipeline`'s
  segment-SSBO ownership under a new `PathSegments` utility — the
  CPU-side staging + per-slot SSBO logic stays, just decoupled from the
  draw dispatch).
- Retire `ColorDraw.h` once all `ColorDraw::vertexFor*` helpers are gone.
  `GradientCtx` moves into `FillPayload` construction in `Graphics.h`.
- Retire `UIVertex` attribute pipeline: with no vertex buffer bound, the
  per-vertex attributes go away. `UIVertex` can stay as a CPU-side
  intermediate if useful, but the GPU no longer sees it.
- Retire `ui2d.vert`, `ui2d.frag`, `path_sdf.vert`, `path_sdf.frag`.

### Sub-paths sanity

For text (the hot path), the batcher treats each DrawGlyphs as emitting N
glyph primitives (N = glyph count). Today it's already one DrawGlyphs
command per paragraph, with N vertices; the new layout flips it to N
instances. Same wall-clock GPU cost; on the scheduler side, two
neighbouring DrawGlyphs with the same font + fill (e.g., TopBar label +
TopBar subtitle) fuse into one `vkCmdDraw(6, N_total, 0, 0)`.

### Stencil + scissor considerations

`cmd.clipBounds` + `cmd.stencilDepth` continue to authoritatively drive
per-command scissor + stencil reference. In a fused batch, *every
primitive shares the same clipBounds and stencilDepth* — guaranteed by
`CommandScheduler::scheduleRegion` only fusing across `sameScopeLineage`
peers with matching `stateKey`. Ensure `stateKey` encodes stencilDepth
(via `blendKey`) so a fuse across a stencil boundary is impossible.

### Testing

- Visual diff across the full plugin UI, Identity vs Fuse.
- 1-primitive fused batch (degenerate case) equals a singleton dispatch
  pixel-for-pixel.
- Stress: presets menu open (dense text + backdrop + list rows); confirm
  the TopBar-button glyph fusion kicks in; confirm the list-row fill
  fusion kicks in.
- Pedal chain: confirm filmstrip `DrawImage` renders unchanged (images
  land on tag 6; until bindless textures land, per-image batching is
  limited to same-texture runs — that's OK).

### Known caveats

- Bindless images (tag 6) need VK_EXT_descriptor_indexing or equivalent
  to truly batch across distinct textures. For this step, keep per-image
  batching *within* the same image only. Cross-image fusion is a later
  phase — each image is its own `stateKey`, so the scheduler already
  correctly avoids mixing them.
- Gradient LUT atlas stays exactly as-is — one atlas, row-per-gradient.
  The gradient's row-norm travels on the primitive payload.

---

## Step E — ClipPipeline, geometry-abstracted

**Goal.** `ClipPipeline` becomes `GeometryPrimitive`-driven: push and pop
both emit one primitive (rrect = tag 1, path-subshape = tag 5). Same-
frame non-overlapping clips batch. Stencil push/pop pipelines still bind
INCR / DECR, but both consume the unified primitive SSBO.

Smallest of the three refactors; lands last because the other two already
proved the geometry-abstracted pattern and the clip work mostly lifts their
scaffolding.

**Acceptance.** Visuals identical. The 10-button TopBar clip runs that
currently serialize as 10 sequential push+pop pairs coalesce into 2
pipelines dispatches (10 pushes in one, 10 pops in one) when the
scheduler picks the same-scope non-overlapping siblings up into one batch.

### Changes

- `modules/jvk/include/jvk/graphics/pipelines/clip/ClipPipeline.h` —
  descriptor layout becomes {set 1: primitive SSBO, set 2: segments SSBO}.
  Drop the shape-type push constant; the primitive's `geometryTag`
  replaces it.
- `modules/jvk/shaders/clip.frag` (new) — tag-dispatched: tag 1 = rrect
  SDF, tag 5 = path winding (same `shapeSDF` helper from `geometry.glsl`).
  Discard outside; no color write.
- `modules/jvk/include/jvk/graphics/Graphics.h` — `pushClipPath` emits
  one primitive per sub-path; `popClipPath` mirrors that. The
  `ClipShapeParams` struct in `Params.h` retires (primitive replaces it).
- `CommandScheduler.cpp` — no change (clip commands are already part of
  the DAG via cover-rect writesPx; they just happen to now batch).

### Testing

- Every button with a rounded clip: visual diff, pixel-identical.
- Clip-to-path (preset menu, knob tracks if any): pixel-identical.
- Dump: confirm a TopBar-wide run shows N rrect primitives in one fused
  push and N in one fused pop.

---

## Step F (optional) — fold pointwise effects into the family

**Goal.** `HSVPipeline` + `EffectPipeline` become `PointwiseEffectPipeline`
with a tag-dispatched fragment. Minor cleanup; the dispatch count isn't
currently a problem for these.

**Acceptance.** Visual parity. No explicit perf target — this is a
consistency win, not a speed win.

### Why it's Step F

The argument for folding is "geometry abstraction is consistent across all
ops". The argument against is that pointwise effects don't actually have
geometry — they're full-screen passes keyed off a region rect. The
compromise: they *do* emit a `GeometryPrimitive` (tag 0, bbox = region),
`payload` carries the op-specific params, and the fragment shader
`#include`s `geometry.glsl` only for `toLocal` + bbox filtering.

If we defer, they stay as-is and the family is 3-pipeline + 2-legacy. If
we land it, the family is 4 pipelines, all speaking `GeometryPrimitive`.

Defer unless cleanup time is cheap.

---

## Rollout summary

| Phase | Code hot | Risk | Visible win |
|-------|---------|------|-------------|
| A: Scaffold | Low | Low | None — prep |
| B: BlurPipeline unified | Medium | Medium (pixel-diff risk) | None — prep for C |
| C: BlurPipeline batched | Medium | Low (B proved the shader) | TopBar blur cluster perf |
| D: FillPipeline | High | High — touches everything | Text + fill batching, most commands per frame |
| E: ClipPipeline | Low | Low | Clip-push/pop batching |
| F: Pointwise effects | Low | Low | Consistency |

Each phase lands as its own branch → merge → measure → land-next. Do
**not** squash phases together even if small; the A/B comparison per
phase is how we prove we haven't broken pixel parity.

---

## Testing harness (applies to every phase)

1. **Golden-image diff.** Capture a reference frame of each screen
   (idle, preset menu open, presets switching, pedal open, mod panel
   visible, full-chain with blur) before starting Phase B. After each
   phase: re-capture, byte-diff. Any non-zero diff in a non-shader-touched
   region is a regression.
2. **Scheduler dump.** Each phase's first run: dump pre-sort and
   post-sort command streams in Identity, SortOnly, and Fuse. Inspect
   `stateKey` distribution, primitive counts, `dataOffset` ranges. The
   dump should read sanely to a human — if it doesn't, the scheduler's
   confused about compatibility.
3. **Render doc / Xcode GPU capture.** One capture per phase: confirm the
   pipeline-bind count drops, the `vkCmdDraw` count drops, and the
   descriptor-set-bind count drops. These three are the whole point.

---

## Open questions to revisit

- **Vertex shader cost.** `gl_InstanceIndex`-driven lookup is cheap on
  desktop but not free on mobile / integrated. If we ever port to
  Android, measure — may want to keep a tiny 6-vert VBO as a fallback.
- **112-byte primitive size.** Could shrink to 96 if we drop `extraB` and
  pack glyph atlas UV into 16-bit normalized. Defer unless SSBO bandwidth
  shows up in profiles.
- **Bindless for images (tag 6).** Current plan keeps per-image fusion
  only. If fully-bindless images land, the image-heavy UIs (plugin window
  backgrounds) collapse one more dispatch class. Track as a separate
  phase, dependent on `VK_EXT_descriptor_indexing` adoption.
- **`geometryTag` dispatch cost.** Ten-way switch per fragment is
  warp-uniform within an instance, but it *does* lengthen the compiled
  shader and can hurt register pressure. If `fill.frag` ends up
  register-bound (live occupancy drops), split into two variants:
  `fill_sdf.frag` for tags 0..3 + 5 and `fill_sampled.frag` for tags 4 +
  6. Same primitive input, different pipeline. Defer until measured.
- **Path decomposition edge cases.** Self-intersecting sub-paths across
  decomposition boundaries — fill rule semantics can differ from the
  union case. Verify during Phase D with a targeted test path.

---

## File touch points

| File | Phase | Change |
|------|-------|--------|
| `pipelines/Params.h` | A | `GeometryPrimitive` struct |
| `shaders/geometry.glsl` (new) | A | Shared SDFs + helpers |
| `shaders/geometry.vert` (new) | A | Bbox → quad, bindless vertex |
| `pipelines/effect/BlurPipeline.h` (new) | B | Absorbs ShapeBlur + PathBlur |
| `shaders/blur.frag` (new) | B | Tag-dispatched blur |
| `Graphics.h` (pushBlur*) | B | Emit one primitive per blur |
| `Renderer.cpp` (Blur handlers) | B | Fold into one |
| `CommandScheduler.cpp` | C | Extend emit → batched fuse |
| `BlurPipeline::dispatch` | C | Instanced draw over SSBO |
| `pipelines/fill/FillPipeline.h` (new) | D | Absorbs Color + Path |
| `shaders/fill.frag` (new) | D | Tag-dispatched fill |
| `Graphics.h` (all push*) | D | Route through `emitPrimitive` |
| `Renderer.cpp` (fill handlers) | D | Fold into one |
| `ClipPipeline.h` | E | Tag-dispatched clip |
| `shaders/clip.frag` (new) | E | Tag-dispatched clip |
| `Graphics.h` (pushClip*) | E | Emit per-subshape primitive |
| `PointwiseEffectPipeline.h` (optional) | F | Absorbs HSV + Effect |
| `shaders/effect.frag` (optional) | F | Tag-dispatched pointwise |

Retired at end of refactor: `ColorPipeline.h`, `PathPipeline.h`
(segment-SSBO half moves to a utility), `ShapeBlurPipeline.h`,
`PathBlurPipeline.h`, `ColorDraw.h`, `ui2d.vert`, `ui2d.frag`,
`path_sdf.vert`, `path_sdf.frag`, `shape_blur.frag`, `path_blur.frag`,
old `clip.vert`, old `clip.frag`, `hsv.frag`, `shader_region.vert`, and
`HSVPipeline.h` + `EffectPipeline.h` (if Phase F lands).

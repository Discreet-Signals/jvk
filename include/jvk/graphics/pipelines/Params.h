#pragma once

namespace jvk {

// Placement of the shape-blur falloff band relative to the shape edge (SDF=0).
//   Centered → falloff straddles the edge, half inside + half outside
//   Inside   → falloff sits entirely inside the shape, ending at the edge
//   Outside  → falloff sits entirely outside the shape, starting at the edge
enum class BlurEdge : uint32_t {
    Centered = 0,
    Inside   = 1,
    Outside  = 2,
};

// Blur kernel strategy. Picks the trade-off between speed and directional
// artefact tolerance:
//
//   Low    — 2 separable passes at 0° and 90°. Cheapest. Visible streaks
//            at `t` boundaries because each pass reads the previous pass's
//            variable-radius output; the axis-aligned streak direction
//            usually aligns with rectangular shape edges so it tucks into
//            the geometry, but for arbitrary shapes a cross artefact is
//            visible. Fine for most UI overlays at small radius.
//   Medium — 3 separable passes at 0°, 60°, 120°. Redistributes the
//            streak across 3 angles (6-fold). Trades the axis-aligned
//            cross for a faint hex; better on organic shapes but worse
//            on rectangles (streaks no longer align with edges).
//   High   — 1 pass, **true 2D** Gaussian — samples every physical pixel
//            in the kernel neighbourhood directly from the scene source,
//            no chained passes. Zero separable-blur artefacts. Cost is
//            O(N²) per fragment vs O(N) per pass, so reserve for hero
//            elements where Low / Medium streak visibly and the blur
//            radius is modest.
//
// TODO: this enum will grow into a real "blur type" selector. Candidates:
//   - Streaky / Anisotropic — deliberately non-isotropic directional
//     blur (Gaussian with a separable-axis imbalance) for frosted / hair-
//     line / motion-glass looks. The "tighter falloff at higher quality"
//     bug we just fixed produced a textured glass effect that's worth
//     revisiting as a proper mode.
//   - Box                    — cheap uniform-radius box filter.
//   - Kawase / DualKawase    — rotated 4-tap patterns; opt-in when we
//                              have a batched command plan that makes
//                              the pyramid setup worthwhile.
//   - BokehHex / BokehDisc   — bokeh-aesthetic non-separable kernels
//                              for DOF-style background content.
// When adding those, consider renaming the enum values to namespace-
// prefixed (GaussianLow, GaussianHigh, ..., Streaky, Box, etc.) so the
// type (Gaussian vs Kawase vs Box) is explicit at call sites.
enum class BlurMode : uint32_t {
    Low    = 0,
    Medium = 1,
    High   = 2,
};

// =============================================================================
// Parameter structs — POD only, packed into Arena via memcpy.
// Non-POD types (Font, FillType) stored in Renderer side vectors by index.
// =============================================================================

struct DrawShaderParams {
    void*                  shader; // Shader*
    juce::Rectangle<float> region;
    float                  scale;
};

struct PushClipRectParams {
    juce::Rectangle<int>   rect;
};

// Clip dispatch reference for the stencil push / pop pipelines. One ref
// shared by both DrawOp::PushClipPath and DrawOp::PopClipPath — the SAME
// GeometryPrimitive is used for both dispatches so the INCR on push and
// DECR on pop cancel exactly (same fragments touched, same op symmetric).
// Graphics::clipToPath remembers the ref on its pathClipStack; the matching
// restoreState → recordPopClip dispatches with the same firstInstance +
// primCount.
//
// Axis-aligned rectangle clips NEVER appear here — they fast-path through
// Graphics::clipToRectangle to a plain scissor update.
//
// `coverRect` is the shape's physical-pixel bounding box (+ 1px AA margin).
// Used only by Graphics to compute the pop command's writesPx/readsPx
// scheduling metadata — the dispatch itself drives scissor from
// DrawCommand::clipBounds and quad cover from the primitive's bbox.
struct ClipDispatchRef {
    uint32_t               firstInstance;   // [prefix]
    uint32_t               primCount;       // [prefix]
    uint32_t               arenaOffset;     // [prefix]
    uint32_t               _pad;
    juce::Rectangle<float> coverRect;
};

struct EffectBlendParams {
    float r, g, b;
    juce::Rectangle<float> region;
    float                  scale;
};

struct BlurParams {
    float                  radius;
    juce::Rectangle<float> region;
    float                  scale;
};

// Full-screen HSV transform. `scale*` defaults to 1.0, `delta*` to 0.0 —
// identity. Applied as `hsv *= (scale...); hsv += (delta...);` in hsv.frag.
// Specialised callers like Graphics::saturate / shiftHue / brightness fill
// in only the relevant fields; everything else stays at defaults.
struct HSVParams {
    float scaleH = 1.0f, scaleS = 1.0f, scaleV = 1.0f;
    float deltaH = 0.0f, deltaS = 0.0f, deltaV = 0.0f;
    juce::Rectangle<float> region;
    float scale;
};

struct EffectResolveParams {
    float                  param;
    uint32_t               type;
    juce::Rectangle<float> region;
    float                  scale;
};

// Common prefix for every dispatch ref emitted into the arena by a
// geometry-abstracted push site. Enables generic ref rewrite from the
// CommandScheduler's Fuse emit path (which doesn't know about op-specific
// tail fields) and from the Renderer's post-scheduler upload pass (which
// fills in `firstInstance` once the primitives are uploaded to the
// pipeline SSBO in scheduled order). Every DispatchRef below starts with
// these three u32s IN THIS ORDER; op-specific fields follow.
//
//   firstInstance  — SSBO index of the first primitive. Written post-
//                    scheduler by the upload pass. Undefined at record.
//   primCount      — number of consecutive primitives (for a single-op
//                    dispatch this is 1; Fuse mode may bump it after
//                    concatenating a run of compatible primitives).
//   arenaOffset    — byte offset in Arena where primCount consecutive
//                    GeometryPrimitive records live. Set at record time;
//                    Fuse may rewrite it to point at a newly-allocated
//                    concatenated span.
struct DispatchRefPrefix {
    uint32_t firstInstance;
    uint32_t primCount;
    uint32_t arenaOffset;
};

// Arena payload for DrawOp::BlurShape / DrawOp::BlurPath. `mode` (kernel
// selector) lives on the ref because the scheduler gates fusion by stateKey
// which already encodes mode — so every primitive in a fused dispatch
// agrees on kernelType / passCount.
struct BlurDispatchRef {
    uint32_t firstInstance;   // [prefix] SSBO index of the first primitive
    uint32_t primCount;       // [prefix] number of consecutive primitives
    uint32_t arenaOffset;     // [prefix] first primitive's offset in arena
    uint32_t mode;            // BlurMode as uint32 (0=Low / 1=Medium / 2=High)
};

// Arena payload for most FillPipeline dispatches (SDF shapes, strokes,
// lines, path). `fillIndex` resolves the colour-LUT descriptor at dispatch
// time (gradient atlas row for gradient fills, 1×1 default for solid).
struct FillDispatchRef {
    uint32_t firstInstance;   // [prefix]
    uint32_t primCount;       // [prefix]
    uint32_t arenaOffset;     // [prefix]
    uint32_t fillIndex;       // Renderer::getFill() slot — colour LUT source
};

// DrawImage dispatch — image texture descriptor is looked up via imageHash
// in ResourceCaches's texture cache at dispatch time. imageHash is u64, so
// it's placed after a pad word to keep 8-byte alignment.
struct FillImageDispatchRef {
    uint32_t firstInstance;   // [prefix]
    uint32_t primCount;       // [prefix]
    uint32_t arenaOffset;     // [prefix]
    uint32_t _pad;
    uint64_t imageHash;
};

// DrawGlyphs dispatch. Today each glyph is one primitive; the atlas page
// can vary per glyph, so the dispatcher iterates `primCount` primitives in
// the uploaded batch and rebinds the shape sampler when the page changes.
// The arena layout is:
//
//     FillGlyphsDispatchRef  (this struct)
//     uint32_t[primCount]    — atlas page indices (one per glyph primitive)
//
// Scheduler fusion can merge runs of same-page glyphs into one contiguous
// dispatch without extra per-command state — the page-indices array
// concatenates naturally.
struct FillGlyphsDispatchRef {
    uint32_t firstInstance;   // [prefix]
    uint32_t primCount;       // [prefix]  glyph count
    uint32_t arenaOffset;     // [prefix]
    uint32_t fillIndex;       // Renderer::getFill() slot
};

// =============================================================================
// GeometryPrimitive — unified per-instance record consumed by every geometry-
// abstracted pipeline (Blur today, Fill/Clip in later phases). One primitive =
// one GPU quad; a batch of N primitives becomes one vkCmdDraw(6, N, 0, first).
//
// Layout is std430-mirrored exactly in shaders/geometry.glsl (same field
// order, same vec4/uvec4 alignment); memcpy from the C++ side into the
// pipeline's per-frame primitive SSBO is verbatim.
//
// The core geometry fields are SHARED across pipelines — they describe WHAT
// shape to draw, not WHAT to do with it. Per-pipeline op-specific data lives
// in extraB + payload (overlaid, documented per pipeline).
//
//   bbox            Physical-pixel AABB (post-transform + AA margin). The
//                   vertex shader expands this directly into a 6-vertex quad;
//                   no forward transform runs in the vertex stage. CPU
//                   precomputes bbox as transform(shapeLocal).expanded(aa).
//
//   flags           x = geometryTag (see table below)
//                   y = shapeFlags  (bit0 inverted, bits1-2 edge, bits3-4
//                                    fillRule, bit5 stroke-vs-fill)
//                   z = pathStart   (tag=5 only — index into segment SSBO)
//                   w = pathCount   (tag=5 only — segment count)
//
//   invXform01/23_half
//                   Inverse affine mapping FRAG.xy (physical pixels) back
//                   into shape-local (logical) coords. Fragment shader uses
//                   it to evaluate SDFs in the shape's native space, so the
//                   caller's juce::AffineTransform never has to be baked
//                   into per-vertex data on the CPU. Layout:
//                     invXform01[4]      = m00, m10, m01, m11
//                     invXform23_half[4] = m02, m12, shapeHalf.x, shapeHalf.y
//                   shapeHalf is co-packed to save a vec4 — it's in logical
//                   units (pre-transform), consumed by the SDF call in tags
//                   0..3. For tag 5 (path) shapeHalf is unused.
//
// Geometry tags (flags.x):
//   0 — Rect                                  (SDF, shapeHalf)
//   1 — Rounded rect                          (SDF, shapeHalf + cornerRadius)
//   2 — Ellipse                               (SDF, shapeHalf = radii)
//   3 — Capsule / line                        (SDF, lineB + lineThickness)
//   4 — MSDF glyph                            (atlas UV in extraB, gradient in extraA if used)
//   5 — Path sub-shape                        (segments via flags.zw)
//   6 — Image                                 (sampler + atlas UV in extraB)
//
// shapeFlags (flags.y) bits:
//   bit 0     inverted           (flip shape-mask alpha, blur-only)
//   bits 1-2  edgePlacement      (blur falloff band placement)
//   bits 3-4  fillRule           (path: 0=non-zero, 1=even-odd)
//   bit 5     stroke             (stroke the shape's SDF instead of fill)
//   bits 6-7  colorSource        (0=solid, 1=linear-gradient, 2=radial-gradient)
//
// Per-pipeline payload conventions (see shaders/geometry.glsl for the
// matching shader side):
//
//   Fill pipeline (SDF shapes, tag 0..3 and tag 5 path):
//     extraA  = (cornerRadius, strokeWidth, lineB.x, lineB.y)
//     extraB  = (colorSource != 0) gradient coeffs: (origin.x, origin.y,
//               dir.x_or_invRadius, dir.y_or_zero)
//     payload = (colorSource != 0) gradient coeffs: (invLen2_or_zero,
//               rowNorm, 0, 0)
//     color   = solid RGBA, or (1,1,1,opacity) when a gradient / sampler
//               supplies RGB
//
//   Fill pipeline (tag 4 MSDF glyph):
//     extraA  = (colorSource != 0) gradient coeffs as above
//     extraB  = atlas UV rect (u0, v0, u1, v1)
//     payload = (invLen2_or_zero, rowNorm, screenPxRange, 0)
//     color   = RGBA (solid) or (1,1,1,opacity) for gradient
//
//   Fill pipeline (tag 6 Image):
//     extraA  = unused (may hold gradient coeffs for tinted images — not used today)
//     extraB  = atlas UV rect (u0, v0, u1, v1)
//     payload = unused
//     color   = (1,1,1,opacity) — texture supplies RGB, alpha scaled by .a
//
//   Blur pipeline (tags 0..3 + 5):
//     extraA  = (cornerRadius, lineThickness, lineB.x, lineB.y)
//     extraB  = (maxRadius,    falloff,       blurStep, strokeHalfWidth)
//     payload = unused
//     color   = unused
//
//   Clip pipeline (tags 1 rrect / 5 path):
//     everything but bbox / flags / invXform* / extraA.x (cornerRadius) / flags.zw
//     is unused. The fragment shader discards outside the SDF and never writes color.
// =============================================================================

struct GeometryPrimitive {
    float    bbox[4];              // 16  physical-pixel AABB (post-transform + AA)
    uint32_t flags[4];             // 16  (tag, shapeFlags, pathStart, pathCount)
    float    invXform01[4];        // 16  m00, m10, m01, m11   (physical → local)
    float    invXform23_half[4];   // 16  m02, m12, shapeHalf.x, shapeHalf.y
    float    extraA[4];            // 16  geometry-type / colorSource extras
    float    extraB[4];            // 16  op-specific payload A
    float    payload[4];           // 16  op-specific payload B
    float    color[4];             // 16  RGBA — solid or (1,1,1,opacity) mask
};
static_assert(sizeof(GeometryPrimitive) == 128,
              "GeometryPrimitive must be 128 B to match shaders/geometry.glsl std430 layout.");

// Tag values. Mirrored in shaders/geometry.glsl.
enum class GeometryTag : uint32_t {
    Rect     = 0,
    RoundRect = 1,
    Ellipse  = 2,
    Capsule  = 3,
    MSDFGlyph = 4,
    Path     = 5,
    Image    = 6,
};

enum class ColorSource : uint32_t {
    Solid    = 0,
    Linear   = 1,
    Radial   = 2,
};

// Packed shapeFlags encoder — use at record time to build flags.y.
// Unpacking convention is documented in shaders/geometry.glsl.
inline uint32_t packShapeFlags(bool        inverted,
                               BlurEdge    edge,
                               uint32_t    fillRule,     // 0 = non-zero, 1 = even-odd
                               bool        stroke,
                               ColorSource cs = ColorSource::Solid) noexcept
{
    return (inverted ? 1u : 0u)
         | ((static_cast<uint32_t>(edge) & 0x3u)  << 1)
         | ((fillRule & 0x3u)                     << 3)
         | ((stroke ? 1u : 0u)                    << 5)
         | ((static_cast<uint32_t>(cs) & 0x3u)    << 6);
}

} // namespace jvk

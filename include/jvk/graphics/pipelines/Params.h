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

struct FillRectParams {
    juce::Rectangle<float> rect;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct FillRoundedRectParams {
    juce::Rectangle<float> rect;
    float                  cornerSize;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct FillEllipseParams {
    juce::Rectangle<float> area;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct StrokeRoundedRectParams {
    juce::Rectangle<float> rect;
    float                  cornerSize;
    float                  lineWidth;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct StrokeEllipseParams {
    juce::Rectangle<float> area;
    float                  lineWidth;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct DrawImageParams {
    // Pre-resolved descriptor set for the cached texture. Captured at record
    // time (inside Graphics::drawImage via renderer_.caches().getTexture)
    // so the worker thread doesn't re-look-up in the shared texture map —
    // avoids a data race against concurrent inserts / evictions on that
    // std::unordered_map from sibling editors' message-thread activity.
    // The same getTexture call also pins the CachedImage via FrameRetained
    // so this VkDescriptorSet (and its backing VkImageView) stays valid for
    // the life of the draw.
    VkDescriptorSet        textureDesc;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
    int                    imageWidth;
    int                    imageHeight;
};

struct DrawGlyphsParams {
    uint32_t               glyphCount;
    juce::AffineTransform  transform;
    uint32_t               fontIndex;
    uint32_t               fillIndex;
    float                  opacity;
    float                  scale;
    // followed in arena by: uint16_t[glyphCount] + Point<float>[glyphCount]
};

struct DrawLineParams {
    juce::Line<float>      line;
    float                  lineWidth;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct DrawShaderParams {
    void*                  shader; // Shader*
    juce::Rectangle<float> region;
    float                  scale;
};

// Analytical SDF path fill. Graphics::fillPath flattens the path to line
// segments in physical-pixel space, uploads them to PathPipeline's storage
// buffer ring (via uploadSegments()), and packs the cover-quad vertices +
// segment range here for later replay. The fragment shader then iterates
// `segmentCount` segments starting at `segmentStart` in the SSBO.
//
// `quadVerts[6]` is a triangle-list quad in physical pixels covering the
// path bounds + 1 AA pixel. The quad carries the fill colour in vertex
// attrs. `fillRule` matches juce::Path::isUsingNonZeroWinding() — 0 for
// non-zero, 1 for even-odd.
struct FillPathParams {
    UIVertex quadVerts[6];  // color + gradientInfo pre-baked per corner
    uint32_t segmentStart;
    uint32_t segmentCount;
    uint32_t fillRule;      // 0 = non-zero winding, 1 = even-odd
    uint32_t fillIndex;     // renderer.getFill() slot → chooses colorLUT descriptor
};

struct PushClipRectParams {
    juce::Rectangle<int>   rect;
};

// Clip shape for the stencil push/pop pipelines. One payload shared by both
// DrawOp::PushClipPath and DrawOp::PopClip — the shapes must match exactly
// so the INCR on push and DECR on pop cancel (same fragments touched, same
// stencil op applied symmetrically).
//
// Axis-aligned rectangle clips NEVER appear here — they fast-path through
// Graphics::clipToRectangle to a plain scissor rect. This payload covers
// the two stencil-backed cases:
//
//   shapeType == 1  Rounded rectangle
//     centerX/Y + halfW/H + cornerRadius encode the shape in physical
//     pixels; the clip fragment shader evaluates roundedRectSDF and
//     discards outside pixels before the stencil op.
//
//   shapeType == 2  Arbitrary path
//     segmentStart/segmentCount index into the shared per-frame segment
//     SSBO (owned by PathPipeline). fillRule selects non-zero (0) or
//     even-odd (1) winding; the clip frag shader walks the segments and
//     computes the winding number per fragment.
//
// coverRect is the bounding box + 1px AA margin in physical pixels —
// emitted as a 6-vertex quad so the clip fragment shader runs for every
// pixel the shape could occupy.
struct ClipShapeParams {
    uint32_t               shapeType;       // 1 = rrect, 2 = path
    float                  centerX, centerY; // rrect only (physical px)
    float                  halfW, halfH;     // rrect only (physical px)
    float                  cornerRadius;     // rrect only (physical px)
    uint32_t               segmentStart;     // path only
    uint32_t               segmentCount;     // path only
    uint32_t               fillRule;         // path only
    juce::Rectangle<float> coverRect;        // quad to cover in physical px
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

// Shape-aware variable-radius blur. Pre-packed in the layout the fragment
// shader's push-constant block expects, minus viewport/direction which the
// pipeline fills in per-pass. Distances are in USER-LOGICAL pixels
// (pre-transform); the shader converts kernel sample offsets to physical
// texels via `blurStep`.
struct BlurShapeParams {
    float    invXform[6];          // inverse affine: frag (physical) → shape-local (logical)
    float    shapeHalf[2];         // rect/rrect/ellipse halfsize (logical)
    float    lineB[2];             // line: endpoint B in shape-local (A is origin)
    float    maxRadius;            // user-logical pixels
    float    falloff;              // user-logical pixels
    float    blurStep;             // physical texels per user-logical pixel
                                   // (= transformScale * displayScale)
    float    cornerRadius;         // rrect only
    float    lineThickness;        // line only
    uint32_t shapeType;            // 0=rect, 1=rrect, 2=ellipse, 3=line
    uint32_t edgePlacement;        // BlurEdge as uint32
    uint32_t inverted;             // 0 or 1
    uint32_t mode;                 // BlurMode as uint32 (pass count / type lives here)
};

struct EffectResolveParams {
    float                  param;
    uint32_t               type;
    juce::Rectangle<float> region;
    float                  scale;
};

// Analytical path SDF blur. Graphics::{draw,fill}BlurredPath flattens the
// path to line segments in PHYSICAL pixels (same as FillPath) and uploads
// them via PathPipeline::uploadSegments. All distances below are also in
// PHYSICAL pixels — the C++ caller pre-multiplies by transformScale ×
// displayScale so the shader runs in one coord space (step = 1 texel).
struct BlurPathParams {
    uint32_t segmentStart;
    uint32_t segmentCount;
    uint32_t fillRule;        // 0 = non-zero, 1 = even-odd
    float    maxRadius;       // physical px
    float    falloff;         // physical px
    float    strokeHalfWidth; // physical px — 0 for fill, >0 for stroke
    uint32_t edgePlacement;   // BlurEdge as uint32
    uint32_t inverted;        // 0 or 1
    uint32_t mode;            // BlurMode as uint32 (pass count / type lives here)
};

} // namespace jvk

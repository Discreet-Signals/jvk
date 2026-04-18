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
    uint64_t               imageHash;
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
// pipeline fills in per-pass. Distances are in LOGICAL pixels; the shader
// multiplies by `displayScale` when converting to physical sample offsets.
struct BlurShapeParams {
    float    invXform[6];          // inverse affine: frag (physical) → shape-local (logical)
    float    shapeHalf[2];         // rect/rrect/ellipse halfsize (logical)
    float    lineB[2];             // line: endpoint B in shape-local (A is origin)
    float    maxRadius;            // logical pixels
    float    falloff;              // logical pixels
    float    displayScale;
    float    cornerRadius;         // rrect only
    float    lineThickness;        // line only
    uint32_t shapeType;            // 0=rect, 1=rrect, 2=ellipse, 3=line
    uint32_t edgePlacement;        // BlurEdge as uint32
    uint32_t inverted;             // 0 or 1
};

struct EffectResolveParams {
    float                  param;
    uint32_t               type;
    juce::Rectangle<float> region;
    float                  scale;
};

} // namespace jvk

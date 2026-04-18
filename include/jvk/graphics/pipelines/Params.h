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

struct PushClipRectParams {
    juce::Rectangle<int>   rect;
};

// Forward decl for the cached-mesh fast path.
struct CachedPathMesh;

struct PushClipPathParams {
    uint32_t               vertexCount;
    uint32_t               vertexArenaOffset = 0; // where fan verts live in arena
                                                  // (ignored when cachedMesh != nullptr)
    const CachedPathMesh*  cachedMesh = nullptr;  // non-null → bind cached VkBuffer
    float                  transform[6];          // affine: m00 m10 m01 m11 m02 m12
    juce::Rectangle<int>   pathBounds;
};

struct PopClipParams {
    uint32_t               vertexCount       = 0; // 0 for rect clips
    uint32_t               vertexArenaOffset = 0; // same fan verts as paired PushClipPath
    const CachedPathMesh*  cachedMesh        = nullptr;
    float                  transform[6];          // same transform as paired push
    juce::Rectangle<int>   fanBounds;
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

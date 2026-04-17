#pragma once

namespace jvk {

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

struct PushClipPathParams {
    uint32_t               vertexCount;
    juce::Rectangle<int>   pathBounds;
    // followed in arena by: UIVertex[vertexCount] (triangulated path)
};

struct PopClipParams {
    uint32_t             vertexCount = 0;  // 0 for rect clips
    juce::Rectangle<int> fanBounds;
    // followed in arena by: UIVertex[vertexCount] (reversed fan for stencil decrement)
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

struct EffectResolveParams {
    float                  param;
    uint32_t               type;
    juce::Rectangle<float> region;
    float                  scale;
};

} // namespace jvk

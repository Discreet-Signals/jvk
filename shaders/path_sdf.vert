#version 450

// Analytical path-SDF vertex shader.
// Passes the quad vertex position through to the fragment shader in
// physical-pixel coordinates. The fragment shader then iterates the
// path's line segments (uploaded as a storage buffer) and evaluates
// min-distance + winding-number per pixel — no precomputed MSDF atlas.

layout(push_constant) uniform PC {
    vec2  viewportSize;  // physical pixel dimensions of the scene target
    uint  segmentStart;  // index of first segment in the segments SSBO
    uint  segmentCount;  // number of segments for this draw
    uint  fillRule;      // 0 = non-zero winding, 1 = even-odd
    float _pad;
} pc;

// UIVertex layout, shared with the rest of jvk's 2D pipelines.
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inShapeInfo;
layout(location = 4) in vec4 inGradientInfo;

layout(location = 0) out vec2 fragPos;       // physical pixel position
layout(location = 1) out vec4 fragColor;
layout(location = 2) out vec4 fragGradientInfo;

void main() {
    fragPos = inPosition;
    fragColor = inColor;
    fragGradientInfo = inGradientInfo;
    // Pixel → Vulkan clip space (Y-down), matching the rest of jvk.
    vec2 ndc = (inPosition / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

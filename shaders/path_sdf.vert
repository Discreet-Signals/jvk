#version 450

// Analytical path-SDF vertex shader — TILE mode.
//
// Graphics::fillPath decomposes each path into 16px tiles on the CPU
// (piet-gpu style): empty tiles are never emitted, interior tiles carry a
// constant winding ("backdrop") and no segments, and edge tiles carry a
// small local segment list. Per-tile data rides the UIVertex attributes:
//   inUV           = tile top-left corner in physical pixels (flat)
//   inShapeInfo.x  = local segment start (relative to pc.segmentStart)
//   inShapeInfo.y  = local segment count (0 = interior/exterior tile)
//   inShapeInfo.z  = backdrop winding at (tileRight, tileTop)
// All 6 vertices of a tile quad carry identical uv/shapeInfo, so `flat`
// interpolation is exact.

layout(push_constant) uniform PC {
    vec2  viewportSize;  // physical pixel dimensions of the scene target
    uint  segmentStart;  // base of this path's tile-local segment data
    uint  segmentCount;  // unused in tile mode (kept for layout stability)
    uint  fillRule;      // 0 = non-zero, 1 = even-odd
    uint  _r0;
    float tileW;     // tile width in physical px (16, coarsened for huge paths)
    float _r2;
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
layout(location = 3) flat out vec4 fragTile;       // shapeInfo passthrough
layout(location = 4) flat out vec2 fragTileOrigin; // tile top-left (px)

void main() {
    fragPos = inPosition;
    fragColor = inColor;
    fragGradientInfo = inGradientInfo;
    fragTile = inShapeInfo;
    fragTileOrigin = inUV;
    // Pixel → Vulkan clip space (Y-down), matching the rest of jvk.
    vec2 ndc = (inPosition / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

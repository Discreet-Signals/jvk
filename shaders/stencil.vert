#version 450

// Applied before NDC conversion. For non-cached paths the verts are already
// in local space; for cached paths the same transform turns the cached local
// mesh into screen coords. Packed as three vec2 "columns": pos = c0*x + c1*y + c2.
layout(push_constant) uniform PC {
    vec2 viewportSize;
    vec2 xformCol0;  // (m00, m10)
    vec2 xformCol1;  // (m01, m11)
    vec2 xformCol2;  // (m02, m12) — translation
} pc;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inShapeInfo;
layout(location = 4) in vec4 inGradientInfo;

void main() {
    vec2 world = pc.xformCol0 * inPosition.x
               + pc.xformCol1 * inPosition.y
               + pc.xformCol2;
    vec2 ndc = (world / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

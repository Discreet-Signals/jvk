#version 450

// Clip-stencil vertex shader. Passes a cover quad through in physical
// pixels so the fragment shader can evaluate the clip SDF per fragment.
// Color / UV / gradient attributes of UIVertex are ignored here — the
// pipeline declares only `inPosition`.

layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  center;           // rrect center (physical px) — unused for path
    vec2  halfSize;         // rrect half-extent (physical px) — unused for path
    float cornerRadius;     // rrect corner radius (physical px)
    uint  shapeType;        // 1 = rrect, 2 = path
    uint  segmentStart;     // path: first segment index in SSBO
    uint  segmentCount;     // path: segment count
    uint  fillRule;         // path: 0 = non-zero, 1 = even-odd
    uint  _pad;
} pc;

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec2 fragPos;

void main() {
    fragPos = inPosition;
    vec2 ndc = (inPosition / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

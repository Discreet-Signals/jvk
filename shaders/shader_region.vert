#version 450

layout(push_constant) uniform PC {
    float resolutionX;  // region width (pixels) — shared with fragment
    float resolutionY;  // region height (pixels) — shared with fragment
    float time;         // elapsed time — shared with fragment
    float viewportW;    // viewport width (pixels) — vertex only
    float viewportH;    // viewport height (pixels) — vertex only
    float regionX;      // region top-left X (pixels) — vertex only
    float regionY;      // region top-left Y (pixels) — vertex only
} pc;

layout(location = 0) out vec2 fragUV;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV = uv;
    vec2 pos = vec2(pc.regionX, pc.regionY) + uv * vec2(pc.resolutionX, pc.resolutionY);
    vec2 ndc = (pos / vec2(pc.viewportW, pc.viewportH)) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

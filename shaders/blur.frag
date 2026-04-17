#version 450

// Separable Gaussian blur pass. Run twice — once horizontal, once vertical —
// with ping-pong framebuffers to get a 2D blur.
//
// viewportSize: extent of the source image (pixels). Used to derive texel size.
// direction:    (1,0) for horizontal pass, (0,1) for vertical.
// radius:       blur radius in pixels. 0 = pass-through (all taps at same UV).
layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  direction;
    float radius;
} pc;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

const float weights[9] = float[](
    0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05
);

void main() {
    vec2 texelSize = 1.0 / pc.viewportSize;
    vec4 color = vec4(0.0);
    for (int i = -4; i <= 4; i++) {
        vec2 offset = pc.direction * float(i) * pc.radius * texelSize;
        color += texture(srcTexture, fragUV + offset) * weights[i + 4];
    }
    outColor = color;
}

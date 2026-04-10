#version 450

layout(push_constant) uniform PC {
    vec2 viewportSize;
} pc;

layout(binding = 0) uniform sampler2D srcTexture;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec4 fragShapeInfo;

layout(location = 0) out vec4 outColor;

const float weights[9] = float[](
    0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05
);

void main() {
    vec2 texelSize = 1.0 / pc.viewportSize;
    vec2 direction = fragColor.xy;
    float radius = fragColor.z;
    vec2 uv = gl_FragCoord.xy * texelSize;

    vec4 color = vec4(0.0);
    for (int i = -4; i <= 4; i++) {
        vec2 offset = direction * float(i) * radius * texelSize;
        color += texture(srcTexture, uv + offset) * weights[i + 4];
    }
    outColor = color;
}

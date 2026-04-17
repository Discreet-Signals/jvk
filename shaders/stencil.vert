#version 450

layout(push_constant) uniform PC {
    vec2 viewportSize;
} pc;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inShapeInfo;
layout(location = 4) in vec4 inGradientInfo;

void main() {
    vec2 ndc = (inPosition / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

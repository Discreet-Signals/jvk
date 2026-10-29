#version 450
#include "jvk_portable.glsl"

// Shared with ui2d.frag and noise.frag (byte-identical view).
JVK_CONSTANTS {
    vec2 viewportSize;
    int  paintsMain;   // read by ui2d.frag
    int  _pad;
} pc;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inShapeInfo;
layout(location = 4) in vec4 inGradientInfo;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec4 fragShapeInfo;
layout(location = 3) out vec4 fragGradientInfo;

void main() {
    vec2 ndc = (inPosition / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    fragColor = inColor;
    fragUV = inUV;
    fragShapeInfo = inShapeInfo;
    fragGradientInfo = inGradientInfo;
}

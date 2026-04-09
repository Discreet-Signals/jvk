#version 450

layout(location = 0) in vec4 fragColor;  // Input color from vertex shader
layout(location = 0) out vec4 outColor;  // Output color

void main() {
    outColor = vec4((vec3(1.0, 1.0, 1.0) - fragColor.rgb), fragColor.a);
}

#version 450

layout(location = 0) in vec3 inPosition; // Vertex position
layout(location = 1) in vec4 inColor;    // Vertex color

layout(location = 0) out vec4 fragColor; // Outputted color for the fragment shader

void main() {
    gl_Position = vec4(inPosition, 1.0);
    fragColor = inColor;
}

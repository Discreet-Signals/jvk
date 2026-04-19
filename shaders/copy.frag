#version 450

// Trivial pass-through shader. Samples one input and writes it verbatim.
// Used by Renderer::execute as a pre-copy step before a clipped effect:
// the destination ping-pong buffer is filled with the source's contents
// so that when the subsequent effect pass discards outside-clip fragments
// (via stencil test), those destination pixels retain the source instead
// of undefined data.
//
// Push constants match blur.frag's layout — dirX/dirY/radius are ignored,
// reused as padding. Vertex shader is shared (blur.vert).

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(push_constant) uniform PC {
    vec2  viewportSize;  // unused (UV already normalised)
    vec2  direction;     // unused
    float radius;        // unused
} pc;

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(srcTexture, fragUV);
}

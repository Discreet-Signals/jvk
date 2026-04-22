#version 450

// Trivial pass-through shader. Samples one input and writes it verbatim.
// Used by Renderer::execute as the ping-pong seed-copy prologue that runs
// before every effect dispatch: the destination half is filled with the
// current scene contents so the effect only has to write its output
// region (tight bbox, stencil-clipped band, etc.) — pixels the effect
// doesn't touch retain the seeded source. Makes every ping-pong effect
// pipeline (blur, HSV, future kernel / convolution passes) safe to scope
// tightly without turning outside-write regions into stale data.
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

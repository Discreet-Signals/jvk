#version 450
#include "jvk_portable.glsl"

// Clip-stencil vertex shader. Passes a cover quad through in physical
// pixels so the fragment shader can evaluate the clip SDF per fragment.
// Color / UV / gradient attributes of UIVertex are ignored here — the
// pipeline declares only `inPosition`.

// The prefix every fragment stage it pairs with shares (clip.frag's block,
// clip_mask.frag's): this stage reads the viewport alone.
JVK_CONSTANTS {
    vec2  viewportSize;
} pc;

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec2 fragPos;

void main() {
    fragPos = inPosition;
    vec2 ndc = (inPosition / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}

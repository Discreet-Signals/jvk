#version 450
#include "jvk_portable.glsl"

// Separable Gaussian blur pass. Run twice — once horizontal, once vertical —
// with ping-pong framebuffers to get a 2D blur.
//
// The kernel arrives made (Renderer: gaussian()): `taps` holds tapCount
// (offset, weight) pairs, the centre first, every other one read at ±offset —
// a true Gaussian of sigma = radius / 2, normalised, each pair of integer
// texels collapsed to one bilinear tap between them (the linear-sampling
// trick). A tap costs two texture reads and a multiply-add: no exp() per tap
// per pixel.
//
// With no taps (tapCount 0) the pass is one bilinear read: the steps of a
// blur pyramid, down (a 2x2 box) and back up. A fragment reads its source at
// its own centre times `scale` (source px per destination px: 1 across, 2
// down, 1 / 2^levels up), exact at any size, odd ones included.
//
// viewportSize: extent of the SOURCE image (pixels).
// direction:    (1,0) for horizontal pass, (0,1) for vertical.
JVK_CONSTANTS {
    vec2  viewportSize;
    vec2  direction;
    float scale;
    uint  tapCount;
} pc;

JVK_TEXTURE(0, srcTexture);
JVK_SAMPLER(0, srcSampler);

JVK_STORAGE(0) Taps {
    vec2 data[];
} taps;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 texelSize = 1.0 / pc.viewportSize;
    vec2 uv = gl_FragCoord.xy * pc.scale * texelSize;
    vec4 color = texture(sampler2D(srcTexture, srcSampler), uv);
    if (pc.tapCount == 0u) { outColor = color; return; }

    color *= taps.data[0].y;
    for (uint i = 1u; i < pc.tapCount; ++i) {
        vec2 t = taps.data[i];
        vec2 off = pc.direction * t.x * texelSize;
        color += (texture(sampler2D(srcTexture, srcSampler), uv + off)
                + texture(sampler2D(srcTexture, srcSampler), uv - off)) * t.y;
    }
    outColor = color;
}
